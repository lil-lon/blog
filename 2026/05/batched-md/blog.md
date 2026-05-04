# Client-side batched MD with MLIPs: torch-sim and ALCHEMI Toolkit

## TL;DR

Both libraries answer the same question — saturating a GPU with GNN-MLIPs by stacking *N* independent simulations into one forward pass — and they share most of the substrate: the same atom-axis batching abstraction (`batch_idx` / `system_idx`), the same `ModelConfig` / `ModelInterface` indirection over MACE / AIMNet2 / SevenNet, the same ensembles, the same FIRE relaxation primitives, and the same low-level kernel package (`nvalchemi-toolkit-ops`). The meaningful difference is *who owns the simulation loop*: in torch-sim it is a Python `for` you can rewrite verbatim; in ALCHEMI it is `BaseDynamics.run` and you customise via Hooks and Stage composition. That single decision propagates to where the neighbour list lives, how the autobatcher decides its budget, and how multi-stage / multi-GPU runs are expressed.

The choice is design fit:

- Pick **torch-sim** if you run on a single GPU, want to read and rewrite the loop, prefer empirical GPU calibration, and like a one-line driver API.
- Pick **ALCHEMI Toolkit** if you need multi-GPU pipelines, hook-based instrumentation, persistent neighbor lists with a Verlet skin, or declared-cap autobatching that cannot OOM.

## Why both libraries look so similar

Saturating a GPU during neural-network inference is generally hard. GNN-based machine-learned interatomic potentials (MLIPs) are no exception: a single forward pass on a 100-atom system touches only a small fraction of the FLOPs and memory bandwidth of a GPU, and running one system at a time leaves the device mostly idle. Molecular dynamics and geometry optimization are awkward in this regard: they are inherently sequential along the time / iteration axis, so a single trajectory's steps cannot be batched against each other. What *can* be batched is independent systems. **Client-side batched MD** stacks many independent simulations into a single forward pass and steps them in lockstep on the same device, recovering GPU efficiency without changing the underlying algorithm.

Two libraries that sit squarely in this niche, both released in the past year, are:

- [`torch-sim`](https://github.com/TorchSim/torch-sim) (`torch-sim-atomistic` on PyPI), from Radical AI.
- [`nvalchemi-toolkit`](https://github.com/NVIDIA/nvalchemi-toolkit), the Python frontend of NVIDIA ALCHEMI.

The shared substrate runs deep:

- **Atom-axis batching.** Both stack *N* systems on `dim=0` of an atom tensor. A per-atom `batch_idx` / `system_idx` identifies which system each atom belongs to; reductions (energy, kinetic energy, forces COM) are scatter operations gated by it. The pattern is `torch_geometric.data.Batch`.
- **MLIP wrapper interface.** Both expose a small `ModelConfig` / `ModelInterface` that declares which outputs (`energy`, `forces`, `stress`) the model can produce, what is computed via autograd, and the cutoff used by the neighbour-list builder. Wrapping MACE, AIMNet2, SevenNet, or your own potential is a matter of implementing this interface.
- **Conservative forces via autograd.** MACE wrappers in both libraries enable `requires_grad_` on positions in the input adapter and rely on MACE's internal autograd path; stress goes through the same displacement trick.
- **Same ensembles.** NVE, NVT (Langevin and Nosé–Hoover), NPT (Langevin / Nosé–Hoover and stochastic-cell-rescale variants).
- **Same relaxation primitives.** Both ship FIRE / FIRE2 with cell filters; torch-sim adds BFGS / LBFGS, ALCHEMI adds variable-cell FIRE2.
- **Same low-level kernels.** Both depend on `nvalchemi-toolkit-ops` (NVIDIA Warp cell-list, thermostat utilities, dispersion / PME) for the fast paths.

At the level of what the GPU does on each step, the two libraries are doing close to the same work. The differentiation lives in how the surrounding simulation loop is expressed.

## Where they actually differ

| Axis | torch-sim | ALCHEMI Toolkit |
| --- | --- | --- |
| State container | Flat `SimState` dataclass; neighbour list lives inside model | Pydantic `AtomicData` → graph-structured `Batch` with explicit `neighbor_list` |
| Driver API | One function `ts.integrate(...)` | Instantiate `NVTLangevin(...)`, register hooks, call `.run(batch)` |
| Extensibility | Functional (`init_func`, `step_func` tuples) | Object-oriented + hooks (`BEFORE_STEP`, `BEFORE_COMPUTE`, `ON_CONVERGE`, …) |
| Neighbour list | Recomputed every step inside `model.forward` (`torchsim_nl`) | Cached in batch, refreshed by `NeighborListHook` (Warp cell-list, optional Verlet skin) |
| Batch sizing | `BinningAutoBatcher` (MD), `InFlightAutoBatcher` (optimize) | `SizeAwareSampler` for inflight batching, attached to any `BaseDynamics` |
| Stage fusion | n/a | `dyn_a + dyn_b` → `FusedStage` shares one forward pass between two integrators |
| Multi-GPU | Single-device by design | `DistributedPipeline` via the `\|` operator, with explicit rank topology |
| Trajectory I/O | `TrajectoryReporter` → torch-sim binary | `ZarrData` sink with CSR-style variable-size graph layout |

The deepest difference is *where the simulation loop lives*. Every row above is downstream of it.

### Loop ownership: a function you write or a hook-graph you compose

**torch-sim — a flat batched state and a functional `integrate()`.** The core data type is `SimState`, a `dataclass` of tensors:

```python
@dataclass(kw_only=True)
class SimState:
    positions: torch.Tensor       # (n_atoms, 3)
    masses: torch.Tensor          # (n_atoms,)
    cell: torch.Tensor            # (n_systems, 3, 3)
    pbc: torch.Tensor             # (3,) or per-system
    atomic_numbers: torch.Tensor  # (n_atoms,)
    system_idx: torch.Tensor      # (n_atoms,), non-decreasing
```

The neighbour list is *not* part of `SimState`; the model wrapper computes it on each forward pass. Running NVT Langevin on a list of ASE `Atoms` is one call:

```python
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from mace.calculators import mace_mp

mace = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32").models[0]
model = MaceModel(model=mace, device="cuda", dtype=torch.float32)

final_state = ts.integrate(
    system=systems,                       # list[ase.Atoms]
    model=model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=1_000, timestep=0.001, temperature=300.0,
)
```

`ts.integrate` is the entire public surface for MD. Internally it is the loop you would have written by hand: pick `(init_func, step_func)` from `INTEGRATOR_REGISTRY`, call `init_func(state, model, kT, dt)`, then `step_func(...)` `n_steps` times, optionally writing to a `TrajectoryReporter`. This is intentional — the design emphasises that an integrator is a pair of pure functions over a state, and a "batched" integrator is the same pair of functions written against a flat tensor with `system_idx`. To customise the loop, you read `runners.py` and rewrite it.

**ALCHEMI Toolkit — an explicit graph and an OO dynamics object.** The `nvalchemi.data` layer is closer to PyTorch Geometric than to a flat tensor blob. A single system is an `AtomicData` (Pydantic-validated, `jaxtyping`-annotated), and a batch is a `Batch` backed by a three-level tensor store:

- `atoms` — node-level (positions, velocities, forces, atomic_numbers, masses)
- `edges` — edge-level (`neighbor_list` `[E, 2]`, `neighbor_list_shifts` `[E, 3]`, `shifts`)
- `system` — graph-level (cell, pbc, energy, stress)

The neighbor list is a first-class field on the batch. It is not recomputed inside the model forward; it is refreshed by a hook that runs at the `BEFORE_COMPUTE` stage of every step.

Running NVT Langevin looks like this:

```python
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

model = MACEWrapper.from_checkpoint("medium-mpa-0", device="cuda", dtype=torch.float32)
batch = Batch.from_data_list(
    [AtomicData.from_atoms(a, device="cuda", dtype=torch.float32) for a in systems],
)
# (omitted: pre-allocate batch["forces"|"energy"|"velocities"] and seed velocities at 300 K)

dynamics = NVTLangevin(model=model, dt=1.0, temperature=300.0, friction=0.01, n_steps=1_000)
dynamics.register_hook(
    NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
)
dynamics.run(batch)
```

The integrator is an object (`NVTLangevin`, `NVTNoseHoover`, `NPT`, …) inheriting from `BaseDynamics`. The step loop is `pre_update → compute → post_update`, with hooks fired at each boundary — neighbour-list rebuild, biased-potential injection, periodic wrapping, custom convergence checks all attach as `Hook` objects rather than reading from the integrator. To customise the loop, you register a hook or compose stages.

The hook-based design is the thing that enables the rest of the toolkit:

- **Verlet skin / cached neighbour lists.** `NeighborListHook` keeps a persistent staging buffer and rebuilds only when atoms have drifted more than the skin. The cell-list path uses an O(N) NVIDIA Warp kernel from the companion `nvalchemi-toolkit-ops` package.
- **Inflight batching.** `SizeAwareSampler` plus `_CommunicationMixin.refill_check` graduate converged systems and pull in fresh ones mid-loop, sized by an atom/edge budget.
- **Pipeline composition.** `dyn_a + dyn_b` produces a `FusedStage` that runs both integrators on a single GPU sharing one model forward pass; `dyn_a | dyn_b` produces a `DistributedPipeline` that splits stages across ranks.

The split — Python `for` over pure functions vs. `BaseDynamics.run` with a hook lifecycle — propagates to everything below.

### Neighbour list: inside the model or on the batch

The two libraries put the neighbour list in different places, and that is a direct consequence of where the loop lives.

**torch-sim — built into `model.forward`.** `MaceModel.forward` calls its `neighbor_list_fn` (default `torchsim_nl`, the in-package cell-list) on every invocation:

```python
# torch_sim/models/mace.py
edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
    state.positions, state.row_vector_cell, state.pbc, self.r_max, state.system_idx,
)
```

There is no cache: every step pays a full NL build. The user does nothing — the list is computed inside the model and never appears at the integrator level.

**ALCHEMI — a hook on the batch.** The neighbour list is stored on the `Batch` (`batch.neighbor_list`, `batch.neighbor_list_shifts`). It is refreshed by `NeighborListHook` registered at `DynamicsStage.BEFORE_COMPUTE`. With `skin > 0` the hook calls `nvalchemiops`'s `batch_neighbor_list_needs_rebuild` to check, per system, whether any atom has drifted more than `skin / 2` since the previous build:

```python
# nvalchemi/hooks/neighbor_list.py
if self.skin > 0.0 and self._ref_positions is not None:
    _batch_nl_rebuild_inplace(
        reference_positions=self._ref_positions,
        current_positions=self._buf_positions,
        batch_idx=self._buf_batch_idx,
        rebuild_flags=self._rebuild_flags,
        skin_distance_threshold=self.skin / 2,
        ...
    )
```

When the flag is clear the previous list is reused; staging buffers (`_buf_positions`, `_buf_cell`, …) are persistent, so even when a rebuild does happen there are no per-step allocations.

Concretely:

| | torch-sim | ALCHEMI |
| --- | --- | --- |
| Where the NL lives | inside `model.forward`, transient | on the batch (`batch.neighbor_list`), persistent |
| Cost per step | one full cell-list build | skin check; full build only on drift |
| Setup the user pays | none (default) | `for hook in model.make_neighbor_hooks(): stage.register_hook(hook)` |
| Customisation | swap `neighbor_list_fn` on the model | construct `NeighborListHook(skin=…, max_neighbors=…)`; or write a custom hook |

For low-temperature MD with small `dt` the Verlet skin can cut NL cost dramatically (rebuilds every tens of steps instead of every step). For FIRE-style relaxation, high-T MD, or NPT with active cell motion the skin invalidates often and the savings shrink.

The hook-vs-model split also explains the boilerplate ALCHEMI asks you to write: a `NeighborListHook` is just one Hook among many (`WrapPeriodicHook`, `LoggingHook`, biased-potential hooks, custom NL implementations). Forcing explicit registration keeps that abstraction uniform — if NL were special-cased into the dynamics class, swapping in a custom NL builder or attaching multiple NLs (e.g. short-range MLIP + long-range Coulomb) would not compose. torch-sim makes the opposite call: NL inside the model wrapper, lower flexibility, less ceremony.

### Autobatch sizing: probe-and-bin or declared-and-stream

Both libraries face the same problem: the user submits a batch of independent systems but the GPU has fixed memory; the runtime has to decide how many systems to put through the model in one forward pass. Both ship an autobatcher to solve it. Their default behaviour is different in two independent ways: how the memory budget is *discovered*, and how the budget is *spent*.

**How the budget is discovered.**

- *torch-sim's default — empirical OOM probing.* When you call `ts.integrate(autobatcher=True)`, the constructed `BinningAutoBatcher` has `max_memory_scaler=None`. On the first `load_states(...)` call, `estimate_max_memory_scaler` replicates the smallest and largest input states geometrically (`scale_factor=1.6`) and runs forward passes until it catches a `CUDA out of memory` exception, then backs off two steps and uses that as the budget. One slow setup pass, calibrated to the actual GPU.
- *ALCHEMI's default — declared caps with a conservative GPU heuristic.* `SizeAwareSampler` requires the user to pass at least one of `max_atoms` / `max_batch_size` upfront. If `max_atoms` is omitted, an internal heuristic (`_estimate_max_atoms_from_gpu`) reads `torch.cuda.get_device_properties().total_memory`, multiplies by `max_gpu_memory_fraction=0.8`, subtracts a 20% model-overhead reserve, and divides by a fixed `300 bytes/atom`. The two combine via `min(user_cap, gpu_estimate)` — the more restrictive wins. No OOM is ever risked.

**What the budget is denominated in.**

| | torch-sim | ALCHEMI |
| --- | --- | --- |
| Budget | one scalar (`max_memory_scaler`) | three independent caps (`max_atoms`, `max_edges`, `max_batch_size`) |
| Metric for the scalar | choice of `n_atoms`, `n_atoms_x_density`, or `n_edges` | atoms and edges are separate caps; density is implicit in `max_atoms` |

The unit of `max_memory_scaler` depends on `memory_scales_with`. With `"n_atoms"` it is integer atoms; with `"n_atoms_x_density"` (the default) it is `atoms² / nm³`; with `"n_edges"` it is integer edges. The bin-pack constraint is just `sum(metric per system) ≤ max_memory_scaler`.

**How the budget is spent.**

- *torch-sim, phase-based.* `BinningAutoBatcher.load_states` runs a first-fit-decreasing bin-pack (`to_constant_volume_bins`) over the full input list, decides every bin upfront, and then iterates: each bin runs to completion (full `n_steps`) before the next bin starts. For optimization workloads where systems converge at different rates there is a separate `InFlightAutoBatcher` that hot-swaps converged systems out and pulls fresh ones in.
- *ALCHEMI, streaming.* `SizeAwareSampler` pre-scans the dataset's metadata (`get_metadata(idx) → (n_atoms, n_edges)`) without loading systems. `build_initial_batch` round-robins across atom-count bins to fill the live batch under the caps. As individual systems graduate (via the integrator's per-system `n_steps` budget or a `ConvergenceHook`), `request_replacement(num_atoms, num_edges)` finds an unconsumed dataset sample that fits the freed slot. The live batch stays on the GPU; only its membership rotates.

Both autobatchers can be driven declaratively, and that mode is the right one for repeatable production runs:

```python
# torch-sim — explicit max_memory_scaler skips the OOM probing entirely.
batcher = BinningAutoBatcher(model=model, memory_scales_with="n_atoms",
                             max_memory_scaler=4096)
ts.integrate(system=systems, model=model, integrator=..., autobatcher=batcher, ...)

# ALCHEMI — explicit max_atoms bypasses the GPU heuristic.
sampler = SizeAwareSampler(dataset=dataset, max_atoms=4096)
fused = FusedStage(
    sub_stages=[(0, NVTLangevin(model=model, dt=1.0, temperature=300.0, n_steps=...))],
    sampler=sampler,
)
fused.run(batch=None, n_steps=...)
```

At identical caps the two would issue the same number of forward passes for a uniform workload — torch-sim picks bin sizes upfront, ALCHEMI fills the live batch by streaming, but in both cases the steady-state batch size is `max_atoms_budget // n_atoms_per_system` and the number of waves is `ceil(N_requests / batch_size)`. The split is in *who pays the calibration cost*: torch-sim pays it once at startup with a probing pass; ALCHEMI pushes it onto the user as an explicit cap, with a heuristic fallback that errs conservative.

### Multi-stage and multi-GPU composition

A typical `relax → equilibrate → production` workflow can be expressed two ways:

```python
# torch-sim — chain of driver calls; thread `state` through.
state = ts.optimize(system=systems, model=model, optimizer=ts.Optimizer.fire, max_steps=...)
state = ts.integrate(system=state, model=model,
                     integrator=ts.Integrator.nvt_langevin, n_steps=..., temperature=300.0)
state = ts.integrate(system=state, model=model,
                     integrator=ts.Integrator.npt_nose_hoover_isotropic,
                     n_steps=..., temperature=300.0, external_pressure=0.0)

# ALCHEMI — `+` composes BaseDynamics objects into a FusedStage.
relax = FIRE2(model=model, dt=0.01, n_steps=...)
equil = NVTLangevin(model=model, dt=1.0, temperature=300.0, friction=0.01, n_steps=...)
prod  = NPT(model=model, dt=1.0, temperature=300.0, pressure=0.0, n_steps=...)
fused = relax + equil + prod
fused.run(batch, n_steps=...)
```

The torch-sim form is a chain of driver calls. The ALCHEMI form is a single `FusedStage` object: one forward pass per step, masked across sub-stages by per-system status; systems migrate to the next stage once their per-system `n_steps` is consumed (or a convergence hook fires).

ALCHEMI also has a `|` operator that builds a `DistributedPipeline`:

```python
# `|` builds a DistributedPipeline. Each stage runs on its own rank
# (one process, one GPU per rank). Systems stream downstream over NCCL.
pipeline = relax | equil | prod
pipeline.run(batch)   # launched via `torchrun --nproc-per-node=N`
```

torch-sim is single-device, so multi-process coordination across GPUs would be on the user.

For uniform workloads `+` and `|` are roughly equivalent in throughput to data parallelism — both spread the same total compute across GPUs. The case where these operators are more than syntactic convenience is *heterogeneous* workloads (different per-stage costs, different models per stage) where stages can be sized independently and converged systems stream forward without manual data movement.

## Closing — when to pick which

The two libraries share the atom-axis batching abstraction (`batch_idx` / `system_idx`), the MLIP wrapper interface, the ensembles, and the low-level kernels. They diverge on who owns the simulation loop, and four observable consequences follow: where the neighbour list lives, how the autobatcher decides its budget, how multi-stage and multi-GPU runs are composed, and whether the loop itself is something you read or something you hook into.

Pick **torch-sim** if:

- You run on a single GPU and the loop is the primary thing you'd want to customise — you would rather rewrite `runners.py` than register a hook.
- Empirical OOM-probing autobatch (no caps to tune, calibrates per GPU) fits how you deploy.
- Functional `(init_func, step_func)` tuples and a one-line driver API match how you think about integrators.

Pick **ALCHEMI Toolkit** if:

- You need multi-GPU pipelines (`|`) or fused single-GPU multi-stage runs (`+`) without writing the orchestration yourself.
- You want hook-based instrumentation — biased potentials, custom convergence, periodic logging — without subclassing the integrator.
- A neighbour list with a Verlet skin matters for your workload (low-T MD, small `dt`).
- Declared-cap autobatching that cannot OOM is a hard requirement (e.g. shared-cluster jobs).

Sources:

- [torch-sim repository](https://github.com/TorchSim/torch-sim)
- [TorchSim documentation](https://radical-ai.github.io/torch-sim/)
- [NVIDIA ALCHEMI Toolkit on GitHub](https://github.com/NVIDIA/nvalchemi-toolkit)
- [ALCHEMI Toolkit documentation](https://nvidia.github.io/nvalchemi-toolkit/)
- [ALCHEMI Toolkit-Ops on GitHub](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
