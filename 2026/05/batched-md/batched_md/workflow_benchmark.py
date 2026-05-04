"""Multi-stage workflow benchmark: relax → MD with inflight batching.

A typical computational-chemistry pipeline: take a set of perturbed
structures, relax each to a force minimum (FIRE), then run a short
NVT MD trajectory on the relaxed geometry. With heterogeneous
convergence step counts across the FIRE phase, this is exactly the
workload inflight batching is designed for: as a system finishes both
phases, its slot is reclaimed and a fresh system is admitted.

Three runners:

* ``ase``       — unbatched baseline.
* ``torchsim`` — InFlightAutoBatcher (FIRE) → BinningAutoBatcher (NVT).
* ``alchemi``   — FusedStage([(0, FIRE2), (1, NVTLangevin)]) + SizeAwareSampler.

cueq is disabled everywhere so the comparison is about the dynamics
loop and the autobatcher mechanics, not cueq utilization. dtype=float32.
``torch.compile`` is intentionally not applied — getting it right with
FusedStage's sampler-driven mode (so that the compile cost lands in
warmup, not measurement) requires more plumbing than is justified for
a first pass; if ``compile`` becomes interesting later it should be a
separate experiment.

Run locally::

    uv run python -m batched_md.workflow_benchmark ase
    uv run python -m batched_md.workflow_benchmark torchsim
    uv run python -m batched_md.workflow_benchmark alchemi
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch_sim as ts
from ase import Atoms
from ase import units as ase_units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.optimize import FIRE
from mace.calculators import mace_mp
from nvalchemi.data import AtomicData
from nvalchemi.dynamics import (
    ConvergenceHook,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics import FIRE as AlchemiFire
from nvalchemi.dynamics.base import FusedStage
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
from torch_sim.models.mace import MaceModel, MaceUrls

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

MACE_CHECKPOINT = MaceUrls.mace_mpa_medium

# Workload sizing.
N_REQUESTS = 64
N_ATOMS = 64
LATTICE_A = 3.61
SUPERCELL = (2, 2, 4)
NOISE_LEVELS = [0.01, 0.03, 0.10, 0.30]  # Å — drives heterogeneous opt step counts

# Memory cap. Holds 32 systems per wave (= half of N_REQUESTS), so
# the workload spans 2 waves on torchsim's BinningAutoBatcher and
# triggers inflight refill on alchemi's SizeAwareSampler. Picking
# budget = N_REQUESTS × N_ATOMS would make the sampler exhaust on
# the first build and skip refill entirely, which would mask one of
# the design differences this benchmark is meant to compare.
MAX_ATOMS_BUDGET = 2048

# Phase 1: FIRE relaxation.
# Defaults match across ASE / torchsim / alchemi: vanilla Bitzek-2006 FIRE
# with maxstep=0.2, n_min=5, f_dec=0.5, f_inc=1.1, alpha_start=0.1, f_alpha=0.99.
# dt=0.1 is the ASE default; alchemi.dynamics.FIRE accepts the same scalar.
FORCE_TOL = 0.05  # eV/Å
MAX_OPT_STEPS = 200  # safety cap per system
FIRE_DT = 0.1

# Phase 2: NVT Langevin MD.
MD_STEPS = 100
MD_DT_FS = 1.0
TEMPERATURE_K = 300.0
FRICTION_INV_FS = 0.01

WARMUP_STEPS = 10


def synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class WorkflowRun:
    framework: str
    n_systems: int
    n_atoms_per_system: int
    max_atoms_budget: int
    wall_time_s: float
    # LAMMPS-style throughput against the *budgeted* workload:
    # n_systems × n_atoms × (MAX_OPT_STEPS + MD_STEPS) / wall_time.
    # The budget is the worst-case integration-step count per system
    # (opt converges faster than MAX_OPT_STEPS in practice). Frameworks
    # that finish the same workload faster show higher throughput here.
    atom_steps_per_second: float


def _make_systems() -> list[Atoms]:
    base = bulk("Cu", crystalstructure="fcc", a=LATTICE_A, cubic=True).repeat(SUPERCELL)
    out: list[Atoms] = []
    for i in range(N_REQUESTS):
        atoms = base.copy()
        sigma = NOISE_LEVELS[i % len(NOISE_LEVELS)]
        rng = torch.Generator().manual_seed(i)
        noise = torch.randn(len(atoms), 3, generator=rng).numpy() * sigma
        atoms.set_positions(atoms.get_positions() + noise)
        out.append(atoms)
    return out


def _record(framework: str, elapsed: float) -> WorkflowRun:
    budgeted_atom_steps = N_REQUESTS * N_ATOMS * (MAX_OPT_STEPS + MD_STEPS)
    return WorkflowRun(
        framework=framework,
        n_systems=N_REQUESTS,
        n_atoms_per_system=N_ATOMS,
        max_atoms_budget=MAX_ATOMS_BUDGET,
        wall_time_s=elapsed,
        atom_steps_per_second=budgeted_atom_steps / elapsed,
    )


def run_torchsim() -> WorkflowRun:
    systems = _make_systems()

    raw = mace_mp(
        model=MACE_CHECKPOINT,
        return_raw_model=True,
        default_dtype=str(DTYPE).removeprefix("torch."),
        device=str(DEVICE),
    )
    model = MaceModel(
        model=raw,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        enable_cueq=False,
    )

    convergence_fn = ts.generate_force_convergence_fn(force_tol=FORCE_TOL)

    # Warmup: short opt + short MD on a single system.
    ts.optimize(
        system=systems[:1],
        model=model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=convergence_fn,
        max_steps=WARMUP_STEPS,
        autobatcher=False,
    )
    ts.integrate(
        system=systems[:1],
        model=model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=WARMUP_STEPS,
        timestep=MD_DT_FS * 1e-3,
        temperature=TEMPERATURE_K,
        autobatcher=False,
    )

    synchronize()
    t0 = time.perf_counter()
    relaxed = ts.optimize(
        system=systems,
        model=model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=convergence_fn,
        max_steps=MAX_OPT_STEPS,
        autobatcher=InFlightAutoBatcher(
            model=model,
            memory_scales_with="n_atoms",
            max_memory_scaler=MAX_ATOMS_BUDGET,
            max_iterations=MAX_OPT_STEPS,
        ),
    )
    ts.integrate(
        system=relaxed,
        model=model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=MD_STEPS,
        timestep=MD_DT_FS * 1e-3,
        temperature=TEMPERATURE_K,
        autobatcher=BinningAutoBatcher(
            model=model,
            memory_scales_with="n_atoms",
            max_memory_scaler=MAX_ATOMS_BUDGET,
        ),
    )
    synchronize()
    elapsed = time.perf_counter() - t0

    return _record("torchsim", elapsed)


def _atoms_to_atomic_data(atoms: Atoms) -> AtomicData:
    n = len(atoms)
    return AtomicData(
        positions=torch.as_tensor(atoms.positions, dtype=DTYPE, device=DEVICE),
        atomic_numbers=torch.as_tensor(atoms.numbers, dtype=torch.long, device=DEVICE),
        atomic_masses=torch.as_tensor(atoms.get_masses(), dtype=DTYPE, device=DEVICE),
        cell=torch.as_tensor(atoms.cell.array, dtype=DTYPE, device=DEVICE).unsqueeze(0),
        pbc=torch.tensor(
            [[bool(p) for p in atoms.pbc]], dtype=torch.bool, device=DEVICE
        ),
        forces=torch.zeros(n, 3, dtype=DTYPE, device=DEVICE),
        energy=torch.zeros(1, 1, dtype=DTYPE, device=DEVICE),
        velocities=torch.zeros(n, 3, dtype=DTYPE, device=DEVICE),
    )


class _AtomsDataset:
    def __init__(self, atoms_list: list[Atoms]) -> None:
        self._atoms = atoms_list
        self._n_atoms = len(atoms_list[0]) if atoms_list else 0

    def __len__(self) -> int:
        return len(self._atoms)

    def get_metadata(self, idx: int) -> tuple[int, int]:
        return self._n_atoms, 0

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        return _atoms_to_atomic_data(self._atoms[idx]), {}


def _build_alchemi_fused(model, atoms_list: list[Atoms]) -> FusedStage:
    ds = _AtomsDataset(atoms_list)
    sampler = SizeAwareSampler(
        dataset=ds,
        max_atoms=MAX_ATOMS_BUDGET,
        max_edges=None,
        max_batch_size=None,
    )
    fire = AlchemiFire(
        model=model,
        dt=FIRE_DT,
        n_steps=MAX_OPT_STEPS,
        convergence_hook=ConvergenceHook.from_fmax(threshold=FORCE_TOL),
    )
    nvt = NVTLangevin(
        model=model,
        dt=MD_DT_FS,
        temperature=TEMPERATURE_K,
        friction=FRICTION_INV_FS,
        n_steps=MD_STEPS,
    )
    # refill_frequency is left at its default (= 1), which is the natural
    # setting for inflight: detect graduations as soon as they happen and
    # backfill the slot. The cost of the per-step check is documented as
    # negligible when no system is graduating (early return in
    # refill_check). Setting this explicitly would be untested territory
    # for inflight workloads.
    fused = FusedStage(sub_stages=[(0, fire), (1, nvt)], sampler=sampler)
    for hook in model.make_neighbor_hooks():
        fused.register_hook(hook)
    return fused


def run_alchemi() -> WorkflowRun:
    from nvalchemi.models.mace import MACEWrapper

    systems = _make_systems()

    model = MACEWrapper.from_checkpoint(
        MACE_CHECKPOINT, device=DEVICE, dtype=DTYPE, enable_cueq=False
    )
    model.set_config("active_outputs", {"energy", "forces"})

    # Warmup on a single system to amortize CUDA kernel JIT, allocator
    # priming, and cueq lazy-init.
    warm = _build_alchemi_fused(model, systems[:1])
    warm.run(batch=None, n_steps=WARMUP_STEPS * 2)

    fused = _build_alchemi_fused(model, systems)

    n_steps_cap = (MAX_OPT_STEPS + MD_STEPS) * (N_REQUESTS + 1)

    synchronize()
    t0 = time.perf_counter()
    with fused:
        fused.run(batch=None, n_steps=n_steps_cap)
    synchronize()
    elapsed = time.perf_counter() - t0

    return _record("alchemi", elapsed)


def run_ase() -> WorkflowRun:
    systems = _make_systems()

    calc = mace_mp(
        model=MACE_CHECKPOINT,
        device=str(DEVICE),
        default_dtype=str(DTYPE).removeprefix("torch."),
        enable_cueq=False,
    )
    # ASE Langevin expects friction in inverse ASE-internal time units.
    ase_friction = FRICTION_INV_FS / ase_units.fs

    def run_one(atoms: Atoms) -> None:
        a = atoms.copy()
        a.calc = calc
        FIRE(a, logfile=None).run(fmax=FORCE_TOL, steps=MAX_OPT_STEPS)
        Langevin(
            a,
            timestep=MD_DT_FS * ase_units.fs,
            temperature_K=TEMPERATURE_K,
            friction=ase_friction,
        ).run(MD_STEPS)

    run_one(systems[0])

    synchronize()
    t0 = time.perf_counter()
    for atoms in systems:
        run_one(atoms)
    synchronize()
    elapsed = time.perf_counter() - t0
    return _record("ase", elapsed)


RUNNERS = {
    "ase": run_ase,
    "torchsim": run_torchsim,
    "alchemi": run_alchemi,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=sorted(RUNNERS))
    args = parser.parse_args()
    print(f"[{args.framework}]")
    run = RUNNERS[args.framework]()
    print(f"  {run}")
    out = (
        Path(__file__).resolve().parent.parent
        / "results"
        / f"workflow-{args.framework}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(run), indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
