# A missing S5 switch in the DFT-D3 CN-chain force

> Notes on a bug I found in the DFT-D3 implementation of NVIDIA/nvalchemi-toolkit-ops ([PR #111](https://github.com/NVIDIA/nvalchemi-toolkit-ops/pull/111)). This post walks through how D3 is computed in that codebase and where the bug was, with the math tied directly to the implementation. All formulas and variable names correspond to `nvalchemiops/interactions/dispersion/_dftd3.py`.

DFT-D3 is a semi-empirical London dispersion correction added on top of a DFT energy. The two-body part is a damped $-C_6/r^6 - C_8/r^8$ pair sum whose coefficients depend on the local atomic environment through a coordination number.

## Background

While validating output values, I noticed that force and stress (virial) drifted slightly, but only when the energy-side cutoff smoothing (the S5 switch) was enabled. The cause: the coordination-number (CN) contribution to the D3 force was missing the smoothing factor $s_w$ that the rest of the force already carried. The energy and the direct force included $s_w$, but the CN-chain term (`dE/dCN`) did not.

When smoothing is disabled (the default), $s_w \equiv 1$, so there was no effect. The bug only surfaced once S5 smoothing was turned on.

## Structure of the D3(BJ) energy

With Becke-Johnson (BJ) damping the two-body dispersion energy is

$$
E_{\mathrm{disp}} = -\frac{1}{2}\sum_{i}\sum_{j\neq i} s_w(r_{ij})
\left[
\frac{s_6 C_6^{ij}}{r_{ij}^{6} + (R_0^{ij})^{6}} +
\frac{s_8 C_8^{ij}}{r_{ij}^{8} + (R_0^{ij})^{8}}
\right]
$$

Here I write the BJ damping length as $R_0^{ij}$. It corresponds to the $f_{\mathrm{damp}}$ in the nvalchemi official documentation, defined in section 3 below.

The $\tfrac12$ cancels the double counting of the ordered pairs $(i,j)$ and $(j,i)$. In the code each directed edge is accumulated and the final write applies `0.5 * sum_energy`. From here on we focus on a single pair $ij$ and drop the $ij$ subscripts.

### 1. Coordination number (Pass 1)

The CN is a smooth (non-integer) count of how many atoms sit within bonding distance:

$$
\mathrm{CN}_i = \sum_{k\neq i}
\frac{1}{1 + \exp\!\big(-k_1 ( (R^{\mathrm{cov}}_i + R^{\mathrm{cov}}_k)/r_{ik} - 1)\big)}
$$

$R^{\mathrm{cov}}$ is the covalent radius and $k_1$ the steepness (default $16$). The distance derivative used later is, in the implementation (`_cn_counting`),

$$
f \equiv \frac{1}{1+\exp(-k_1(R^{\mathrm{cov}}/r - 1))},\qquad
\frac{\partial f}{\partial r} = - f(1-f) k_1 \frac{R^{\mathrm{cov}}}{r^{2}}
$$

with $R^{\mathrm{cov}} = R^{\mathrm{cov}}_i + R^{\mathrm{cov}}_k$.

### 2. CN dependence of $C_6$ (Gaussian interpolation)

$C_6$ depends on the environment through CN. The implementation (`_c6ab_interpolate`) takes a Gaussian-weighted average over a $5\times5$ reference grid defined per element pair:

$$
L_{pq} = \exp\!\Big(k_3\big[(\mathrm{CN}_i - \mathrm{CN}^{\mathrm{ref}}_{i,pq})^2 + (\mathrm{CN}_j - \mathrm{CN}^{\mathrm{ref}}_{j,pq})^2\big]\Big),
\qquad k_3 = -4
$$

$$
C_6^{ij} = \frac{\sum_{pq} C_{6,pq}^{\mathrm{ref}} L_{pq}}{\sum_{pq} L_{pq}}
$$

Because $k_3 < 0$, $L_{pq}$ is a Gaussian centered on the reference point $(\mathrm{CN}^{\mathrm{ref}}_{i},\mathrm{CN}^{\mathrm{ref}}_{j})$ (the code stabilizes this with a log-sum-exp). With $d_i \equiv \mathrm{CN}_i - \mathrm{CN}^{\mathrm{ref}}_{i}$ and the weight sums

$$
w = \sum_{pq} L_{pq},\quad
w_{d_i} = \sum_{pq} L_{pq} d_i,\quad
z_{d_i} = \sum_{pq} C_{6,pq}^{\mathrm{ref}} L_{pq} d_i
$$

the CN derivative is

$$
\frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i} = \frac{2 k_3}{w}\big(z_{d_i} - C_6^{ij} w_{d_i}\big)
$$

(the `dC6_dCNi` in the code), and similarly for $\mathrm{CN}_j$. This is where $C_6$ acquires a dependence on atomic coordinates through CN, which is what produces the CN-chain force term below.

### 3. BJ damping, $R_0$, and why $C_8/C_6$ is a constant

The implementation (`_bj_damping`) treats the per-pair ratio

$$
\frac{C_8^{ij}}{C_6^{ij}} = 3 \mathrm{r4r2}_i \mathrm{r4r2}_j \;\equiv\; \mathrm{r4r2}_{ij}
$$

as a constant. Here $\mathrm{r4r2}_A$ is a **per-element parameter from the D3 reference implementation** (the code simply reads the Fortran `r2r4` array as `r4r2`), used only to build $C_8/C_6 = 3 \mathrm{r4r2}_i \mathrm{r4r2}_j$. Since $\mathrm{r4r2}_A$ depends only on the element, not on distance or CN, the ratio $C_8/C_6$ factors out as a per-pair constant.

> Physically this corresponds to the D3 relation $C_8 = 3 C_6\sqrt{Q_i Q_j}$, where $Q_A$ is an atomic quantity built from $\langle r^4\rangle/\langle r^2\rangle$ expectation values and the nuclear charge. For the exact definition and normalization, see the original D3 paper or `dftd3.f`. All this post needs is that the quantity is a per-element constant.

The BJ damping radius is

$$
R_0^{ij} = a_1\sqrt{C_8^{ij}/C_6^{ij}} + a_2
$$

- $a_1$ (dimensionless) and $a_2$ (length) are functional-dependent BJ parameters.
- $\sqrt{C_8/C_6}$ has units of length ($C_8/C_6 \sim \text{length}^2$), so $R_0$ is a length.
- $R_0$ is the effective radius at which damping turns on. For $r \ll R_0$ the denominator saturates at $R_0^{6,8}$, which removes the short-range $1/r^6$ divergence.

Writing the implementation variables `damp_6` and `damp_8` as

$$
\mathrm{damp}_6 = \frac{s_6}{r^6 + R_0^6},\qquad
\mathrm{damp}_8 = \frac{s_8 (C_8/C_6)}{r^8 + R_0^8},\qquad
\mathrm{damp}(r) \equiv \mathrm{damp}_6 + \mathrm{damp}_8
$$

the unsmoothed single-pair energy becomes

$$
E = - C_6 \mathrm{damp}(r)
$$

since $C_6\cdot\mathrm{damp}_8 = s_8 C_8/(r^8+R_0^8)$, which matches the original sum. Note that `damp_6` and `damp_8` are not the bare BJ denominators but the full contributions including the numerator scalings.

### 4. The S5 energy-side switch

To send the energy smoothly to zero near the cutoff, the implementation (`_s5_switch`) uses a quintic smoothstep. With $t = (r - r_{\mathrm{on}})/(r_{\mathrm{off}} - r_{\mathrm{on}})$:

$$
s_w(r) =
\begin{cases}
1 & r \le r_{\mathrm{on}} \\
1 - (10t^3 - 15t^4 + 6t^5) & r_{\mathrm{on}} < r < r_{\mathrm{off}} \\
0 & r \ge r_{\mathrm{off}}
\end{cases}
\qquad
\frac{d s_w}{dr} = \frac{-30t^2 + 60t^3 - 30t^4}{r_{\mathrm{off}} - r_{\mathrm{on}}}
$$

$s_w$ is $1$ at $r_{\mathrm{on}}$ and $0$ at $r_{\mathrm{off}}$, with continuous first and second derivatives ($C^2$) at both ends. The implementation disables smoothing when $r_{\mathrm{off}} \le r_{\mathrm{on}}$ ($s_w \equiv 1$), and the default is $r_{\mathrm{on}} = r_{\mathrm{off}} = 0$, so **smoothing is off by default**.

The smoothed single-pair energy is

$$
E^{\mathrm{sw}} = s_w(r) E = - s_w(r) C_6 \mathrm{damp}(r)
$$

## Forces: direct term and CN-chain term

Force is the negative coordinate gradient, $F_i = - \partial E_{\mathrm{disp}}/\partial \boldsymbol{x}_i$. The distance dependence of $E^{\mathrm{sw}} = -s_w C_6 \mathrm{damp}$ enters through two routes.

> The $\partial E/\partial r$ expressions below are derivatives with respect to the scalar distance. The conversion to a vector force follows the implementation's `r_hat` (unit direction vector) sign convention: the code distributes the result per atom and per component as `F = (dE/dr) * r_hat`, with the sign from projecting onto $\boldsymbol{x}_i$ absorbed into the choice of `r_hat` direction.

1. **Direct**: the explicit $r$ dependence in $s_w(r)$ and $\mathrm{damp}(r)$.
2. **Through CN**: $C_6$ depends on $\mathrm{CN}_i, \mathrm{CN}_j$, and each CN depends on all surrounding bond lengths (a many-body coupling).

### Direct term (Pass 2)

The derivative with respect to the pair's own distance $r_{ij}$, holding CN fixed. In `_dispersion_energy_force`:

$$
\left.\frac{\partial E^{\mathrm{sw}}}{\partial r}\right|_{\mathrm{direct}}
= s_w \frac{\partial E}{\partial r} + E \frac{d s_w}{dr}
= - C_6\big[ s_w \mathrm{damp}'(r) + s_w'(r) \mathrm{damp}(r) \big]
$$

(the code's `dE_dr_direct_sw = sw * dE_dr_direct + e_ij * dsw_dr`). At this point $s_w$ is correctly applied.

### CN-chain term (build `dE/dCN` in Pass 2, convert to force in Pass 3)

CN is a many-body quantity, so it cannot be reduced to a per-pair force. The implementation uses two passes.

**Pass 2**: accumulate the per-atom quantity $\partial E_{\mathrm{disp}}/\partial \mathrm{CN}_i$. Since $E^{\mathrm{sw}} = s_w E$ and $s_w$ does not depend on CN,

$$
\frac{\partial E^{\mathrm{sw}}}{\partial \mathrm{CN}_i} = s_w \frac{\partial E}{\partial \mathrm{CN}_i}
= - s_w(r_{ij}) \mathrm{damp}(r_{ij}) \frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i}
$$

summed over pairs:

$$
\frac{\partial E_{\mathrm{disp}}}{\partial \mathrm{CN}_i} = \sum_{j\neq i}\Big(- s_w(r_{ij}) \mathrm{damp}(r_{ij}) \frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i}\Big)
$$

which is exactly the fixed code:

```python
dE_dCN_acc += -damp_sum * dC6_dCNi * sw
```

> **Why the $\tfrac12$ appears in the energy but not in `dE/dCN`**: the global energy is $\tfrac12$ times an ordered-pair sum, but the D3 pair quantity is symmetric, so the two ordered pairs $(i,j)$ and $(j,i)$ correspond to the same physical pair. The derivative with respect to $\mathrm{CN}_i$ picks up both, which pairs with the energy's $\tfrac12$ and cancels it. The implementation can therefore keep the one-sided row sum over $j$ directly as `dE_dCN[i]`.

**Pass 3**: convert to a force on bond $(i,k)$ through $\mathrm{CN}_i = \sum_k f(r_{ik})$. In `_cn_forces_contrib_kernel_matrix`:

$$
\left.\frac{\partial E_{\mathrm{disp}}}{\partial r_{ik}}\right|_{\mathrm{CN}}
= \Big(\frac{\partial E_{\mathrm{disp}}}{\partial \mathrm{CN}_i} + \frac{\partial E_{\mathrm{disp}}}{\partial \mathrm{CN}_k}\Big)\frac{\partial f(r_{ik})}{\partial r_{ik}}
$$

```python
dE_dr_chain = (dE_dCN_i + dE_dCN_j) * dCN_dr
F_chain = dE_dr_chain * r_hat
```

The key point: **$s_w$ does not appear in Pass 3**. Since $s_w$ is CN-independent, folding it once into `dE/dCN` in Pass 2 is sufficient (formally $\partial E^{\mathrm{sw}}/\partial \mathrm{CN} = s_w \partial E/\partial \mathrm{CN}$). The $\partial C_6/\partial r$ coupling cannot be closed within a single pair: it must be accumulated over all pairs (Pass 2) and then distributed to each bond through $\partial \mathrm{CN}/\partial r$ (Pass 3).

## The bug

From $E^{\mathrm{sw}} = s_w E$ and the CN-independence of $s_w$, the CN derivative must carry the same $s_w$:

$$
\frac{\partial E^{\mathrm{sw}}}{\partial \mathrm{CN}_i} = s_w \frac{\partial E}{\partial \mathrm{CN}_i}
$$

The original code accumulated `dE/dCN` **without** $s_w$:

```python
# buggy (before)
dE_dCN_acc += -damp_sum * dC6_dCNi        # sw is missing
# fixed (after)
dE_dCN_acc += -damp_sum * dC6_dCNi * sw
```

As a result, when smoothing was active ($s_w < 1$), the CN-chain force (and the corresponding virial) reconstructed in Pass 3 was **not the exact gradient of the smoothed energy** $E^{\mathrm{sw}}$. The direct term carried $s_w$ while the CN-chain term did not, an inconsistency whose error scales as $(1 - s_w)$. The fix multiplies `dE/dCN` by $s_w$ in all four Pass-2 kernels (neighbor-matrix and neighbor-list, each in plain and virial variants).

### Why the default was unaffected

The default disables smoothing ($r_{\mathrm{on}} = r_{\mathrm{off}} = 0 \Rightarrow s_w \equiv 1$). The missing factor was $1$, so results were unchanged. The bug only manifests when S5 smoothing is explicitly enabled, so users on the default configuration were never affected.

## Comparison with other implementations

For reference, [torch-dftd](https://github.com/pfnet-research/torch-dftd) computes forces with PyTorch autograd rather than the hand-derived analytic derivatives (direct plus CN-chain) used here. With autograd, once $E^{\mathrm{sw}} = s_w E$ is assembled, $s_w$ propagates to every path automatically, so this class of bug (a factor dropped on one specific term) is structurally hard to introduce. The trade-off is that the analytic approach here avoids autograd overhead, at the cost of having to maintain this consistency by hand. The finite-difference test added in the PR is the regression guard for exactly that.

## Reference materials
* alchemi toolkit ops d3 documentation: https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/components/dispersion.html
* VASP D3: https://vasp.at/wiki/DFT-D3
* torch-dftd: https://github.com/pfnet-research/torch-dftd
