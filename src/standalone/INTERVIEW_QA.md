# Technical Assignment — CUDA Matrix-Free Linear Elasticity Operator

**From:** Andrew Comerford, Solver Development Team, Luminary
**Due:** End of day, next Friday

## Objective

Write CUDA code in **C++17** that computes the **matrix-free action** of the small-strain linear-elasticity operator for a 3D unstructured finite element solver using tetrahedral elements.

Given a displacement field $u$, compute:

$$y = Ku$$

where $K$ is the (implicit, never assembled) global stiffness operator, defined weakly by the bilinear form:

$$a(v, u) = \int_\Omega \varepsilon(v) : \sigma(u) \, d\Omega$$

with:

$$\varepsilon(u) = \frac{1}{2}\left(\nabla u + \nabla u^T\right) \qquad \text{(small-strain tensor)}$$

$$\sigma(u) = \lambda \,\text{tr}\big(\varepsilon(u)\big)\, I + 2\mu\, \varepsilon(u) \qquad \text{(isotropic linear elastic stress)}$$

Here $\Omega$ is the physical solid domain represented by the finite element mesh, and $\varepsilon(v):\sigma(u)$ denotes the double-dot (Frobenius) contraction between strain and stress tensors.

$y$ is the global nodal vector obtained by **assembling** the contributions from all elements.

## Constraints

- **Do not** form or store an element stiffness matrix $K_e$ or the global stiffness matrix $K$.
- Use a tetrahedral element with **quadratic ($P_2$) displacement interpolation** and an **affine geometry mapping**.
  - Because the mapping is affine, the geometric Jacobian $J$ is **constant** within each element.
  - The shape-function gradients still **vary** across the element and must be evaluated at each quadrature point.
- The Lamé parameters $\lambda(x_q)$, $\mu(x_q)$ vary spatially and are provided **at the quadrature points**.

## Flexibility / What You May Assume

Any information about the mesh, geometry, basis functions, material, or solution variables may be passed into the kernel or exposed via lightweight accessors. You may assume the following are available:

- Physical-space shape-function gradients
- Quadrature weights
- Jacobian information

## Deliverable Expectations

- Consider and document any aspects of the finite element discretization or implementation you feel are important.
- Use code comments to flag issues you're aware of but chose not to address given time constraints.
- **Document and justify** your design choices.
- Describe (or include) how you would **validate** the implementation — tests are welcome but not required to run.
- You do **not** need to implement a full linear solver.

## Submission Requirements

| Requirement | Detail |
|---|---|
| Language | CUDA / C++17 |
| Validity | Must compile/analyze cleanly on [Compiler Explorer](https://godbolt.org/) |
| Execution | Does **not** need to run on a real mesh |
| Length | < 1000 lines |
| Line width | ≤ 80 characters |
| Style | Consistent, sensible formatting |

---

# Interview Q&A: CUDA matrix-free elasticity operator

Prep notes for walking through [`matrix_free_fea.cu`](matrix_free_fea.cu) (same physics as the split port under [`../cuda/`](../cuda/)).

**How to use this:** lead with structural FEM soundness. Treat CUDA as the mapping of that algorithm onto a GPU—not the main claim. Prefer short spoken answers; open the `.cu` and point at symbols when you explain.

---

## 1. Elevator: what did you build?

**Q: What is this file?**

A matrix-free evaluation of $y = Ku$ for **small-strain isotropic linear elasticity** on **P2 (10-node) tetrahedra** with **affine (linear) geometry**. $K$ is never assembled—element integrals are recomputed each time the operator is applied.

**Q: Is this a full FE solver?**

No. This artifact is the **Krylov matvec**: one application of the global stiffness operator. Explicitly out of scope here: Dirichlet/Neumann BCs, a linear solver, and mesh I/O. The mesh is assumed already resident on the device as a flattened `MeshView`.

(The broader repo’s CPU path adds Dirichlet lifting, CG / Jacobi PCG, and closed-form BVPs. If asked “how would you solve a BVP?”, point there—then come back to the operator.)

**Q: Why matrix-free?**

An assembled sparse $K$ for P2 tets is large; SpMV is often memory-bound. Matrix-free keeps the working set near the vectors $u$ and $y$, and spends flops recomputing geometry/quadrature. On GPUs that tradeoff is often favorable. The cost is that you pay the element integral on **every** matvec (every CG iteration).

---

## 2. FEM / continuum mechanics (primary focus)

**Q: What physics and weak form?**

Small-strain isotropic linear elasticity. The bilinear form is

$$
a(v,u) = \int_\Omega \varepsilon(v):\sigma(u)\,d\Omega,
$$

with

$$
\varepsilon(u) = \operatorname{sym}(\nabla u), \qquad
\sigma = \lambda\,\operatorname{tr}(\varepsilon)\,I + 2\mu\,\varepsilon.
$$

Choosing $v = N_i\,e_d$ gives the nodal internal force that `ComputeElementOperator` accumulates.

**Q: Walk the element algorithm without forming $B$ or $K_e$.**

At each quadrature point (`ComputeElementOperator`):

1. Pull physical gradients $\nabla N_i = J^{-T}\,\partial N_i/\partial\xi$ (`PhysicalGradients`).
2. Build $\nabla u = \sum_i u_i \otimes \nabla N_i$ (same as $B u_e$, never explicit).
3. $\varepsilon = \operatorname{sym}(\nabla u)$, then $\sigma = \lambda\operatorname{tr}(\varepsilon)I + 2\mu\varepsilon$.
4. Accumulate $y_i \leftarrow y_i + w_q\,\det(J)\,\sigma\,\nabla N_i$ (same as $B^T\sigma$, never explicit).

$J$ is **constant** on the element (affine map from four corners) and is computed **once** before the QP loop. Shape-function gradients **vary** per QP because the displacement basis is quadratic.

**Q: Why P2 tets with affine geometry?**

- P2 (10 nodes) is a standard structural element (e.g. Abaqus C3D10-style ordering in the comments).
- Geometry uses only the **four corners** → linear map → cheap, constant $J$.
- Mid-edge nodes carry displacement DOFs only; they do not warp the geometry in this implementation (no isoparametric P2 geometry).

**Q: Why a 4-point Hammer–Stroud rule?**

Exact for polynomials of total degree $\le 2$. With P2 basis, $\nabla N$ is degree 1 in reference coordinates; products in the energy integrand are degree 2 when $\lambda,\mu$ are **pointwise** (not interpolated with higher-order polynomials). Weights include the reference-tet volume factor $1/6$, so each weight is $1/24$.

**Q: Why Lamé parameters per quadrature point?**

Keeps material evaluation outside the polynomial degree of the shape functions, so the 4-point rule stays justified. Spatially varying material is allowed without changing the quadrature argument.

**Q: Why double precision?**

Structural residual and energy checks are sensitive; assignment validation leans on near-exact identities (rigid modes, equilibrium). `float` is a deliberate later optimization, not the default for correctness work.

**Q: Orientation / `det(J)`?**

Assumed $\det(J) > 0$ (non-degenerate, right-handed ordering)—**not** checked at runtime. The `MeshView` comment shows a two-element patch where the second tet swaps corners so both elements keep positive Jacobian when apexes sit on opposite sides of a shared face.

**Q: Null space?**

Unconstrained $K$ is only **positive semi-definite**: six rigid-body modes (3 translations, 3 infinitesimal rotations). That is physics, not a bug. A real solve needs enough Dirichlet data to make the reduced system SPD.

---

## 3. Overall approach and algorithm shape

Per-element pipeline (one CUDA thread does all three phases):

```text
  gather                    apply                         scatter
  ───────────────────       ──────────────────────────    ─────────────────────
  corners + u_local    →    ComputeElementOperator    →   atomicAdd → y_global
  (via connectivity)        (quadrature; no K_e)          (shared nodes race)
```

**Q: How is the global operator structured?**

Same three phases as a textbook matrix-free assembly loop (and as the CPU `ApplyGlobalLinearElasticityOperator` in `assembly.hpp`):

1. **Gather** — connectivity → 4 corner coords + 10 nodal displacements into thread-local arrays.
2. **Apply** — `ComputeElementOperator` → `y_local` (never forms $K_e$).
3. **Scatter-add** — fold `y_local` into `y_global` at the same connectivity slots.

**Q: How does that map to CUDA?**

`GlobalOperatorKernel`: **one thread per element**. Thread index `elem = blockIdx.x * blockDim.x + threadIdx.x`; early return if `elem >= num_elements`. The file header states the intent explicitly: finer intra-element parallelism yields little benefit at this granularity; throughput and latency hiding come from **many concurrent element threads**.

**Q: Who zeros `y_global`?**

The **caller** (e.g. `cudaMemset`) before launch. Documented assumption: the kernel **accumulates** rather than assigns. Concurrent threads cannot each independently reset a shared global array.

**Q: What is `MeshView`?**

Device-resident flattened view — no ownership, no host pointers:

- `node_coords` — at least `num_nodes` entries
- `connectivity` — exactly `num_elements * 10` (row-major)
- `material` — exactly `num_elements * 4` (Lamé at each QP)
- sizing contract is assumed, not checked at runtime

The `MeshView` comment includes a two-element shared-face example (14 nodes) and explains why element 1 swaps two corners so both tets keep $\det(J) > 0$.

---

## 4. Implementation choices (from the `.cu` annotations)

Be ready to say: *I’m not pitching deep CUDA expertise. I chose a mapping that preserves the FEM algorithm one-to-one with the CPU gather / apply / scatter loop.*

These are the **design choices called out in the file header** of `matrix_free_fea.cu`:

| Choice | Annotation / justification | Tradeoff |
|---|---|---|
| Custom `Vector3` / `Matrix3` | Avoid fragile device support in libraries like Eigen; plain scalar members so the type lives in **registers** inside the kernel | Hand-rolled, deliberately narrow API |
| One thread / element | Each thread runs 4 QPs and 30 local DOFs serially | Fat threads; occupancy limited by live state |
| Block size 128 | Multiple of the 32-thread warp; kept modest because each thread carries ~30 DOFs of live state — a larger block would only cut occupancy further | Not autotuned |
| `atomicAdd` scatter | Concurrent writes to shared nodes; **mesh coloring considered but omitted** to avoid preprocessing complexity | Contention on high-valence nodes |
| Double throughout | Correctness / energy checks; scalar template is a planned TODO | Slower atomics; needs CC ≥ 6.0 for `atomicAdd` on `double` |
| `MFFEA_HOST_DEVICE` | Same math builds under a non-CUDA compiler so it can be exercised on the host | Dual-path macros |
| Closed-form $3\times3$ inverse | Adjugate formula once per element, not a factorization in a hot inner loop | Singular $J$ not checked — relies on $\det(J) > 0$ |

**Q: Why not finer parallelism inside an element?**

At this granularity (4 QPs, 30 local DOFs), splitting one element across a warp adds synchronization and little arithmetic intensity. Latency hiding comes from launching many element threads — as annotated in the file header.

**Q: Why atomics instead of mesh coloring?**

Coloring removes write conflicts (conflict-free colors → plain `+=`) but needs a real dual-graph coloring preprocess. The header is explicit: coloring was **considered and deferred** for time / complexity. It remains the first serious performance upgrade for large meshes.

**Q: Quadrature weight annotation?**

Weights already include the reference-tet volume $1/6$, so each of the four equal weights is $1/24$. The integral over the reference tet is $\sum_q w_q\, f(\xi_q)$ with **no further scaling**. Physical scaling enters only through $\det(J)$ in the accumulation.

---

## 5. Assumptions and intentional simplifications

Frame these as **scope / time-box**, not “I forgot.” Directly from the file header:

**Assumptions (no runtime checks):**

- $\det(J) > 0$ for every element (non-degenerate, correctly ordered)
- Connectivity indices are valid into `node_coords` / `u_global`
- `y_global` is zeroed by the caller before launch
- `MeshView` array sizes match `num_nodes` / `num_elements` as above
- All pointers are **device** memory, not host

**Explicitly out of scope in this file:**

- No linear solver — this is a Krylov matvec only
- No boundary-condition mechanism
- No mesh I/O — mesh expected already resident and flattened
- No degenerate-element handling

| Omission | Why it was OK for the assignment | With more time (header TODOs) |
|---|---|---|
| No `det(J)` check | Valid meshes only | Runtime degenerate-element detection |
| No connectivity bounds checks | Caller-owned mesh | Debug asserts / bounds checks |
| Device pointers only | Clear API ownership | Explicit upload helpers |
| No mesh I/O | Not the operator | Readers / generators |
| No BCs / solver here | Assignment does not require a full solve | Dirichlet + CG/PCG wrapping the matvec |
| Double only | Correctness first | Template on `float` / `double` (C++20 concepts noted) |
| No CPU–GPU parity harness | Strategy documented in the appendix | Differencing harness on identical `(mesh, u, material)` |

---

## 6. Testing and validation

### Strategy in the `.cu` appendix (assignment-facing)

The file documents black-box properties that should hold to machine epsilon or FP tolerance, **independent of mesh or material**:

1. **Rigid translation** → $y = 0$ ($\nabla u = 0$). Single-element check is enough.
2. **Infinitesimal rigid rotation** $u = \omega \times (x - x_{\mathrm{ref}})$ → $y = 0$. A **finite** rotation is nonlinear and leaves an $O(\theta^2)$ residual in small-strain theory — not a clean check. Use the skew velocity form so $\varepsilon = \operatorname{sym}(\nabla u)$ vanishes exactly.
3. **Self-equilibrium** $\sum_i y_i = 0$ for **any** $u$ and **any** (even spatially varying) material — partition of unity $\Rightarrow \sum_i \nabla N_i = 0$.
4. **PSD** — eigenvalues of assembled $K \ge 0$ (elastic strain energy $u\cdot(Ku)/2$ cannot be negative for a valid material).
5. **Energy consistency** — $u\cdot(Ku)$ vs an **independently** written quadrature of $\tfrac{1}{2}\sigma:\varepsilon$ that does **not** call `ComputeElementOperator`.
6. **Linear-field refinement invariance** — P2 reproduces linears exactly; strain energy identical across refinements (e.g. cube divisions 1, 2, 3).
7. **Analytical uniaxial stretch** — closed-form $U = \tfrac{1}{2}(\lambda+2\mu)\varepsilon_0^2 V$; strongest check because the reference shares **no code** with the operator under test.

**Recommended unit-test layout** (from the same appendix): `test_vector3_matrix3`, `test_quadrature`, `test_shape_functions`, `test_geometry`, `test_element_operator` (checks 1–3, 7), `test_assembly` (3–6), `test_kernel_launch` (same $y$ across block sizes 32/64/128/256 — physics must be invariant to work partitioning).

**Cantilever / Abaqus comparison:** stronger for bending, but needs BCs + solver → explicitly **out of scope** for this operator file. Say so clearly.

### What the repo actually validates today

Be transparent:

- **CPU** path: extensive GoogleTest (element kernel, assembly, validation, end-to-end BVP, CG/PCG) — same algorithm as this CUDA file.
- **CUDA** path: **not** in CMake; **no** automated CPU–GPU output differencing yet. The standalone header lists that harness as TODO.
- Honest line: *“Physics and operator identities are validated on the CPU reference implementation of the same algorithm. The next verification step for this kernel is a parity test: same mesh / $u$ / material on CPU and GPU, diff $y$ to tolerance. Also check invariance across block sizes.”*

---

## 7. Solver robustness (if the conversation expands past the matvec)

The assignment does **not** require a solver, but Andrew may ask how this operator plugs into a robust structural solve. The CPU path (`boundary_conditions.hpp`, `linear_solver.hpp`) is the concrete answer.

**Q: Why isn’t “call CG on $K$” enough?**

Unconstrained $K$ is only **positive semi-definite** (six rigid-body modes). CG needs an SPD operator. Robustness starts with **Dirichlet data that pins every rigid-body mode**, then solving the reduced free–free system $K_{FF}$.

**How the CPU path does that (matrix-free):**

1. Keep CG iterates **zero on constrained nodes** (never store the prescribed values in the unknown).
2. Matvec = apply full $K$, then **zero constrained rows** → that *is* $K_{FF}$ when the input is already zero on $\Gamma_D$.
3. Lift prescriptions **once** into the RHS: $b_F = f_F - K_{FC} u_D$ (`ComputeDirichletRHS`), not every iteration.
4. After CG, write prescribed values back (`ApplyPrescribedValues`).

Wrong patterns that break symmetry / SPD: feeding $u_D$ into the Krylov unknown, or replacing constrained residual rows with identity without also keeping search directions consistent.

**Q: What makes the *iterative* solve robust?**

| Concern | What we do | Why it matters |
|---|---|---|
| Singular / under-constrained $K$ | Enough Dirichlet; `pAp <= 0` **breakdown guard** in PCG | Avoids NaNs; returns `converged = false` with last good iterate |
| Stopping criterion | $\\|r\\| < \tau \max(\\|b\\|, 1)$ | Relative residual with a floor — zero RHS does not demand an impossible absolute tol |
| Ill-conditioning as $h \to 0$ | Jacobi diagonal of $K_{FF}$ (constrained entries set to 1) | Lowers iteration count when per-DOF stiffness varies; does **not** cancel $h^{-2}$ growth |
| Heterogeneous material | Same Jacobi from `ComputeGlobalDiagonal` | Clusters spectrum by scaling out local stiffness magnitude |
| Identity preconditioner | Must match plain CG iterate-for-iterate | Guards against a broken PCG recurrence |
| Residual monitoring | History of $\\|r\\|$ from the recurrence | Distinguishes slow convergence from breakdown |

**Spoken narrative you can use:**

> Robustness has two layers. At the continuum/discrete level, the unconstrained elasticity operator is SPSD — you must constrain rigid modes or CG is solving a singular system. At the algebraic level, PCG must refuse to divide by a non-positive $p^TAp$, stop on a relative residual with a sensible floor, and use a preconditioner that matches the *reduced* operator (including setting constrained Jacobi entries to 1 so $M^{-1}$ is a no-op on DOFs that stay zero). Jacobi helps; mesh-independent convergence needs multigrid-class methods later. None of that lives in the CUDA assignment file — the file is the matvec those solvers call.

**Q: Operator-level robustness (still in the CUDA file)?**

- Double precision for residual/energy identities.
- Exact quadrature for the polynomial degree you claim (avoids “soft” force imbalance from under-integration).
- Documented assumptions instead of silent NaNs from inverted elements — and a clear TODO to **check** $\det(J)$ in production.
- Validation that catches wrong $J^{-T}$ vs $J^{-1}$, missing transpose in $\varepsilon$, or broken scatter (equilibrium / energy / analytical stretch).

---

## 8. Performance considerations

**No unverified speedup claims.** Talk in mechanisms and what you’d measure.

**Why matrix-free can win (file overview):** trades extra compute for much lower memory traffic — favorable on bandwidth-limited GPUs. Working set is vectors, not the assembled sparse matrix.

**Bottlenecks you expect:**

- `atomicAdd` contention at shared nodes (especially high-valence vertices) — called out in the header as the coloring motivation
- Irregular gathers through connectivity (poor coalescing)
- Register pressure → limited occupancy (fat per-element state; reason block size stays modest)
- Double-precision atomics (CC ≥ 6.0)

**What you’d profile:** Nsight Compute — SM throughput, atomic replay, DRAM throughput, occupancy. Scale wall time vs element count. Optional: compare to assembled CSR SpMV at the same DOF count (fairness: include assembly cost if comparing “time to solution” over a long Krylov run).

**Likely improvements (header TODOs + roadmap):** mesh coloring; `float` where acceptable; SoA layouts; careful shared-memory staging of gathered node data; keep the FEM algorithm fixed while changing data motion.

**Caveat for low-order tets:** matrix-free advantages are often larger for high-order hexes (sum factorization). P2 tets are near the interesting crossover—worth saying if asked “is matrix-free always faster?”

---

## 9. What you’d change with more time

Prioritized list matching the `.cu` header TODOs and the repo roadmap:

1. **CPU–GPU output differencing harness** (same `(mesh, u, material)` → compare $y$).
2. **Wire CUDA into CMake** and run that harness in CI.
3. **Mesh coloring** to eliminate `atomicAdd` contention (header: “should provide a significant speedup on large meshes”).
4. **Scalar template** (`float` / `double`; header mentions C++20 concepts).
5. **Runtime checks** for $\det(J) \le 0$ and connectivity bounds.
6. **GPU matvec inside existing CG/PCG** (CPU already has the solver stack).
7. **Block Jacobi** (nodal $3\times3$) and, longer term, multigrid-class preconditioning for mesh-independent iteration counts.

---

## 10. “Walk me through the kernel” (60–90 seconds)

Use this as an oral crib sheet while screen-sharing:

1. **`LaunchGlobalOperatorKernel`** — grid from `num_elements`, block size 128; `y_global` already zeroed by caller.
2. **Thread** picks `elem`; loads connectivity row.
3. **Gather** — 4 corners from `node_coords`, 10 displacements from `u_global`.
4. **`ComputeElementOperator`** — `ComputeElementGeometry` once ($J$, $J^{-1}$, $\det J$); 4 QPs: shape grads → physical grads → $\nabla u$ → $\varepsilon$ → $\sigma$ → accumulate `y_local`.
5. **Scatter** — `atomicAdd` of each `y_local` component into `y_global[global_node]`.
6. **Physics invariant** — result must not depend on block size; only on mesh, material, and $u$.

---

## 11. If you’re unsure (transparency)

Andrew cares about working through gaps logically. Useful lines:

- “I’d go back to the weak form and check whether this identity has to hold—e.g. partition of unity for $\sum y_i = 0$.”
- “I’d verify on a single element with a rigid mode before debugging the scatter.”
- “I’m less sure about the GPU occupancy details; I’d measure with Nsight rather than guess. Structurally I’d expect atomics and register pressure to dominate — that’s why coloring is in the TODO list.”
- “Finite rotations aren’t a valid small-strain null-space test; I’d use $u=\omega\times x$ instead.”
- “For a full solve I’d wrap this matvec in Dirichlet-reduced CG with a breakdown guard on $p^TAp$; that lives on the CPU path today.”
- “Robustness for industrial unattended runs is more about consistent convergence across geometries than peak FLOPs — that points at better preconditioning after the matvec is solid.”

---

## Quick symbol map

| Symbol / API | Role |
|---|---|
| `ComputeElementOperator` | Local $y_e = K_e u_e$ via quadrature |
| `GlobalOperatorKernel` | One thread / element, gather–apply–scatter |
| `LaunchGlobalOperatorKernel` | Host launch wrapper (block size 128) |
| `MeshView` | Device mesh + material view |
| `TetrahedronQuadratureRule` | 4-point Hammer–Stroud (weights include $1/6$) |
| `EvaluateShapeFunctions` | P2 $N_i$ and $\partial N_i/\partial\xi$ |
| `ComputeElementGeometry` | Affine $J$, $J^{-1}$, $\det J$ from 4 corners |

Related CPU references if the conversation expands past the assignment file: [`../assembly.hpp`](../assembly.hpp), [`../boundary_conditions.hpp`](../boundary_conditions.hpp), [`../linear_solver.hpp`](../linear_solver.hpp).
