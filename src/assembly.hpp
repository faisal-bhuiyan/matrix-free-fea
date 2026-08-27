#pragma once

#include <array>
#include <vector>

#include "element_kernel.hpp"
#include "mesh.hpp"
#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Global matrix-free linear elasticity operator
//---------------------------------------------------------------------------

/**
 * @brief Applies the global stiffness operator y = K * u without ever forming
 * K -> matrix-free.
 *
 * This is the central routine of the library. A conventional FE code would
 * assemble the sparse global matrix K once and then multiply by it. Here K is
 * never built: every call re-derives each element's contribution on the fly
 * from the mesh geometry and material data. The working set shrinks from the
 * assembled matrix's nonzeros to little more than the solution vectors ->
 * one to two orders of magnitude less memory for P2 tets. The cost is
 * recomputing the element integrals on every apply, which is the trade a
 * matrix-free formulation makes. On bandwidth-bound hardware, and GPUs in
 * particular, that recomputation is often cheaper than the memory traffic it
 * replaces.
 *
 * Each element is processed in three phases:
 *
 *   - Gather -> copy the element's 4 corner coordinates and 10 nodal
 *     displacements out of the global arrays into small contiguous local
 *     buffers, using the element's connectivity as the index map.
 *
 *   - Apply -> call @ref ComputeElementLinearElasticityOperator to evaluate
 *     y_local = K_e * u_local by quadrature -> K_e is likewise never formed.
 *
 *   - Scatter-add -> accumulate the 10 local force vectors back into y_global
 *     at those same connectivity slots. A node shared by several elements
 *     receives one contribution per element that touches it, which is exactly
 *     what summing the element integrals over the mesh means.
 *
 * Mapping to CUDA: this serial loop is the direct analogue of the intended GPU
 * kernel. The element loop becomes the thread grid (one thread per element)
 * and the loop body carries over unchanged. Only the scatter step differs ->
 * `+=` must become `atomicAdd` -> since several threads may write the same
 * shared global node concurrently. The sequential loop here is race-free, so
 * plain `+=` suffices.
 *
 * @param mesh      Global node coordinates and per-element connectivity
 * @param u_global  Input displacement field, one Vector3 per global node ->
 *                  must hold mesh.NumNodes() entries
 * @param per_element_mat_props Lamé parameters at each element's 4 quadrature
 *                  points, indexed as @p per_element_mat_props[elem][qp] in
 *                  TetrahedronQuadratureRule() order -> must hold
 *                  mesh.NumElements() entries
 * @param y_global  Output: global internal-force vector.  Resized to
 *                  mesh.NumNodes() and zero-initialised here, so any prior
 *                  contents are discarded
 */
inline void ApplyGlobalLinearElasticityOperator(
    const Mesh& mesh, const std::vector<Vector3>& u_global,
    const std::vector<std::array<LinearElasticMaterial, 4>>&
        per_element_mat_props,
    std::vector<Vector3>& y_global
) {
    // Every call recomputes y from scratch -> discard whatever was passed in
    y_global.assign(mesh.NumNodes(), Vector3::Zero());

    // One iteration per element -> becomes one CUDA thread per element
    for (int elem = 0; elem < mesh.NumElements(); ++elem) {
        // Maps the element's 10 local slots -> global node indices
        const auto& connectivity = mesh.elements[elem];

        // (1) Gather -> pull element's data out of the global arrays so
        // the kernel can walk it contiguously
        Vector3 corner_nodes[4]{};
        mesh.ElementCorners(elem, corner_nodes);

        // Copy element's 10 nodal displacements out of the global array
        // into a local contiguous buffer
        Vector3 u_local[kNodesPerTetElement]{};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            u_local[node_idx] = u_global[connectivity[node_idx]];
        }

        // (2) Apply -> {y_local} = [K_e] * {u_local} by quadrature -> [K_e] is
        // never formed
        Vector3 y_local[kNodesPerTetElement]{};
        ComputeElementLinearElasticityOperator(
            corner_nodes, u_local, per_element_mat_props[elem].data(), y_local
        );

        // (3) Scatter-add -> fold local forces back into the global vector.
        // Shared nodes accumulate one term per adjacent element -> this is the
        // only step a CUDA port must change -> += becomes atomicAdd
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            y_global[connectivity[node_idx]] += y_local[node_idx];
        }
    }
}

}  // namespace matrix_free_fea
