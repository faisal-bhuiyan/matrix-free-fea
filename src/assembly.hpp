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
 * @brief Apply the global matrix-free stiffness operator: y = K * u.
 *
 * Iterates over all elements and performs the standard gather -> element
 * operator -> scatter-add loop.  This is the serial-CPU analogue of the CUDA
 * assembly kernel: the per-element body (gather, @ref ComputeElementOperator,
 * scatter-add) maps directly to a CUDA kernel's per-thread body, and the
 * element loop becomes the thread grid.  The only addition the CUDA port
 * requires is an atomicAdd for the scatter step, since multiple threads may
 * write the same shared global node concurrently. Here the sequential loop
 * makes plain += race-free.
 *
 * @param mesh               Global node coordinates and element connectivity.
 * @param u_global           Input displacement field, one Vector3 per global
 * node.
 * @param material_per_element Lamé parameters at each element's 4 quadrature
 *                           points; indexed as material_per_element[e][q].
 * @param y_global           Output: global internal-force vector, resized and
 *                           zero-initialised to mesh.NumNodes() entries.
 */
inline void ApplyGlobalOperator(
    const Mesh& mesh, const std::vector<Vector3>& u_global,
    const std::vector<std::array<LinearElasticMaterial, 4>>&
        material_per_element,
    std::vector<Vector3>& y_global
) {
    y_global.assign(mesh.NumNodes(), Vector3::Zero());
    for (int elem = 0; elem < mesh.NumElements(); ++elem) {
        const auto& connectivity = mesh.elements[elem];

        // Gather -> copy element's corner coords and nodal displacements from
        // global arrays into contiguous local buffers
        Vector3 corners[4]{};
        mesh.ElementCorners(elem, corners);

        Vector3 u_local[kNodesPerTetElement]{};
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            u_local[i] = u_global[connectivity[i]];
        }

        // Element operator (matrix-free) -> y_local = K_e * u_local
        Vector3 y_local[kNodesPerTetElement]{};
        ComputeElementOperator(
            corners, u_local, material_per_element[elem].data(), y_local
        );

        // Scatter-add -> accumulate local contributions into global force
        // vector -> race-free for sequential code, CUDA port uses atomicAdd
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            y_global[connectivity[i]] += y_local[i];
        }
    }
}

}  // namespace matrix_free_fea
