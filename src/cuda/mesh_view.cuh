#pragma once

#include "element_kernel.cuh"

namespace matrix_free_fea {

/**
 * @brief Device-resident, flattened view of the mesh.
 *
 * Owning containers such as std::vector do not survive a trip to the device,
 * so the mesh is described here in flattened form: raw pointers into device
 * memory plus the two sizes. Building it -> copying each array across with
 * cudaMemcpy -> is the caller's job and out of this file's scope. This struct
 * only describes the layout the kernel expects to already find in device
 * memory.
 */
struct MeshView {
    /// Physical (x,y,z) coordinate of each global node -> [num_nodes]
    const Vector3* node_coords;

    /// Per-element connectivity, row-major ->
    /// [num_elements * kNodesPerTetElement]
    const int* connectivity;

    /// Lamé parameters per element per quadrature point, row-major ->
    /// [num_elements * kQuadraturePoints]
    const LinearElasticMaterial* material;

    int num_nodes;     ///< Number of global nodes
    int num_elements;  ///< Number of elements
};

}  // namespace matrix_free_fea
