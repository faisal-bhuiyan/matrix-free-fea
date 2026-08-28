/**
 * @file assembly_kernels.cu
 * @brief Global matrix-free operator kernel and host launch wrapper.
 */
#include "mesh_view.cuh"

//---------------------------------------------------------------------------
// Global matrix-free linear elasticity operator
//---------------------------------------------------------------------------

/**
 * @brief Applies the global stiffness operator y = K * u without ever forming
 * K -> one CUDA thread per element.
 *
 * Each thread handles exactly one element and runs three phases:
 *
 *   - Gather -> copy the element's 4 corner coordinates and 10 nodal
 *     displacements out of the global arrays into thread-local buffers, using
 *     the element's connectivity row as the index map.
 *
 *   - Apply -> call @ref ComputeElementLinearElasticityOperator to evaluate
 *     y_local = K_e * u_local by quadrature -> K_e is likewise never formed.
 *
 *   - Scatter-add -> accumulate the 10 local force vectors back into y_global
 *     at those same connectivity slots.
 *
 * The scatter is the only phase that needs synchronisation. Elements sharing
 * a face, edge, or vertex reach the same global node from different threads,
 * so plain += would race and each component goes through atomicAdd instead.
 * atomicAdd on double requires compute capability 6.0 or newer.
 *
 * @param mesh      Device-resident mesh -> see @ref MeshView
 * @param u_global  Device array, input displacement field -> [num_nodes]
 * @param y_global  Output: device array, global internal-force vector ->
 *                  [num_nodes]. The caller must zero this before launch (e.g.
 *                  cudaMemset), since the kernel accumulates and concurrent
 *                  threads cannot each independently reset a shared array
 */
__global__ void GlobalLinearElasticOperatorKernel(
    MeshView mesh, const Vector3* u_global, Vector3* y_global
) {
    const int elem{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (elem >= mesh.num_elements) {
        return;
    }

    // Maps the element's 10 local slots -> global node indices
    const int* connectivity{mesh.connectivity + elem * kNodesPerTetElement};

    // (1) Gather -> pull element's data out of the global arrays so the
    // element operator can walk it contiguously
    Vector3 corner_nodes[kCornersPerTetElement];
    for (int node_idx = 0; node_idx < kCornersPerTetElement; ++node_idx) {
        corner_nodes[node_idx] = mesh.node_coords[connectivity[node_idx]];
    }

    // Copy element's 10 nodal displacements out of the global array
    // into a local contiguous buffer
    Vector3 u_local[kNodesPerTetElement];
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        u_local[node_idx] = u_global[connectivity[node_idx]];
    }

    // (2) Apply -> {y_local} = [K_e] * {u_local} by quadrature -> [K_e] is
    // never formed
    Vector3 y_local[kNodesPerTetElement];
    ComputeElementLinearElasticityOperator(
        corner_nodes, u_local, mesh.material + elem * kQuadraturePoints, y_local
    );

    // (3) Scatter-add -> fold local forces back into the global vector.
    // Shared nodes accumulate one term per adjacent element -> concurrent
    // threads may target the same node, so += becomes atomicAdd
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        const int global_node{connectivity[node_idx]};
        atomicAdd(&(y_global[global_node].x), y_local[node_idx].x);
        atomicAdd(&(y_global[global_node].y), y_local[node_idx].y);
        atomicAdd(&(y_global[global_node].z), y_local[node_idx].z);
    }
}

/**
 * @brief Host-side launch wrapper for @ref GlobalLinearElasticOperatorKernel.
 *
 * The block size is a multiple of the 32-thread warp size, as full-warp
 * utilisation requires, and is kept modest because each thread carries 30
 * local DOFs of live state -> a larger block would only cut occupancy
 * further.
 *
 * @param mesh      Device-resident mesh -> see @ref MeshView
 * @param u_global  Device array, input displacement field -> [num_nodes]
 * @param y_global  Output: device array, global internal-force vector ->
 *                  [num_nodes]. Must already be zeroed by the caller, since
 *                  the kernel accumulates rather than assigns
 */
inline void LaunchGlobalLinearElasticOperatorKernel(
    const MeshView& mesh, const Vector3* u_global, Vector3* y_global
) {
    constexpr int kBlockSize{128};
    const int grid_size{(mesh.num_elements + kBlockSize - 1) / kBlockSize};

    GlobalLinearElasticOperatorKernel<<<grid_size, kBlockSize>>>(
        mesh, u_global, y_global
    );
}

}  // namespace matrix_free_fea
