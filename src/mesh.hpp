#pragma once

#include <array>
#include <vector>

#include "shape_functions.hpp"
#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Global-to-local connectivity for one P2 tetrahedral element.
 *
 * Maps the element's 10 local node slots (in the ordering defined by
 * @ref EvaluateShapeFunctions) to global node indices in the Mesh node array.
 */
using ElementConnectivity = std::array<int, kNodesPerTetElement>;

/**
 * @brief Minimal mesh: global node coordinates and per-element connectivity.
 *
 * Stores only what the matrix-free operator needs: the physical coordinates
 * of every global node and, for each element, the 10 global indices that
 * identify its nodes.  Reading the mesh from a file or generating it with an
 * external mesher is intentionally out of scope for this module.
 */
struct Mesh {
    std::vector<Vector3>
        node_coords;  ///< Physical (x,y,z) coordinate of each global node

    std::vector<ElementConnectivity>
        elements;  ///< Per-element connectivity arrays

    /**
     * @brief Return the number of global nodes.
     */
    int NumNodes() const { return static_cast<int>(node_coords.size()); }

    /**
     * @brief Return the number of elements.
     */
    int NumElements() const { return static_cast<int>(elements.size()); }

    /**
     * @brief Copy the 4 corner-node coordinates of an element into @p corners.
     *
     * Corners occupy local slots 0, 1, 2, 3 -> only these four enter the affine
     * geometric map (see geometry.hpp), so the remaining 6 edge-midpoint
     * slots are not needed here.
     *
     * @param elem_index Index of the element (0-based).
     * @param corners Output array of length 4 filled with the physical
     *                coordinates of local corners 0, 1, 2, 3.
     */
    void ElementCorners(int elem_index, Vector3 corners[4]) const {
        const auto& connectivity{elements[elem_index]};
        for (int i = 0; i < 4; ++i) {
            corners[i] = node_coords[connectivity[i]];
        }
    }

    /**
     * @brief Copy all 10 nodal coordinates of an element into @p nodes.
     *
     * @param elem_index Index of the element (0-based).
     * @param nodes   Output array of length 10 filled with the physical
     *                coordinates of all local nodes in EvaluateShapeFunctions
     *                order.
     */
    void ElementNodes(
        int elem_index, Vector3 nodes[kNodesPerTetElement]
    ) const {
        const auto& connectivity{elements[elem_index]};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            nodes[node_idx] = node_coords[connectivity[node_idx]];
        }
    }
};

}  // namespace matrix_free_fea
