#pragma once

#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Constants and data types
//---------------------------------------------------------------------------

/// @brief Number of nodes per P2/quadratic tetrahedron element
constexpr int kNodesPerTetElement{10};

/**
 * @brief Holds shape function values and reference-space gradients at a single
 *        quadrature point.
 *
 * Holds the output of @ref EvaluateShapeFunctions for the standard quadratic
 * (P2) 10-node tetrahedron. Node ordering matches common conventions
 * (e.g. Abaqus C3D10).
 *
 *   Corners:    0, 1, 2, 3
 *   Edge nodes: 4 = mid(0,1)   5 = mid(1,2)   6 = mid(0,2)
 *               7 = mid(0,3)   8 = mid(1,3)   9 = mid(2,3)
 */
struct ShapeFunctionData {
    /// N_i(xi, eta, zeta) -> shape function values
    double shape_func_values[kNodesPerTetElement];

    /// dN_i/d(xi, eta, zeta) -> Reference-space shape function derivatives
    Vector3 shape_func_derivatives[kNodesPerTetElement];
};

//---------------------------------------------------------------------------
// Evaluation
//---------------------------------------------------------------------------

/**
 * @brief Evaluates all 10 P2/quadratic shape functions and their reference
 * space gradients at a given point in the reference tetrahedron.
 *
 * The reference tetrahedron has vertices at (0,0,0), (1,0,0), (0,1,0) and
 * (0,0,1) in (xi, eta, zeta).  The four barycentric coordinates are
 *
 *   L0 = 1 - xi - eta - zeta   (corner 0, the origin)
 *   L1 = xi                    (corner 1)
 *   L2 = eta                   (corner 2)
 *   L3 = zeta                  (corner 3)
 *
 * and the two families of shape functions are
 *
 *   Corner nodes (a = 0..3):  N_a = L_a * (2*L_a - 1)
 *   Edge nodes   (p,q pair):  N   = 4 * L_p * L_q
 *
 * @param xi   First barycentric coordinate (>= 0)
 * @param eta  Second barycentric coordinate (>= 0)
 * @param zeta Third barycentric coordinate (>= 0, xi+eta+zeta <= 1)
 * @return A ShapeFunctionData holding all N_i values and dN_i/d(xi, eta, zeta)
 * vectors
 */
inline ShapeFunctionData EvaluateShapeFunctions(
    double xi, double eta, double zeta
) {
    const double L[4] = {
        1. - xi - eta - zeta,  // L0
        xi,                    // L1
        eta,                   // L2
        zeta                   // L3
    };

    // dL_a/d(xi,eta,zeta), a = 0, 1, 2, 3 -> constant across the element
    // since the geometric map is affine
    static const Vector3 dL[4] = {
        Vector3{-1., -1., -1.},  // dL0
        Vector3{1., 0., 0.},     // dL1
        Vector3{0., 1., 0.},     // dL2
        Vector3{0., 0., 1.},     // dL3
    };

    ShapeFunctionData shape_func{};

    // Corner nodes (4): N_a = L_a * (2*L_a - 1), a = 0, 1, 2, 3
    for (int node_idx = 0; node_idx < 4; ++node_idx) {
        // shape function values for corner nodes
        shape_func.shape_func_values[node_idx] =
            L[node_idx] * (2. * L[node_idx] - 1.);

        // reference-space gradients for corner nodes -> dL_a/dxi
        shape_func.shape_func_derivatives[node_idx] =
            dL[node_idx] * (4. * L[node_idx] - 1.);
    }

    // Edge nodes (6): N = 4*L_p*L_q -> pairing matches the node-ordering above
    static const int edge_pairs[6][2]{
        {0, 1},  // edge 0
        {1, 2},  // edge 1
        {0, 2},  // edge 2
        {0, 3},  // edge 3
        {1, 3},  // edge 4
        {2, 3}   // edge 5
    };

    for (int edge_idx = 0; edge_idx < 6; ++edge_idx) {
        const int edge_node_1{edge_pairs[edge_idx][0]};
        const int edge_node_2{edge_pairs[edge_idx][1]};
        const int node_idx{4 + edge_idx};

        // shape function values for edge nodes -> N = 4 * L_p * L_q
        shape_func.shape_func_values[node_idx] =
            4. * L[edge_node_1] * L[edge_node_2];

        // reference-space gradients for edge nodes ->
        // dN/dxi = 4 * L_q * (dL_node_1/dxi) + 4 * L_p * (dL_node_2/dxi)
        shape_func.shape_func_derivatives[node_idx] =
            dL[edge_node_1] * (4. * L[edge_node_2]) +
            dL[edge_node_2] * (4. * L[edge_node_1]);
    }
    return shape_func;
}

}  // namespace matrix_free_fea
