#pragma once

#include "shape_functions.hpp"
#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Pre-computed affine geometry for one tetrahedral element.
 *
 * Encapsulates the geometric Jacobian J = dx/d(xi,eta,zeta) and its inverse
 * for the affine map from the reference tetrahedron to the physical element.
 * Because the map is linear in the corner-node coordinates (independent of
 * the quadratic displacement basis), all three quantities are constant across
 * the element and need only be computed once per element, not once per
 * quadrature point.
 */
struct ElementGeometry {
    Matrix3 jacobian;          ///< Jacobian dx/d(xi, eta, zeta) ->
                               ///< columns = [x1-x0 | x2-x0 | x3-x0]
    Matrix3 inverse_jacobian;  ///< Inverse of Jacobian -> used to pull physical
                               ///< space gradients back to reference space
    double det_jacobian;       ///< det(J) -> positive for a non-inverted,
                               ///< right-handed element
};

//---------------------------------------------------------------------------
// Functions
//---------------------------------------------------------------------------

/**
 * @brief Computes the affine element geometry from the 4 corner-node
 * coordinates.
 *
 * The affine map is x(xi,eta,zeta) = x0 + xi*(x1-x0) + eta*(x2-x0) +
 * zeta*(x3-x0), so J = [x1-x0 | x2-x0 | x3-x0].  J is assembled with
 * @ref FromColumns, inverted with Eigen's dense 3x3 inverse, and its
 * determinant is stored for use in the quadrature weight scaling.
 *
 * @param corner_nodes Physical (x,y,z) coordinates of the 4 corner nodes, in
 * the same ordering as @ref EvaluateShapeFunctions (corner node 0 maps to the
 * reference tet's origin)
 * @return A fully populated ElementGeometry with J, inverse_jacobian, and
 *         det_jacobian
 */
inline ElementGeometry ComputeElementGeometry(const Vector3 corner_nodes[4]) {
    ElementGeometry element_geometry{};
    element_geometry.jacobian = FromColumns(
        corner_nodes[1] - corner_nodes[0],  // x1 - x0
        corner_nodes[2] - corner_nodes[0],  // x2 - x0
        corner_nodes[3] - corner_nodes[0]   // x3 - x0
    );
    element_geometry.det_jacobian = element_geometry.jacobian.determinant();
    element_geometry.inverse_jacobian = element_geometry.jacobian.inverse();
    return element_geometry;
}

/**
 * @brief Transforms reference-space shape-function gradients to physical space.
 *
 * Applies the pull-back dN/dx = J^{-T} * dN/dxi for all kNodesPerTetElement
 * nodes at once. J^{-T} is constant across the element (affine map) but
 * must still be applied per quadrature point because dN_dxi varies with the
 * evaluation point (quadratic P2/basis).
 *
 * @param J_inv_transpose      Transpose of J^{-1} -> constant for element
 * @param dN_dxi     Reference-space gradients dN_i/d(xi,eta,zeta) -> one per
 *                   node (P2/quadratic basis)
 * @param dN_dx      Output: physical-space gradients dN_i/dx -> one per node
 */
inline void PhysicalGradients(
    const Matrix3& J_inv_transpose, const Vector3 dN_dxi[kNodesPerTetElement],
    Vector3 dN_dx[kNodesPerTetElement]
) {
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        dN_dx[node_idx] = J_inv_transpose * dN_dxi[node_idx];
    }
}

}  // namespace matrix_free_fea
