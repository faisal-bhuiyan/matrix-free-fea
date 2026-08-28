#pragma once

#include "types_device.cuh"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Represents a single quadrature point and its weight in the reference
 * tetrahedron.
 *
 * Holds the barycentric coordinates (xi, eta, zeta) packed into a Vector3
 * and the associated integration weight for one point in the Hammer-Stroud
 * quadrature rule. The weight already incorporates the reference-tetrahedron
 * volume (1/6), so the integral of a function f over the reference tet is
 * approximated as -> sum_q (weight_q * f(coords_q)) with no further scaling.
 */
struct QuadraturePoint {
    Vector3 coords;  ///< Barycentric coordinates (xi, eta, zeta) of the point
    double weight;   ///< Integration weight (includes reference-tet volume 1/6)
};

/// @brief Number of points in the quadrature rule below
constexpr int kQuadraturePoints{4};

//---------------------------------------------------------------------------
// Quadrature Rule
//---------------------------------------------------------------------------

/**
 * @brief Writes the 4-point Hammer-Stroud quadrature rule for the reference
 *        tetrahedron into @p quadrature_rule.
 *
 * The rule is exact for polynomials up to total degree 2. The P2/quadratic
 * tetrahedron's shape-function gradients are degree-1 polynomials in
 * (xi, eta, zeta) -> integrand epsilon(v):sigma(u) -> a product of two such
 * gradients -> is degree 2 and is integrated exactly.
 *
 * Material properties lambda and mu are evaluated pointwise at each quadrature
 * point rather than interpolated -> they do not raise the polynomial degree.
 *
 * The reference tetrahedron has volume 1/6; the 4 points share equal weight,
 * giving each weight = (1/6) / 4 = 1/24.
 *
 * The rule fills a caller-supplied array from literal constants rather than
 * returning a reference to a function-local static. That keeps the function
 * pure and stateless -> there is no device-side static whose initialisation
 * order and thread-safety across concurrently launched threads would need
 * reasoning about. The constants a and b below are (5 + 3*sqrt(5))/20 and
 * (5 - sqrt(5))/20 respectively.
 *
 * @param quadrature_rule Output: the 4 QuadraturePoint values of the rule
 */
MFFEA_HOST_DEVICE inline void TetrahedronQuadratureRule(
    QuadraturePoint quadrature_rule[kQuadraturePoints]
) {
    constexpr double a{0.5854101966249685};
    constexpr double b{0.1381966011250105};
    constexpr double weight{1. / 24.};

    quadrature_rule[0] = QuadraturePoint{Vector3(a, b, b), weight};
    quadrature_rule[1] = QuadraturePoint{Vector3(b, a, b), weight};
    quadrature_rule[2] = QuadraturePoint{Vector3(b, b, a), weight};
    quadrature_rule[3] = QuadraturePoint{Vector3(b, b, b), weight};
}

//---------------------------------------------------------------------------
// Constants and data types
//---------------------------------------------------------------------------

/// @brief Number of nodes per P2/quadratic tetrahedron element
constexpr int kNodesPerTetElement{10};

/// @brief Number of corner nodes per tetrahedron -> only these four enter the
/// affine geometric map, while all 10 carry the quadratic displacement basis
constexpr int kCornersPerTetElement{4};

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
 * The dL and edge_pairs tables are plain locals rather than function-local
 * statics, for the same reason as @ref TetrahedronQuadratureRule. Both are
 * tiny (4 and 6 entries), so rebuilding them per call costs less than a
 * device-side static would.
 *
 * @param xi   First barycentric coordinate (>= 0)
 * @param eta  Second barycentric coordinate (>= 0)
 * @param zeta Third barycentric coordinate (>= 0, xi+eta+zeta <= 1)
 * @return A ShapeFunctionData holding all N_i values and dN_i/d(xi, eta, zeta)
 * vectors
 */
MFFEA_HOST_DEVICE inline ShapeFunctionData EvaluateShapeFunctions(
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
    const Vector3 dL[4] = {
        Vector3(-1., -1., -1.),  // dL0
        Vector3(1., 0., 0.),     // dL1
        Vector3(0., 1., 0.),     // dL2
        Vector3(0., 0., 1.),     // dL3
    };

    ShapeFunctionData shape_func{};

    // Corner nodes (4): N_a = L_a * (2*L_a - 1), a = 0, 1, 2, 3
    for (int node_idx = 0; node_idx < kCornersPerTetElement; ++node_idx) {
        // shape function values for corner nodes
        shape_func.shape_func_values[node_idx] =
            L[node_idx] * (2. * L[node_idx] - 1.);

        // reference-space gradients for corner nodes -> dL_a/dxi
        shape_func.shape_func_derivatives[node_idx] =
            dL[node_idx] * (4. * L[node_idx] - 1.);
    }

    // Edge nodes (6): N = 4*L_p*L_q -> pairing matches the node-ordering above
    const int edge_pairs[6][2]{
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
 * @ref FromColumns, inverted with the closed-form 3x3 adjugate, and its
 * determinant is stored for use in the quadrature weight scaling. A
 * non-degenerate, non-inverted element (det(J) > 0) is assumed and not
 * checked.
 *
 * @param corner_nodes Physical (x,y,z) coordinates of the 4 corner nodes, in
 * the same ordering as @ref EvaluateShapeFunctions (corner node 0 maps to the
 * reference tet's origin)
 * @return A fully populated ElementGeometry with J, inverse_jacobian, and
 *         det_jacobian
 */
MFFEA_HOST_DEVICE inline ElementGeometry ComputeElementGeometry(
    const Vector3 corner_nodes[kCornersPerTetElement]
) {
    ElementGeometry element_geometry{};
    element_geometry.jacobian = FromColumns(
        corner_nodes[1] - corner_nodes[0],  // x1 - x0
        corner_nodes[2] - corner_nodes[0],  // x2 - x0
        corner_nodes[3] - corner_nodes[0]   // x3 - x0
    );
    element_geometry.det_jacobian = element_geometry.jacobian.Determinant();
    element_geometry.inverse_jacobian = element_geometry.jacobian.Inverse();
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
MFFEA_HOST_DEVICE inline void PhysicalGradients(
    const Matrix3& J_inv_transpose, const Vector3 dN_dxi[kNodesPerTetElement],
    Vector3 dN_dx[kNodesPerTetElement]
) {
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        dN_dx[node_idx] = J_inv_transpose * dN_dxi[node_idx];
    }
}

//---------------------------------------------------------------------------
// Constants
//---------------------------------------------------------------------------

/// @brief Spatial dimension
constexpr int kDimensions{3};

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Lamé material parameters at a single quadrature point.
 *
 * Lambda and mu are provided pointwise rather than interpolated by a
 * polynomial basis, so they do not raise the polynomial degree of the
 * integrand and the 4-point quadrature rule remains exact.
 */
struct LinearElasticMaterial {
    double lambda;  ///< First Lamé parameter (related to bulk modulus)
    double mu;      ///< Second Lamé parameter (shear modulus)
};

//---------------------------------------------------------------------------
// Compute element operator
//---------------------------------------------------------------------------

/**
 * @brief Computes y_e = K_e * u_e for one element without forming K_e.
 *
 * Implements the matrix-free evaluation of the local stiffness action via
 * the weak-form identity:
 *
 *   (y_e)_{i,d} = integral_Omega  ( sigma(u) * grad(N_i) )_d  dOmega
 *
 * derived from a(v, u) = integral epsilon(v):sigma(u) with v = N_i * e_d.
 * At each quadrature point the algorithm:
 *
 *   1. Evaluates dN/dx = J^{-T} * dN/dxi  (physical-space gradients)
 *   2. Assembles grad(u) = sum_i u_i (x) dN_i/dx  (B * u_e, never explicit)
 *   3. Computes epsilon = sym(grad(u)) and sigma = lambda*tr(eps)*I + 2*mu*eps
 *   4. Accumulates y_i += w_q * det(J) * sigma * grad(N_i) -> (B^T * sigma,
 *      never explicit)
 *
 * The geometric Jacobian J is constant across the element (affine map) and is
 * therefore computed once before the quadrature loop.
 *
 * This is the unit of work one CUDA thread performs, in full, for one element
 * -> see @ref GlobalLinearElasticOperatorKernel below.
 *
 * @param corner_nodes   Physical coordinates of the 4 corner nodes
 * @param u_local        Nodal displacements, one Vector3 per node in
 *                       EvaluateShapeFunctions ordering
 * @param material_at_QP Lamé parameters at each of the 4 quadrature points,
 *                       in TetrahedronQuadratureRule() order
 * @param y_local        Output: local internal-force vector, one Vector3 per
 *                       node.  Zero-initialised by this function
 */
MFFEA_HOST_DEVICE inline void ComputeElementLinearElasticityOperator(
    const Vector3 corner_nodes[kCornersPerTetElement],
    const Vector3 u_local[kNodesPerTetElement],
    const LinearElasticMaterial material_at_QP[kQuadraturePoints],
    Vector3 y_local[kNodesPerTetElement]
) {
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        y_local[node_idx] = Vector3::Zero();
    }

    // Affine map -> J, inverse_jacobian, det_jacobian are constant across
    // the element -> compute once here rather than inside quadrature loop
    const ElementGeometry element_geometry{
        ComputeElementGeometry(corner_nodes)
    };
    const Matrix3 J_inverse_transpose{
        element_geometry.inverse_jacobian.Transpose()
    };

    QuadraturePoint quadrature_rule[kQuadraturePoints];
    TetrahedronQuadratureRule(quadrature_rule);

    for (int qp_idx = 0; qp_idx < kQuadraturePoints; ++qp_idx) {
        const QuadraturePoint& qp{quadrature_rule[qp_idx]};

        // Shape-function gradients DO vary per quadrature point even
        // though J does NOT -> quadratic P2 basis vs. linear geometry
        const ShapeFunctionData shape_func{
            EvaluateShapeFunctions(qp.coords.x, qp.coords.y, qp.coords.z)
        };

        Vector3 dN_dx[kNodesPerTetElement];
        PhysicalGradients(
            J_inverse_transpose, shape_func.shape_func_derivatives, dN_dx
        );

        // Strain: epsilon(u) = 0.5*(grad(u) + grad(u)^T)
        // grad(u)_{jk} = sum_i (u_local[i][j] * dN_dx[i][k]) ->
        //  (= B*u_e, implicit)
        Matrix3 grad_displacement{Matrix3::Zero()};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            const Vector3& u_i{u_local[node_idx]};
            const Vector3& dN_i_dx{dN_dx[node_idx]};
            for (int dim_1 = 0; dim_1 < kDimensions; ++dim_1) {
                for (int dim_2 = 0; dim_2 < kDimensions; ++dim_2) {
                    grad_displacement(dim_1, dim_2) +=
                        u_i[dim_1] * dN_i_dx[dim_2];
                }
            }
        }
        const Matrix3 epsilon{
            (grad_displacement + grad_displacement.Transpose()) * 0.5
        };

        // Stress: sigma = lambda*tr(eps)*I + 2*mu*eps
        const double lambda{material_at_QP[qp_idx].lambda};
        const double mu{material_at_QP[qp_idx].mu};
        const Matrix3 sigma{
            Matrix3::Identity() * (lambda * epsilon.Trace()) +
            epsilon * (2. * mu)
        };

        // Accumulate: y_i += w_q * det(J) * sigma * grad(N_i) ->
        // (= B^T*sigma -> implicit)
        const double scale{qp.weight * element_geometry.det_jacobian};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            y_local[node_idx] += (sigma * dN_dx[node_idx]) * scale;
        }
    }
}

}  // namespace matrix_free_fea
