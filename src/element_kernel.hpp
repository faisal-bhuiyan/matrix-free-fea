#pragma once

#include "geometry.hpp"
#include "quadrature.hpp"
#include "shape_functions.hpp"
#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Constants
//---------------------------------------------------------------------------

constexpr int kDimensions{3};  ///< Spatial dimension
constexpr int kElementDOFs{
    kNodesPerTetElement * kDimensions
};  ///< DOFs per element -> 10 nodes * 3 dimensions = 30

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
 * @brief Compute y_e = K_e * u_e for one element without forming K_e.
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
 * @param corner_nodes   Physical coordinates of the 4 corner nodes
 * @param u_local        Nodal displacements, one Vector3 per node in
 *                       EvaluateShapeFunctions ordering
 * @param material_at_QP Lamé parameters at each of the 4 quadrature points,
 *                       in TetQuadratureRule() order
 * @param y_local        Output: local internal-force vector, one Vector3 per
 *                       node.  Zero-initialised by this function
 */
inline void ComputeElementOperator(
    const Vector3 corner_nodes[4], const Vector3 u_local[kNodesPerTetElement],
    const LinearElasticMaterial material_at_QP[4],
    Vector3 y_local[kNodesPerTetElement]
) {
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        y_local[i] = Vector3::Zero();
    }

    // Affine map -> J, inverse_jacobian, det_jacobian are constant across
    // the element -> compute once here rather than inside quadrature loop
    const ElementGeometry element_geometry =
        ComputeElementGeometry(corner_nodes);
    const Matrix3 J_inverse_transpose =
        element_geometry.inverse_jacobian.transpose();

    const auto& quadrature_rule = TetQuadratureRule();
    for (int qp_idx = 0; qp_idx < 4; ++qp_idx) {
        const QuadraturePoint& qp = quadrature_rule[qp_idx];

        // Shape-function gradients DO vary per quadrature point even
        // though J does not (quadratic P2 basis vs. linear geometry)
        const ShapeFunctionData shape_func =
            EvaluateShapeFunctions(qp.coords.x(), qp.coords.y(), qp.coords.z());

        Vector3 dN_dx[kNodesPerTetElement]{};
        PhysicalGradients(
            J_inverse_transpose, shape_func.shape_func_derivatives, dN_dx
        );

        // Strain: epsilon(u) = 0.5*(grad(u) + grad(u)^T)
        // grad(u)_{jk} = sum_i (u_local[i][j] * dN_dx[i][k]) ->
        //  (= B*u_e, implicit)
        Matrix3 grad_displacement{Matrix3::Zero()};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            const Vector3& u_i = u_local[node_idx];
            const Vector3& dN_i_dx = dN_dx[node_idx];
            for (int dim_1 = 0; dim_1 < kDimensions; ++dim_1) {
                for (int dim_2 = 0; dim_2 < kDimensions; ++dim_2) {
                    grad_displacement(dim_1, dim_2) +=
                        u_i[dim_1] * dN_i_dx[dim_2];
                }
            }
        }
        const Matrix3 epsilon{
            (grad_displacement + grad_displacement.transpose()) * 0.5
        };

        // Stress: sigma = lambda*tr(eps)*I + 2*mu*eps
        const double lambda{material_at_QP[qp_idx].lambda};
        const double mu{material_at_QP[qp_idx].mu};
        const Matrix3 sigma{
            Matrix3::Identity() * (lambda * epsilon.trace()) +
            epsilon * (2. * mu)
        };

        // Accumulate: y_i += w_q * det(J) * sigma * grad(N_i) ->
        // (= B^T*sigma, implicit)
        const double scale{qp.weight * element_geometry.det_jacobian};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            y_local[node_idx] += (sigma * dN_dx[node_idx]) * scale;
        }
    }
}

}  // namespace matrix_free_fea
