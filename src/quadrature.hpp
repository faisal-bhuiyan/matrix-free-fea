#pragma once

#include <array>
#include <cmath>

#include "types.hpp"

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

//---------------------------------------------------------------------------
// Quadrature Rule
//---------------------------------------------------------------------------

/**
 * @brief Returns the 4-point Hammer-Stroud quadrature rule for the reference
 *        tetrahedron.
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
 * @return A const reference to a static array of 4 QuadraturePoint values.
 *         The node coordinates are computed from their exact closed-form
 *         expressions at first call and cached for the lifetime of the program.
 */
inline const std::array<QuadraturePoint, 4>& TetrahedronQuadratureRule() {
    static const double a{(5. + 3. * std::sqrt(5.)) / 20.};
    static const double b{(5. - std::sqrt(5.)) / 20.};
    static constexpr double weight{1. / 24.};

    static const std::array<QuadraturePoint, 4> quadrature_rule{{
        {Vector3{a, b, b}, weight},
        {Vector3{b, a, b}, weight},
        {Vector3{b, b, a}, weight},
        {Vector3{b, b, b}, weight},
    }};
    return quadrature_rule;
}

}  // namespace matrix_free_fea
