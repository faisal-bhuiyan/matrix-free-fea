// test_quadrature.cpp — unit tests for quadrature.hpp.
//
// Verifies weight sum, point locations, permutation symmetry, exactness
// through degree 2 against the closed-form factorial formula, and
// inexactness at degree 3 (pinning the stated accuracy limit).

#include <cmath>

#include <gtest/gtest.h>

#include "quadrature.hpp"

using namespace matrix_free_fea;

namespace {

double Factorial(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; ++i) {
        f *= i;
    }
    return f;
}

// Closed form: integral_ref_tet xi^a eta^b zeta^c dV = a! b! c! / (a+b+c+3)!
double ExactMonomialIntegral(int a, int b, int c) {
    return Factorial(a) * Factorial(b) * Factorial(c) /
           Factorial(a + b + c + 3);
}

double QuadratureMonomial(int a, int b, int c) {
    const auto& rule = TetQuadratureRule();
    double sum = 0.0;
    for (const auto& qp : rule) {
        const double xi = qp.coords.x();
        const double eta = qp.coords.y();
        const double zeta = qp.coords.z();
        sum +=
            qp.weight * std::pow(xi, a) * std::pow(eta, b) * std::pow(zeta, c);
    }
    return sum;
}

}  // namespace

TEST(Quadrature, WeightsSumToReferenceVolume) {
    const auto& rule = TetQuadratureRule();
    double sum = 0.0;
    for (const auto& qp : rule) {
        EXPECT_NEAR(qp.weight, 1.0 / 24.0, 1e-15);
        sum += qp.weight;
    }
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-15);
}

TEST(Quadrature, PointsInsideReferenceTet) {
    const auto& rule = TetQuadratureRule();
    const double a = (5. + 3. * std::sqrt(5.)) / 20.;
    const double b = (5. - std::sqrt(5.)) / 20.;
    EXPECT_NEAR(a + 3. * b, 1.0, 1e-14);

    for (const auto& qp : rule) {
        const double xi = qp.coords.x();
        const double eta = qp.coords.y();
        const double zeta = qp.coords.z();
        EXPECT_GE(xi, 0.0);
        EXPECT_GE(eta, 0.0);
        EXPECT_GE(zeta, 0.0);
        EXPECT_LE(xi + eta + zeta, 1.0 + 1e-14);
        // Fourth barycentric coordinate L0 = 1 - xi - eta - zeta >= 0.
        EXPECT_GE(1.0 - xi - eta - zeta, -1e-14);
    }
}

TEST(Quadrature, PermutationSymmetry) {
    // The four points are the four permutations of (a,b,b,b) in barycentric
    // coordinates, so the set of (xi,eta,zeta) must be closed under cycling.
    const auto& rule = TetQuadratureRule();
    const double a = (5. + 3. * std::sqrt(5.)) / 20.;
    const double b = (5. - std::sqrt(5.)) / 20.;

    // Expected: (a,b,b), (b,a,b), (b,b,a), (b,b,b).
    // The last has L0 = a.
    bool found[4] = {false, false, false, false};
    for (const auto& qp : rule) {
        const double xi = qp.coords.x();
        const double eta = qp.coords.y();
        const double zeta = qp.coords.z();
        if (std::fabs(xi - a) < 1e-12 && std::fabs(eta - b) < 1e-12 &&
            std::fabs(zeta - b) < 1e-12) {
            found[0] = true;
        } else if (
            std::fabs(xi - b) < 1e-12 && std::fabs(eta - a) < 1e-12 &&
            std::fabs(zeta - b) < 1e-12
        ) {
            found[1] = true;
        } else if (
            std::fabs(xi - b) < 1e-12 && std::fabs(eta - b) < 1e-12 &&
            std::fabs(zeta - a) < 1e-12
        ) {
            found[2] = true;
        } else if (
            std::fabs(xi - b) < 1e-12 && std::fabs(eta - b) < 1e-12 &&
            std::fabs(zeta - b) < 1e-12
        ) {
            found[3] = true;
        }
    }
    EXPECT_TRUE(found[0]);
    EXPECT_TRUE(found[1]);
    EXPECT_TRUE(found[2]);
    EXPECT_TRUE(found[3]);
}

TEST(Quadrature, ExactThroughDegree2) {
    // constant -> 1/6
    EXPECT_NEAR(
        QuadratureMonomial(0, 0, 0), ExactMonomialIntegral(0, 0, 0), 1e-14
    );
    EXPECT_NEAR(ExactMonomialIntegral(0, 0, 0), 1.0 / 6.0, 1e-15);

    // linear: xi -> 1/24
    EXPECT_NEAR(
        QuadratureMonomial(1, 0, 0), ExactMonomialIntegral(1, 0, 0), 1e-14
    );
    EXPECT_NEAR(ExactMonomialIntegral(1, 0, 0), 1.0 / 24.0, 1e-15);

    // quadratic: xi^2 -> 1/60
    EXPECT_NEAR(
        QuadratureMonomial(2, 0, 0), ExactMonomialIntegral(2, 0, 0), 1e-14
    );
    EXPECT_NEAR(ExactMonomialIntegral(2, 0, 0), 1.0 / 60.0, 1e-15);

    // mixed quadratic: xi*eta -> 1/120
    EXPECT_NEAR(
        QuadratureMonomial(1, 1, 0), ExactMonomialIntegral(1, 1, 0), 1e-14
    );
    EXPECT_NEAR(ExactMonomialIntegral(1, 1, 0), 1.0 / 120.0, 1e-15);

    // All degree-2 monoms via symmetry.
    EXPECT_NEAR(
        QuadratureMonomial(0, 2, 0), ExactMonomialIntegral(0, 2, 0), 1e-14
    );
    EXPECT_NEAR(
        QuadratureMonomial(0, 0, 2), ExactMonomialIntegral(0, 0, 2), 1e-14
    );
    EXPECT_NEAR(
        QuadratureMonomial(1, 0, 1), ExactMonomialIntegral(1, 0, 1), 1e-14
    );
    EXPECT_NEAR(
        QuadratureMonomial(0, 1, 1), ExactMonomialIntegral(0, 1, 1), 1e-14
    );
}

TEST(Quadrature, InexactAtDegree3) {
    // xi^3 exact integral = 3! / 6! = 6 / 720 = 1/120.
    const double exact = ExactMonomialIntegral(3, 0, 0);
    const double approx = QuadratureMonomial(3, 0, 0);
    EXPECT_NEAR(exact, 1.0 / 120.0, 1e-15);
    // Rule must NOT integrate degree 3 exactly.
    EXPECT_GT(std::fabs(approx - exact), 1e-6);
}
