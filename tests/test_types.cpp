// test_types.cpp — unit tests for types.hpp (Dot, FromColumns, DoubleDot).

#include <gtest/gtest.h>

#include "types.hpp"

using namespace matrix_free_fea;

TEST(Types, DotOrthogonality) {
    const Vector3 e0(1., 0., 0.);
    const Vector3 e1(0., 1., 0.);
    const Vector3 e2(0., 0., 1.);
    EXPECT_NEAR(Dot(e0, e1), 0.0, 1e-15);
    EXPECT_NEAR(Dot(e0, e2), 0.0, 1e-15);
    EXPECT_NEAR(Dot(e1, e2), 0.0, 1e-15);
}

TEST(Types, DotSelfEqualsSquaredNorm) {
    const Vector3 a(1.5, -2.0, 0.5);
    EXPECT_NEAR(Dot(a, a), a.squaredNorm(), 1e-15);
}

TEST(Types, DotCommutativity) {
    const Vector3 a(1.5, -2.0, 0.5);
    const Vector3 b(-0.3, 0.7, 1.1);
    EXPECT_NEAR(Dot(a, b), Dot(b, a), 1e-15);
}

TEST(Types, FromColumnsIdentity) {
    const Matrix3 I = FromColumns(
        Vector3(1., 0., 0.), Vector3(0., 1., 0.), Vector3(0., 0., 1.)
    );
    EXPECT_TRUE(I.isIdentity(1e-15));
}

TEST(Types, FromColumnsStoresArguments) {
    const Vector3 c0(1., 2., 3.);
    const Vector3 c1(4., 5., 6.);
    const Vector3 c2(7., 8., 9.);
    const Matrix3 M = FromColumns(c0, c1, c2);
    EXPECT_TRUE(M.col(0).isApprox(c0, 1e-15));
    EXPECT_TRUE(M.col(1).isApprox(c1, 1e-15));
    EXPECT_TRUE(M.col(2).isApprox(c2, 1e-15));
}

TEST(Types, FromColumnsDeterminant) {
    // Known matrix with det = 1*5*9 + 2*6*7 + 3*4*8 - 3*5*7 - 2*4*9 - 1*6*8
    // = 45 + 84 + 96 - 105 - 72 - 48 = 0
    const Matrix3 M = FromColumns(
        Vector3(1., 2., 3.), Vector3(4., 5., 6.), Vector3(7., 8., 9.)
    );
    EXPECT_NEAR(M.determinant(), 0.0, 1e-12);

    // Upper-triangular with det = product of diagonals.
    const Matrix3 T = FromColumns(
        Vector3(2., 0., 0.), Vector3(1., 3., 0.), Vector3(4., 5., 6.)
    );
    EXPECT_NEAR(T.determinant(), 36.0, 1e-12);
}

TEST(Types, DoubleDotCommutativity) {
    Matrix3 A;
    A << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    Matrix3 B;
    B << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9;
    EXPECT_NEAR(DoubleDot(A, B), DoubleDot(B, A), 1e-15);
}

TEST(Types, DoubleDotWithIdentityIsTrace) {
    Matrix3 A;
    A << 1.5, 2., 3., 4., -0.5, 6., 7., 8., 2.0;
    EXPECT_NEAR(DoubleDot(A, Matrix3::Identity()), A.trace(), 1e-15);
}

TEST(Types, DoubleDotMatchesExplicitLoop) {
    Matrix3 A;
    A << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    Matrix3 B;
    B << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9;
    double expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            expected += A(i, j) * B(i, j);
        }
    }
    EXPECT_NEAR(DoubleDot(A, B), expected, 1e-14);
}
