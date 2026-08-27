// test_geometry.cpp — unit tests for geometry.hpp.
//
// All non-trivial checks run on a non-symmetric Jacobian so a J^{-1} vs
// J^{-T} transpose mistake is visible.

#include <cmath>

#include <gtest/gtest.h>

#include "geometry.hpp"
#include "shape_functions.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

namespace {

void SkewedCorners(Vector3 corners[4]) {
    corners[0] = Vector3(0.0, 0.0, 0.0);
    corners[1] = Vector3(2.0, 0.0, 0.0);
    corners[2] = Vector3(0.5, 1.5, 0.0);
    corners[3] = Vector3(0.3, 0.2, 1.0);
}

void ReferenceCorners(Vector3 corners[4]) {
    corners[0] = Vector3(0.0, 0.0, 0.0);
    corners[1] = Vector3(1.0, 0.0, 0.0);
    corners[2] = Vector3(0.0, 1.0, 0.0);
    corners[3] = Vector3(0.0, 0.0, 1.0);
}

}  // namespace

TEST(Geometry, ReferenceTetIdentityJacobian) {
    Vector3 corners[4];
    ReferenceCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    EXPECT_TRUE(geo.jacobian.isIdentity(1e-14));
    EXPECT_TRUE(geo.inverse_jacobian.isIdentity(1e-14));
    EXPECT_NEAR(geo.det_jacobian, 1.0, 1e-14);
}

TEST(Geometry, ColumnStructure) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    EXPECT_TRUE(geo.jacobian.col(0).isApprox(corners[1] - corners[0], 1e-14));
    EXPECT_TRUE(geo.jacobian.col(1).isApprox(corners[2] - corners[0], 1e-14));
    EXPECT_TRUE(geo.jacobian.col(2).isApprox(corners[3] - corners[0], 1e-14));
}

TEST(Geometry, DeterminantEqualsSixTimesVolume) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    // Volume of tet = |det([c1-c0|c2-c0|c3-c0])| / 6.
    const double volume = std::fabs(geo.det_jacobian) / 6.0;
    EXPECT_NEAR(geo.det_jacobian, 6.0 * volume, 1e-14);
    EXPECT_GT(geo.det_jacobian, 0.0);
}

TEST(Geometry, InverseRoundTrip) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    const Matrix3 product = geo.inverse_jacobian * geo.jacobian;
    EXPECT_TRUE(product.isIdentity(1e-12));
}

TEST(Geometry, TranslationInvariance) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo0 = ComputeElementGeometry(corners);

    const Vector3 shift(3.7, -1.2, 0.5);
    for (int i = 0; i < 4; ++i) {
        corners[i] += shift;
    }
    const ElementGeometry geo1 = ComputeElementGeometry(corners);
    EXPECT_TRUE(geo0.jacobian.isApprox(geo1.jacobian, 1e-14));
    EXPECT_NEAR(geo0.det_jacobian, geo1.det_jacobian, 1e-14);
}

TEST(Geometry, ScalingScalesDetByCube) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo0 = ComputeElementGeometry(corners);

    const double s = 2.5;
    for (int i = 0; i < 4; ++i) {
        corners[i] *= s;
    }
    const ElementGeometry geo1 = ComputeElementGeometry(corners);
    EXPECT_NEAR(geo1.det_jacobian, geo0.det_jacobian * s * s * s, 1e-10);
}

TEST(Geometry, OrientationFlipOnCornerSwap) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo0 = ComputeElementGeometry(corners);

    std::swap(corners[1], corners[2]);
    const ElementGeometry geo1 = ComputeElementGeometry(corners);
    EXPECT_NEAR(geo1.det_jacobian, -geo0.det_jacobian, 1e-12);
}

TEST(Geometry, PhysicalGradientsIdentityPullback) {
    Vector3 corners[4];
    ReferenceCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    const Matrix3 J_inv_T = geo.inverse_jacobian.transpose();

    const ShapeFunctionData sf = EvaluateShapeFunctions(0.2, 0.3, 0.1);
    Vector3 dN_dx[kNodesPerTetElement];
    PhysicalGradients(J_inv_T, sf.shape_func_derivatives, dN_dx);

    for (int i = 0; i < kNodesPerTetElement; ++i) {
        ExpectVector3Near(
            dN_dx[i], sf.shape_func_derivatives[i], 1e-14, "node"
        );
    }
}

TEST(Geometry, PhysicalGradientsPartitionOfUnity) {
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    const Matrix3 J_inv_T = geo.inverse_jacobian.transpose();

    // Physical nodal coordinates (affine map of reference nodes).
    const auto& ref = ReferenceNodeCoords();
    Vector3 physical_nodes[kNodesPerTetElement];
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        const Vector3 xi(ref[i][0], ref[i][1], ref[i][2]);
        physical_nodes[i] = corners[0] + geo.jacobian * xi;  // affine map
    }

    const ShapeFunctionData sf = EvaluateShapeFunctions(0.25, 0.2, 0.15);
    Vector3 dN_dx[kNodesPerTetElement];
    PhysicalGradients(J_inv_T, sf.shape_func_derivatives, dN_dx);

    Vector3 sum = Vector3::Zero();
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        sum += dN_dx[i];
    }
    ExpectVector3Near(sum, Vector3::Zero(), 1e-12);
}

TEST(Geometry, PhysicalGradientsReproduceIdentity) {
    // sum_i dN_i/dx (outer) x_i == I  -- catches J^{-1} vs J^{-T} bugs.
    Vector3 corners[4];
    SkewedCorners(corners);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    ASSERT_FALSE(geo.jacobian.isApprox(geo.jacobian.transpose(), 1e-6))
        << "test requires a non-symmetric Jacobian";

    const Matrix3 J_inv_T = geo.inverse_jacobian.transpose();

    const auto& ref = ReferenceNodeCoords();
    Vector3 physical_nodes[kNodesPerTetElement];
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        const Vector3 xi(ref[i][0], ref[i][1], ref[i][2]);
        physical_nodes[i] = corners[0] + geo.jacobian * xi;
    }

    const ShapeFunctionData sf = EvaluateShapeFunctions(0.2, 0.15, 0.1);
    Vector3 dN_dx[kNodesPerTetElement];
    PhysicalGradients(J_inv_T, sf.shape_func_derivatives, dN_dx);

    Matrix3 sum = Matrix3::Zero();
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        // Outer product: dN_dx[i] * physical_nodes[i]^T
        sum += dN_dx[i] * physical_nodes[i].transpose();
    }
    EXPECT_TRUE(sum.isIdentity(1e-10)) << "sum =\n" << sum;
}
