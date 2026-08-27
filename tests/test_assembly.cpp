// test_assembly.cpp
// Validation for mesh.hpp + assembly.hpp.
//
// Existing patch / force-balance / symmetry tests, plus gather/scatter
// isolation, linearity, and the y_global reset contract.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "assembly.hpp"
#include "element_kernel.hpp"
#include "mesh.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

namespace {

void MakeDefaultLinearField(const Mesh& mesh, std::vector<Vector3>& u) {
    Matrix3 A;
    A << 0.02, 0.01, -0.005, 0.01, -0.03, 0.004, -0.005, 0.004, 0.015;
    const Vector3 b(0.001, -0.002, 0.0005);
    u = MakeLinearField(mesh, A, b);
}

}  // namespace

TEST(Assembly, CornerNodeEquilibrium) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    std::vector<Vector3> u_global;
    MakeDefaultLinearField(mesh, u_global);

    std::vector<Vector3> y_global;
    ApplyGlobalOperator(mesh, u_global, material, y_global);

    const int corner_nodes[] = {0, 1, 2, 3, 4};
    for (int n : corner_nodes) {
        ExpectVector3Near(y_global[n], Vector3::Zero(), 1e-6);
    }
}

TEST(Assembly, GlobalForceBalance) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    std::vector<Vector3> u_global;
    MakeDefaultLinearField(mesh, u_global);

    std::vector<Vector3> y_global;
    ApplyGlobalOperator(mesh, u_global, material, y_global);

    Vector3 total = Vector3::Zero();
    for (const Vector3& y : y_global) {
        total += y;
    }
    ExpectVector3Near(total, Vector3::Zero(), 1e-6);
}

TEST(Assembly, GlobalSymmetry) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const auto u_global = MakeRandomField(mesh, 42);
    const auto v_global = MakeRandomField(mesh, 43);

    std::vector<Vector3> Ku, Kv;
    ApplyGlobalOperator(mesh, u_global, material, Ku);
    ApplyGlobalOperator(mesh, v_global, material, Kv);

    double vTKu = 0.0;
    double uTKv = 0.0;
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        vTKu += Dot(v_global[n], Ku[n]);
        uTKv += Dot(u_global[n], Kv[n]);
    }

    const double diff = std::fabs(vTKu - uTKv);
    const double scale = std::max(std::fabs(vTKu), std::fabs(uTKv));
    EXPECT_LT(diff, 1e-8 * std::max(scale, 1.0))
        << "v^T(Ku) = " << vTKu << ", u^T(Kv) = " << uTKv;
}

TEST(Assembly, SingleElementMatchesKernel) {
    Mesh mesh = MakeSkewedTetMesh();
    ASSERT_EQ(mesh.NumElements(), 1);

    LinearElasticMaterial mat[4];
    for (int q = 0; q < 4; ++q) {
        mat[q] = {1.0e5, 0.5e5};
    }
    const auto material = UniformMaterial(1, 1.0e5, 0.5e5);

    Vector3 corners[4];
    mesh.ElementCorners(0, corners);
    Vector3 nodes[kNodesPerTetElement];
    mesh.ElementNodes(0, nodes);

    Vector3 u_local[kNodesPerTetElement];
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        u_local[i] = nodes[i] * 0.01;
    }

    Vector3 y_kernel[kNodesPerTetElement];
    ComputeElementOperator(corners, u_local, mat, y_kernel);

    std::vector<Vector3> u_global(mesh.NumNodes());
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        u_global[mesh.elements[0][i]] = u_local[i];
    }
    std::vector<Vector3> y_global;
    ApplyGlobalOperator(mesh, u_global, material, y_global);

    for (int i = 0; i < kNodesPerTetElement; ++i) {
        ExpectVector3Near(y_global[mesh.elements[0][i]], y_kernel[i], 1e-10);
    }
}

TEST(Assembly, DisconnectedPairIsolatesScatter) {
    Mesh mesh = MakeDisconnectedPairMesh();
    ASSERT_EQ(mesh.NumElements(), 2);

    const auto material = UniformMaterial(2, 1.0e5, 0.5e5);
    auto u_global = MakeRandomField(mesh, 99);

    std::vector<Vector3> y_global;
    ApplyGlobalOperator(mesh, u_global, material, y_global);

    // Independently compute each element's contribution.
    for (int e = 0; e < 2; ++e) {
        Vector3 corners[4];
        mesh.ElementCorners(e, corners);
        Vector3 u_local[kNodesPerTetElement];
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            u_local[i] = u_global[mesh.elements[e][i]];
        }
        LinearElasticMaterial mat[4];
        for (int q = 0; q < 4; ++q) {
            mat[q] = material[e][q];
        }
        Vector3 y_local[kNodesPerTetElement];
        ComputeElementOperator(corners, u_local, mat, y_local);

        for (int i = 0; i < kNodesPerTetElement; ++i) {
            ExpectVector3Near(y_global[mesh.elements[e][i]], y_local[i], 1e-10);
        }
    }
}

TEST(Assembly, GlobalLinearity) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const auto u1 = MakeRandomField(mesh, 1);
    const auto u2 = MakeRandomField(mesh, 2);
    const double a = 1.7;
    const double b = -0.4;

    std::vector<Vector3> u_combo(mesh.NumNodes());
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        u_combo[n] = a * u1[n] + b * u2[n];
    }

    std::vector<Vector3> y1, y2, y_combo;
    ApplyGlobalOperator(mesh, u1, material, y1);
    ApplyGlobalOperator(mesh, u2, material, y2);
    ApplyGlobalOperator(mesh, u_combo, material, y_combo);

    for (int n = 0; n < mesh.NumNodes(); ++n) {
        ExpectVector3Near(y_combo[n], a * y1[n] + b * y2[n], 1e-6);
    }
}

TEST(Assembly, ZeroInputGivesZeroOutput) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);
    const std::vector<Vector3> u(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );
    std::vector<Vector3> y;
    ApplyGlobalOperator(mesh, u, material, y);
    ASSERT_EQ(static_cast<int>(y.size()), mesh.NumNodes());
    for (const auto& yi : y) {
        ExpectVector3Near(yi, Vector3::Zero(), 1e-15);
    }
}

TEST(Assembly, OutputIsResizedAndZeroed) {
    Mesh mesh = MakeTwoElementPatchMesh();
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);
    const std::vector<Vector3> u(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );

    // Pre-populate with wrong size and garbage values.
    std::vector<Vector3> y(3, Vector3(999., 999., 999.));
    ApplyGlobalOperator(mesh, u, material, y);

    ASSERT_EQ(static_cast<int>(y.size()), mesh.NumNodes());
    for (const auto& yi : y) {
        ExpectVector3Near(yi, Vector3::Zero(), 1e-15);
    }
}
