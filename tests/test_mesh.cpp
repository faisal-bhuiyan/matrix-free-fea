// test_mesh.cpp — unit tests for mesh.hpp accessors and factory invariants.

#include <set>

#include <gtest/gtest.h>

#include "geometry.hpp"
#include "mesh.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

TEST(Mesh, EmptyMeshCounts) {
    Mesh mesh;
    EXPECT_EQ(mesh.NumNodes(), 0);
    EXPECT_EQ(mesh.NumElements(), 0);
}

TEST(Mesh, ReferenceTetCounts) {
    Mesh mesh = MakeReferenceTetMesh();
    EXPECT_EQ(mesh.NumElements(), 1);
    EXPECT_EQ(mesh.NumNodes(), 10);  // 4 corners + 6 midpoints
}

TEST(Mesh, ElementCornersMatchConnectivity) {
    Mesh mesh = MakeSkewedTetMesh();
    Vector3 corners[4];
    mesh.ElementCorners(0, corners);
    for (int i = 0; i < 4; ++i) {
        ExpectVector3Near(
            corners[i], mesh.node_coords[mesh.elements[0][i]], 1e-15
        );
    }
}

TEST(Mesh, ElementNodesMatchConnectivity) {
    Mesh mesh = MakeSkewedTetMesh();
    Vector3 nodes[kNodesPerTetElement];
    mesh.ElementNodes(0, nodes);
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        ExpectVector3Near(
            nodes[i], mesh.node_coords[mesh.elements[0][i]], 1e-15
        );
    }
}

TEST(Mesh, ElementCornersAgreeWithFirstFourNodes) {
    Mesh mesh = MakeTwoElementPatchMesh();
    for (int e = 0; e < mesh.NumElements(); ++e) {
        Vector3 corners[4];
        Vector3 nodes[kNodesPerTetElement];
        mesh.ElementCorners(e, corners);
        mesh.ElementNodes(e, nodes);
        for (int i = 0; i < 4; ++i) {
            ExpectVector3Near(corners[i], nodes[i], 1e-15);
        }
    }
}

TEST(Mesh, InvariantsReferenceTet) {
    CheckMeshInvariants(MakeReferenceTetMesh());
}

TEST(Mesh, InvariantsSkewedTet) {
    CheckMeshInvariants(MakeSkewedTetMesh());
}

TEST(Mesh, InvariantsDisconnectedPair) {
    CheckMeshInvariants(MakeDisconnectedPairMesh());
}

TEST(Mesh, InvariantsTwoElementPatch) {
    CheckMeshInvariants(MakeTwoElementPatchMesh());
}

TEST(Mesh, InvariantsCubeMesh) {
    CheckMeshInvariants(MakeCubeMesh(1));
    CheckMeshInvariants(MakeCubeMesh(2));
}

TEST(Mesh, TwoElementPatchSharedFaceNodes) {
    Mesh mesh = MakeTwoElementPatchMesh();
    ASSERT_EQ(mesh.NumElements(), 2);

    std::set<int> e0(mesh.elements[0].begin(), mesh.elements[0].end());
    std::set<int> e1(mesh.elements[1].begin(), mesh.elements[1].end());
    std::set<int> shared;
    for (int n : e0) {
        if (e1.count(n)) {
            shared.insert(n);
        }
    }
    // Shared face ABC plus its three edge midpoints: {0,1,2,5,6,7}.
    const std::set<int> expected = {0, 1, 2, 5, 6, 7};
    EXPECT_EQ(shared, expected);
}

TEST(Mesh, CubeMeshElementCount) {
    EXPECT_EQ(MakeCubeMesh(1).NumElements(), 6);
    EXPECT_EQ(MakeCubeMesh(2).NumElements(), 48);
    EXPECT_EQ(MakeCubeMesh(3).NumElements(), 162);
}

TEST(Mesh, DisconnectedPairSharesNoNodes) {
    Mesh mesh = MakeDisconnectedPairMesh();
    ASSERT_EQ(mesh.NumElements(), 2);
    std::set<int> e0(mesh.elements[0].begin(), mesh.elements[0].end());
    for (int n : mesh.elements[1]) {
        EXPECT_EQ(e0.count(n), 0u) << "shared node " << n;
    }
}
