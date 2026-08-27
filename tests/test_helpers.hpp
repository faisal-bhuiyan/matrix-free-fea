#pragma once

#include <array>
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "assembly.hpp"
#include "element_kernel.hpp"
#include "geometry.hpp"
#include "mesh.hpp"
#include "shape_functions.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

namespace matrix_free_fea {
namespace test {

//---------------------------------------------------------------------------
// Material builders
//---------------------------------------------------------------------------

inline std::vector<std::array<LinearElasticMaterial, 4>> UniformMaterial(
    int num_elements, double lambda, double mu
) {
    std::vector<std::array<LinearElasticMaterial, 4>> mat(
        static_cast<std::size_t>(num_elements)
    );
    for (int e = 0; e < num_elements; ++e) {
        for (int q = 0; q < 4; ++q) {
            mat[static_cast<std::size_t>(e)][q] = {lambda, mu};
        }
    }
    return mat;
}

inline std::vector<std::array<LinearElasticMaterial, 4>> VaryingMaterial(
    int num_elements
) {
    std::vector<std::array<LinearElasticMaterial, 4>> mat(
        static_cast<std::size_t>(num_elements)
    );
    const LinearElasticMaterial patterns[4] = {
        {1.0e5, 0.5e5},
        {1.2e5, 0.4e5},
        {0.9e5, 0.6e5},
        {1.1e5, 0.5e5},
    };
    for (int e = 0; e < num_elements; ++e) {
        for (int q = 0; q < 4; ++q) {
            mat[static_cast<std::size_t>(e)][q] = patterns[q];
        }
    }
    return mat;
}

//---------------------------------------------------------------------------
// Field builders
//---------------------------------------------------------------------------

inline std::vector<Vector3> MakeLinearField(
    const Mesh& mesh, const Matrix3& A, const Vector3& b
) {
    std::vector<Vector3> u(static_cast<std::size_t>(mesh.NumNodes()));
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        u[static_cast<std::size_t>(n)] = A * mesh.node_coords[n] + b;
    }
    return u;
}

inline std::vector<Vector3> MakeRigidTranslation(
    const Mesh& mesh, const Vector3& t
) {
    return std::vector<Vector3>(static_cast<std::size_t>(mesh.NumNodes()), t);
}

inline std::vector<Vector3> MakeRigidRotation(
    const Mesh& mesh, const Vector3& omega
) {
    std::vector<Vector3> u(static_cast<std::size_t>(mesh.NumNodes()));
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        u[static_cast<std::size_t>(n)] = omega.cross(mesh.node_coords[n]);
    }
    return u;
}

inline std::vector<Vector3> MakeRandomField(const Mesh& mesh, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Vector3> u(static_cast<std::size_t>(mesh.NumNodes()));
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        u[static_cast<std::size_t>(n)] =
            Vector3(dist(rng), dist(rng), dist(rng));
    }
    return u;
}

//---------------------------------------------------------------------------
// Reference-space nodal coordinates (EvaluateShapeFunctions ordering)
//---------------------------------------------------------------------------

inline const double (&ReferenceNodeCoords()) [kNodesPerTetElement][3] {
    static const double coords[kNodesPerTetElement][3] = {
        {0.0, 0.0, 0.0},  // 0: corner 0
        {1.0, 0.0, 0.0},  // 1: corner 1
        {0.0, 1.0, 0.0},  // 2: corner 2
        {0.0, 0.0, 1.0},  // 3: corner 3
        {0.5, 0.0, 0.0},  // 4: mid(0,1)
        {0.5, 0.5, 0.0},  // 5: mid(1,2)
        {0.0, 0.5, 0.0},  // 6: mid(0,2)
        {0.0, 0.0, 0.5},  // 7: mid(0,3)
        {0.5, 0.0, 0.5},  // 8: mid(1,3)
        {0.0, 0.5, 0.5},  // 9: mid(2,3)
    };
    return coords;
}

//---------------------------------------------------------------------------
// Dense matrix assembly (via repeated matrix-free applications)
//---------------------------------------------------------------------------

inline Eigen::MatrixXd AssembleElementMatrix(
    const Vector3 corners[4], const LinearElasticMaterial material_at_QP[4]
) {
    Eigen::MatrixXd K(kElementDOFs, kElementDOFs);
    K.setZero();
    for (int col = 0; col < kElementDOFs; ++col) {
        Vector3 u_local[kNodesPerTetElement];
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            u_local[i] = Vector3::Zero();
        }
        u_local[col / kDimensions][col % kDimensions] = 1.0;

        Vector3 y_local[kNodesPerTetElement];
        ComputeElementLinearElasticityOperator(
            corners, u_local, material_at_QP, y_local
        );

        for (int row = 0; row < kElementDOFs; ++row) {
            K(row, col) = y_local[row / kDimensions][row % kDimensions];
        }
    }
    return K;
}

inline Eigen::MatrixXd AssembleGlobalMatrix(
    const Mesh& mesh,
    const std::vector<std::array<LinearElasticMaterial, 4>>& material
) {
    const int ndof = mesh.NumNodes() * kDimensions;
    Eigen::MatrixXd K(ndof, ndof);
    K.setZero();
    for (int col = 0; col < ndof; ++col) {
        std::vector<Vector3> u(
            static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
        );
        u[static_cast<std::size_t>(col / kDimensions)][col % kDimensions] = 1.0;

        std::vector<Vector3> y;
        ApplyGlobalLinearElasticityOperator(mesh, u, material, y);

        for (int row = 0; row < ndof; ++row) {
            K(row, col) = y[static_cast<std::size_t>(row / kDimensions)]
                           [row % kDimensions];
        }
    }
    return K;
}

//---------------------------------------------------------------------------
// Matchers / checkers
//---------------------------------------------------------------------------

inline void ExpectVector3Near(
    const Vector3& actual, const Vector3& expected, double tol,
    const char* label = ""
) {
    EXPECT_NEAR(actual.x(), expected.x(), tol) << label;
    EXPECT_NEAR(actual.y(), expected.y(), tol) << label;
    EXPECT_NEAR(actual.z(), expected.z(), tol) << label;
}

/**
 * @brief Check connectivity range, uniqueness, positive det(J), and that
 *        every edge node sits at the midpoint of its corner pair.
 */
inline void CheckMeshInvariants(const Mesh& mesh) {
    ASSERT_GT(mesh.NumNodes(), 0);
    ASSERT_GT(mesh.NumElements(), 0);

    for (int e = 0; e < mesh.NumElements(); ++e) {
        const auto& conn = mesh.elements[static_cast<std::size_t>(e)];

        // Indices in range, no duplicates within the element.
        std::set<int> seen;
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            EXPECT_GE(conn[i], 0) << "elem " << e << " slot " << i;
            EXPECT_LT(conn[i], mesh.NumNodes())
                << "elem " << e << " slot " << i;
            EXPECT_TRUE(seen.insert(conn[i]).second)
                << "elem " << e << " duplicate node " << conn[i];
        }

        // Positive Jacobian determinant.
        Vector3 corners[4];
        mesh.ElementCorners(e, corners);
        const ElementGeometry geo = ComputeElementGeometry(corners);
        EXPECT_GT(geo.det_jacobian, 0.0) << "elem " << e;

        // Edge nodes at true midpoints of their corner pairs.
        Vector3 nodes[kNodesPerTetElement];
        mesh.ElementNodes(e, nodes);
        for (int edge = 0; edge < 6; ++edge) {
            const Vector3 expected_mid =
                (nodes[kEdgePairs[edge][0]] + nodes[kEdgePairs[edge][1]]) * 0.5;
            ExpectVector3Near(
                nodes[4 + edge], expected_mid, 1e-12,
                ("elem " + std::to_string(e) + " edge " + std::to_string(edge))
                    .c_str()
            );
        }
    }
}

}  // namespace test
}  // namespace matrix_free_fea
