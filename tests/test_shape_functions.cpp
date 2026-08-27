// test_shape_functions.cpp
// Sanity checks for shape_functions.hpp.
//
// Checks:
//   1. Partition of unity: sum_i N_i(p) == 1 at several points.
//   2. Gradient partition of unity: sum_i dN_i/dxi == 0.
//   3. Kronecker-delta property at nodal coordinates.
//   4. Finite-difference validation of analytic derivatives.
//   5. Linear and quadratic completeness.

#include <cmath>

#include <gtest/gtest.h>

#include "shape_functions.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

namespace {

void ExpectPartitionOfUnity(double xi, double eta, double zeta) {
    ShapeFunctionData s = EvaluateShapeFunctions(xi, eta, zeta);
    double sumN = 0.0;
    Vector3 sumDN(0.0, 0.0, 0.0);
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        sumN += s.shape_func_values[i];
        sumDN += s.shape_func_derivatives[i];
    }
    EXPECT_NEAR(sumN, 1.0, 1e-9)
        << "at (" << xi << "," << eta << "," << zeta << ")";
    EXPECT_NEAR(sumDN.x(), 0.0, 1e-9)
        << "at (" << xi << "," << eta << "," << zeta << ")";
    EXPECT_NEAR(sumDN.y(), 0.0, 1e-9)
        << "at (" << xi << "," << eta << "," << zeta << ")";
    EXPECT_NEAR(sumDN.z(), 0.0, 1e-9)
        << "at (" << xi << "," << eta << "," << zeta << ")";
}

}  // namespace

TEST(ShapeFunctions, PartitionOfUnity) {
    ExpectPartitionOfUnity(0.2, 0.3, 0.1);
    ExpectPartitionOfUnity(
        0.585410196624968, 0.138196601125011, 0.138196601125011
    );
    ExpectPartitionOfUnity(0.0, 0.0, 0.0);
    ExpectPartitionOfUnity(1.0, 0.0, 0.0);
    ExpectPartitionOfUnity(0.25, 0.25, 0.25);
}

TEST(ShapeFunctions, KroneckerDelta) {
    const auto& node_coords = ReferenceNodeCoords();
    for (int j = 0; j < kNodesPerTetElement; ++j) {
        ShapeFunctionData s = EvaluateShapeFunctions(
            node_coords[j][0], node_coords[j][1], node_coords[j][2]
        );
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(s.shape_func_values[i], expected, 1e-9)
                << "shape_func_values[" << i << "] at node " << j;
        }
    }
}

TEST(ShapeFunctions, FiniteDifferenceGradients) {
    const double h = 1e-7;
    const double xi = 0.2;
    const double eta = 0.15;
    const double zeta = 0.1;

    ShapeFunctionData s = EvaluateShapeFunctions(xi, eta, zeta);

    auto value_at = [](double x, double y, double z, int i) {
        return EvaluateShapeFunctions(x, y, z).shape_func_values[i];
    };

    for (int i = 0; i < kNodesPerTetElement; ++i) {
        const double dN_dxi =
            (value_at(xi + h, eta, zeta, i) - value_at(xi - h, eta, zeta, i)) /
            (2.0 * h);
        const double dN_deta =
            (value_at(xi, eta + h, zeta, i) - value_at(xi, eta - h, zeta, i)) /
            (2.0 * h);
        const double dN_dzeta =
            (value_at(xi, eta, zeta + h, i) - value_at(xi, eta, zeta - h, i)) /
            (2.0 * h);

        EXPECT_NEAR(s.shape_func_derivatives[i].x(), dN_dxi, 1e-6)
            << "node " << i << " d/dxi";
        EXPECT_NEAR(s.shape_func_derivatives[i].y(), dN_deta, 1e-6)
            << "node " << i << " d/deta";
        EXPECT_NEAR(s.shape_func_derivatives[i].z(), dN_dzeta, 1e-6)
            << "node " << i << " d/dzeta";
    }
}

TEST(ShapeFunctions, LinearCompleteness) {
    // sum_i N_i * X_i == X for any point in the reference tet.
    const auto& nodes = ReferenceNodeCoords();
    const double pts[][3] = {
        {0.2, 0.3, 0.1},
        {0.1, 0.1, 0.1},
        {0.25, 0.25, 0.25},
    };
    for (const auto& p : pts) {
        ShapeFunctionData s = EvaluateShapeFunctions(p[0], p[1], p[2]);
        Vector3 sum = Vector3::Zero();
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            sum += s.shape_func_values[i] *
                   Vector3(nodes[i][0], nodes[i][1], nodes[i][2]);
        }
        ExpectVector3Near(sum, Vector3(p[0], p[1], p[2]), 1e-12);
    }
}

TEST(ShapeFunctions, QuadraticCompleteness) {
    // sum_i N_i * f(X_i) == f(X) for a quadratic f.
    // f(x,y,z) = x^2 + 2*y*z + 3*x*y + z
    auto f = [](double x, double y, double z) {
        return x * x + 2.0 * y * z + 3.0 * x * y + z;
    };

    const auto& nodes = ReferenceNodeCoords();
    const double pts[][3] = {
        {0.2, 0.3, 0.1},
        {0.1, 0.1, 0.1},
        {0.25, 0.25, 0.25},
        {0.4, 0.1, 0.05},
    };
    for (const auto& p : pts) {
        ShapeFunctionData s = EvaluateShapeFunctions(p[0], p[1], p[2]);
        double sum = 0.0;
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            sum += s.shape_func_values[i] *
                   f(nodes[i][0], nodes[i][1], nodes[i][2]);
        }
        EXPECT_NEAR(sum, f(p[0], p[1], p[2]), 1e-12)
            << "at (" << p[0] << "," << p[1] << "," << p[2] << ")";
    }
}

TEST(ShapeFunctions, EdgeNodePairing) {
    // Edge midpoints must be exactly halfway between their corner pairs
    // in reference space, matching kEdgePairs.
    const auto& nodes = ReferenceNodeCoords();
    for (int e = 0; e < 6; ++e) {
        const int p = kEdgePairs[e][0];
        const int q = kEdgePairs[e][1];
        const int mid = 4 + e;
        EXPECT_NEAR(nodes[mid][0], 0.5 * (nodes[p][0] + nodes[q][0]), 1e-15);
        EXPECT_NEAR(nodes[mid][1], 0.5 * (nodes[p][1] + nodes[q][1]), 1e-15);
        EXPECT_NEAR(nodes[mid][2], 0.5 * (nodes[p][2] + nodes[q][2]), 1e-15);
    }
}
