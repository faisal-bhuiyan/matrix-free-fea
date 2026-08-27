// test_element_kernel.cpp
// Validation for geometry.hpp + element_kernel.hpp.
//
// Existing property tests (rigid translation, force equilibrium, corner
// nodes under linear field) plus linearity, symmetry, PSD, null-space
// dimension, rigid rotation, uniaxial stretch, and a cross-check against
// an independent naive B^T D B implementation.

#include <cmath>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "element_kernel.hpp"
#include "geometry.hpp"
#include "quadrature.hpp"
#include "shape_functions.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

namespace {

void MakeTestElement(Vector3 corners[4], Vector3 all_nodes[10]) {
    // Non-symmetric Jacobian so J^{-1} vs J^{-T} mistakes are visible.
    corners[0] = Vector3(0., 0., 0.);
    corners[1] = Vector3(2., 0., 0.);
    corners[2] = Vector3(0.5, 1.5, 0.);
    corners[3] = Vector3(0.3, 0.2, 1.);

    all_nodes[0] = corners[0];
    all_nodes[1] = corners[1];
    all_nodes[2] = corners[2];
    all_nodes[3] = corners[3];
    all_nodes[4] = (corners[0] + corners[1]) * 0.5;
    all_nodes[5] = (corners[1] + corners[2]) * 0.5;
    all_nodes[6] = (corners[0] + corners[2]) * 0.5;
    all_nodes[7] = (corners[0] + corners[3]) * 0.5;
    all_nodes[8] = (corners[1] + corners[3]) * 0.5;
    all_nodes[9] = (corners[2] + corners[3]) * 0.5;
}

void UniformMat(LinearElasticMaterial mat[4], double lambda, double mu) {
    for (int q = 0; q < 4; ++q) {
        mat[q] = {lambda, mu};
    }
}

// Independent textbook B^T D B element stiffness (Voigt notation).
// Builds K_e by quadrature of B^T C B * det(J) * w, then applies to u.
void NaiveBtdBOperator(
    const Vector3 corners[4], const Vector3 u_local[kNodesPerTetElement],
    const LinearElasticMaterial material_at_QP[4],
    Vector3 y_local[kNodesPerTetElement]
) {
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        y_local[i] = Vector3::Zero();
    }

    const ElementGeometry geo = ComputeElementGeometry(corners);
    const Matrix3 J_inv_T = geo.inverse_jacobian.transpose();
    const auto& rule = TetrahedronQuadratureRule();

    // Flatten u into a 30-vector
    Eigen::VectorXd u(kElementDOFs);
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        for (int d = 0; d < kDimensions; ++d) {
            u(i * kDimensions + d) = u_local[i][d];
        }
    }

    Eigen::MatrixXd K(kElementDOFs, kElementDOFs);
    K.setZero();

    for (int qp_idx = 0; qp_idx < 4; ++qp_idx) {
        const auto& qp = rule[static_cast<std::size_t>(qp_idx)];
        const ShapeFunctionData sf =
            EvaluateShapeFunctions(qp.coords.x(), qp.coords.y(), qp.coords.z());
        Vector3 dN_dx[kNodesPerTetElement];
        PhysicalGradients(J_inv_T, sf.shape_func_derivatives, dN_dx);

        // B is 6 x 30 (Voigt)
        Eigen::MatrixXd B(6, kElementDOFs);
        B.setZero();
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            const double dx = dN_dx[i].x();
            const double dy = dN_dx[i].y();
            const double dz = dN_dx[i].z();
            const int c = i * kDimensions;
            // eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy
            B(0, c + 0) = dx;
            B(1, c + 1) = dy;
            B(2, c + 2) = dz;
            B(3, c + 1) = dz;
            B(3, c + 2) = dy;
            B(4, c + 0) = dz;
            B(4, c + 2) = dx;
            B(5, c + 0) = dy;
            B(5, c + 1) = dx;
        }

        const double lambda = material_at_QP[qp_idx].lambda;
        const double mu = material_at_QP[qp_idx].mu;
        // Isotropic elasticity matrix in Voigt (engineering shear)
        Eigen::MatrixXd C(6, 6);
        C.setZero();
        C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2. * mu;
        C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
        C(3, 3) = C(4, 4) = C(5, 5) = mu;

        const double scale = qp.weight * geo.det_jacobian;
        K += B.transpose() * C * B * scale;
    }

    const Eigen::VectorXd y = K * u;
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        y_local[i] = Vector3(
            y(i * kDimensions + 0), y(i * kDimensions + 1),
            y(i * kDimensions + 2)
        );
    }
}

double LocalDot(
    const Vector3 a[kNodesPerTetElement], const Vector3 b[kNodesPerTetElement]
) {
    double s = 0.;
    for (int i = 0; i < kNodesPerTetElement; ++i) {
        s += Dot(a[i], b[i]);
    }
    return s;
}

}  // namespace

TEST(ElementKernel, RigidBodyTranslation) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    Vector3 u_local[10];
    const Vector3 translation(0.37, -1.2, 0.5);
    for (int i = 0; i < 10; ++i) {
        u_local[i] = translation;
    }

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    Vector3 y_local[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_local);

    for (int i = 0; i < 10; ++i) {
        ExpectVector3Near(y_local[i], Vector3::Zero(), 1e-8);
    }
}

TEST(ElementKernel, ForceEquilibrium) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    const double alpha = 0.1;
    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = all_nodes[i] * alpha;
    }

    LinearElasticMaterial mat[4] = {
        {1.e5, 0.5e5}, {1.2e5, 0.4e5}, {0.9e5, 0.6e5}, {1.1e5, 0.5e5}
    };

    Vector3 y_local[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_local);

    Vector3 sum = Vector3::Zero();
    for (int i = 0; i < 10; ++i) {
        sum += y_local[i];
    }
    ExpectVector3Near(sum, Vector3::Zero(), 1e-6);
}

TEST(ElementKernel, CornerNodesUnderLinearField) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    Matrix3 A;
    A << 0.2, 0.1, -0.05, 0.1, -0.3, 0.04, -0.05, 0.04, 0.15;

    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = A * all_nodes[i];
    }

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    Vector3 y_local[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_local);

    for (int i = 0; i < 4; ++i) {
        ExpectVector3Near(y_local[i], Vector3::Zero(), 1e-6);
    }
}

TEST(ElementKernel, RigidBodyRotation) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    const Vector3 omega(0.2, -0.1, 0.3);
    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = omega.cross(all_nodes[i]);
    }

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    Vector3 y_local[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_local);

    for (int i = 0; i < 10; ++i) {
        ExpectVector3Near(y_local[i], Vector3::Zero(), 1e-7);
    }
}

TEST(ElementKernel, LinearityInDisplacement) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    Vector3 u1[10], u2[10];
    for (int i = 0; i < 10; ++i) {
        u1[i] = all_nodes[i] * 0.1;
        u2[i] = Vector3(0.02, -0.03, 0.01) + all_nodes[i] * 0.05;
    }

    const double a = 2.5;
    const double b = -1.3;
    Vector3 u_combo[10], y1[10], y2[10], y_combo[10], y_expected[10];
    for (int i = 0; i < 10; ++i) {
        u_combo[i] = a * u1[i] + b * u2[i];
    }

    ComputeElementLinearElasticityOperator(corners, u1, mat, y1);
    ComputeElementLinearElasticityOperator(corners, u2, mat, y2);
    ComputeElementLinearElasticityOperator(corners, u_combo, mat, y_combo);

    for (int i = 0; i < 10; ++i) {
        y_expected[i] = a * y1[i] + b * y2[i];
        ExpectVector3Near(y_combo[i], y_expected[i], 1e-6);
    }
}

TEST(ElementKernel, LinearityInMaterial) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = all_nodes[i] * 0.1;
    }

    LinearElasticMaterial mat1[4], mat2[4], mat_combo[4];
    UniformMat(mat1, 1.e5, 0.5e5);
    UniformMat(mat2, 2.e5, 0.3e5);
    const double a = 0.4;
    const double b = 0.6;
    for (int q = 0; q < 4; ++q) {
        mat_combo[q] = {
            a * mat1[q].lambda + b * mat2[q].lambda,
            a * mat1[q].mu + b * mat2[q].mu,
        };
    }

    Vector3 y1[10], y2[10], y_combo[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat1, y1);
    ComputeElementLinearElasticityOperator(corners, u_local, mat2, y2);
    ComputeElementLinearElasticityOperator(
        corners, u_local, mat_combo, y_combo
    );

    for (int i = 0; i < 10; ++i) {
        ExpectVector3Near(y_combo[i], a * y1[i] + b * y2[i], 1e-5);
    }
}

TEST(ElementKernel, ElementSymmetry) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1., 1.);
    Vector3 u[10], v[10];
    for (int i = 0; i < 10; ++i) {
        u[i] = Vector3(dist(rng), dist(rng), dist(rng));
        v[i] = Vector3(dist(rng), dist(rng), dist(rng));
    }

    Vector3 Ku[10], Kv[10];
    ComputeElementLinearElasticityOperator(corners, u, mat, Ku);
    ComputeElementLinearElasticityOperator(corners, v, mat, Kv);

    const double vKu = LocalDot(v, Ku);
    const double uKv = LocalDot(u, Kv);
    EXPECT_NEAR(
        vKu, uKv, 1e-8 * std::max({1., std::fabs(vKu), std::fabs(uKv)})
    );
}

TEST(ElementKernel, PositiveSemiDefinite) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-1., 1.);
    for (int trial = 0; trial < 20; ++trial) {
        Vector3 u[10];
        for (int i = 0; i < 10; ++i) {
            u[i] = Vector3(dist(rng), dist(rng), dist(rng));
        }
        Vector3 Ku[10];
        ComputeElementLinearElasticityOperator(corners, u, mat, Ku);
        EXPECT_GE(LocalDot(u, Ku), -1e-8);
    }
}

TEST(ElementKernel, NullSpaceDimension) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    const Eigen::MatrixXd K = AssembleElementMatrix(corners, mat);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    ASSERT_EQ(solver.info(), Eigen::Success);

    const auto& evals = solver.eigenvalues();
    int null_count = 0;
    const double tol = 1e-6 * evals.cwiseAbs().maxCoeff();
    for (int i = 0; i < evals.size(); ++i) {
        if (std::fabs(evals[i]) <= tol) {
            ++null_count;
        } else {
            EXPECT_GT(evals[i], 0.) << "eigenvalue " << i;
        }
    }
    EXPECT_EQ(null_count, 6);
}

TEST(ElementKernel, MatchesNaiveBtdB) {
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    LinearElasticMaterial mat[4];
    UniformMat(mat, 1.e5, 0.5e5);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = Vector3(dist(rng), dist(rng), dist(rng));
    }

    Vector3 y_mf[10], y_naive[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_mf);
    NaiveBtdBOperator(corners, u_local, mat, y_naive);

    for (int i = 0; i < 10; ++i) {
        ExpectVector3Near(y_mf[i], y_naive[i], 1e-6);
    }
}

TEST(ElementKernel, UniaxialStretchStress) {
    // Uniform stretch u = (eps * x, 0, 0) => epsilon_xx = eps, others 0.
    // sigma_xx = (lambda + 2*mu) * eps.
    // Check via energy: u.Ku = integral sigma:epsilon = sigma_xx * eps * V.
    Vector3 corners[4], all_nodes[10];
    MakeTestElement(corners, all_nodes);

    const double eps = 0.1;
    const double lambda = 1.e5;
    const double mu = 0.5e5;
    LinearElasticMaterial mat[4];
    UniformMat(mat, lambda, mu);

    Vector3 u_local[10];
    for (int i = 0; i < 10; ++i) {
        u_local[i] = Vector3(eps * all_nodes[i].x(), 0., 0.);
    }

    Vector3 y_local[10];
    ComputeElementLinearElasticityOperator(corners, u_local, mat, y_local);

    const double energy = 0.5 * LocalDot(u_local, y_local);
    const ElementGeometry geo = ComputeElementGeometry(corners);
    const double volume = geo.det_jacobian / 6.;
    const double sigma_xx = (lambda + 2. * mu) * eps;
    const double expected_energy = 0.5 * sigma_xx * eps * volume;
    EXPECT_NEAR(energy, expected_energy, 1e-6 * std::max(1., expected_energy));
}
