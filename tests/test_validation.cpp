// test_validation.cpp
// Black-box validation of ApplyGlobalOperator on MakeCubeMesh.
//
// True patch test (interior nodes), 6 rigid-body modes, null-space rank,
// PSD spectrum, energy consistency, refinement invariance, and analytical
// uniaxial stretch energy.

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "assembly.hpp"
#include "geometry.hpp"
#include "quadrature.hpp"
#include "shape_functions.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

using namespace matrix_free_fea;
using namespace matrix_free_fea::test;

namespace {

bool IsStrictlyInterior(const Vector3& x, double tol = 1e-12) {
    return x.x() > tol && x.x() < 1.0 - tol && x.y() > tol &&
           x.y() < 1.0 - tol && x.z() > tol && x.z() < 1.0 - tol;
}

double GlobalDot(const std::vector<Vector3>& a, const std::vector<Vector3>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += Dot(a[i], b[i]);
    }
    return s;
}

// Independent quadrature of integral(0.5 * sigma : epsilon) over the mesh.
double IndependentStrainEnergy(
    const Mesh& mesh, const std::vector<Vector3>& u,
    const std::vector<std::array<LinearElasticMaterial, 4>>& material
) {
    double energy = 0.0;
    const auto& rule = TetQuadratureRule();

    for (int e = 0; e < mesh.NumElements(); ++e) {
        Vector3 corners[4];
        mesh.ElementCorners(e, corners);
        const ElementGeometry geo = ComputeElementGeometry(corners);
        const Matrix3 J_inv_T = geo.inverse_jacobian.transpose();

        Vector3 u_local[kNodesPerTetElement];
        for (int i = 0; i < kNodesPerTetElement; ++i) {
            u_local[i] = u[mesh.elements[e][i]];
        }

        for (int q = 0; q < 4; ++q) {
            const auto& qp = rule[static_cast<std::size_t>(q)];
            const ShapeFunctionData sf = EvaluateShapeFunctions(
                qp.coords.x(), qp.coords.y(), qp.coords.z()
            );
            Vector3 dN_dx[kNodesPerTetElement];
            PhysicalGradients(J_inv_T, sf.shape_func_derivatives, dN_dx);

            Matrix3 grad_u = Matrix3::Zero();
            for (int i = 0; i < kNodesPerTetElement; ++i) {
                grad_u += u_local[i] * dN_dx[i].transpose();
            }
            const Matrix3 epsilon = 0.5 * (grad_u + grad_u.transpose());
            const double lambda = material[e][q].lambda;
            const double mu = material[e][q].mu;
            const Matrix3 sigma =
                Matrix3::Identity() * (lambda * epsilon.trace()) +
                epsilon * (2.0 * mu);

            energy +=
                0.5 * DoubleDot(sigma, epsilon) * qp.weight * geo.det_jacobian;
        }
    }
    return energy;
}

Matrix3 DefaultStrainMatrix() {
    Matrix3 A;
    A << 0.02, 0.01, -0.005, 0.01, -0.03, 0.004, -0.005, 0.004, 0.015;
    return A;
}

}  // namespace

TEST(Validation, TruePatchTestInteriorNodes) {
    Mesh mesh = MakeCubeMesh(3);
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const Matrix3 A = DefaultStrainMatrix();
    const Vector3 b(0.001, -0.002, 0.0005);
    const auto u = MakeLinearField(mesh, A, b);

    std::vector<Vector3> y;
    ApplyGlobalOperator(mesh, u, material, y);

    int interior_count = 0;
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        if (!IsStrictlyInterior(mesh.node_coords[n])) {
            continue;
        }
        ++interior_count;
        ExpectVector3Near(y[n], Vector3::Zero(), 1e-5);
    }
    EXPECT_GT(interior_count, 0)
        << "cube mesh with divisions=3 must have interior nodes";
}

TEST(Validation, RigidBodyNullSpace) {
    Mesh mesh = MakeCubeMesh(2);
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const Vector3 translations[3] = {
        Vector3(1., 0., 0.),
        Vector3(0., 1., 0.),
        Vector3(0., 0., 1.),
    };
    for (const auto& t : translations) {
        const auto u = MakeRigidTranslation(mesh, t);
        std::vector<Vector3> y;
        ApplyGlobalOperator(mesh, u, material, y);
        for (int n = 0; n < mesh.NumNodes(); ++n) {
            ExpectVector3Near(y[n], Vector3::Zero(), 1e-7);
        }
    }

    const Vector3 rotations[3] = {
        Vector3(1., 0., 0.),
        Vector3(0., 1., 0.),
        Vector3(0., 0., 1.),
    };
    for (const auto& omega : rotations) {
        const auto u = MakeRigidRotation(mesh, omega);
        std::vector<Vector3> y;
        ApplyGlobalOperator(mesh, u, material, y);
        for (int n = 0; n < mesh.NumNodes(); ++n) {
            ExpectVector3Near(y[n], Vector3::Zero(), 1e-6);
        }
    }
}

TEST(Validation, NullSpaceRankExactlySix) {
    Mesh mesh = MakeCubeMesh(1);
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const Eigen::MatrixXd K = AssembleGlobalMatrix(mesh, material);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    ASSERT_EQ(solver.info(), Eigen::Success);

    const auto& evals = solver.eigenvalues();
    const double tol = 1e-6 * evals.cwiseAbs().maxCoeff();
    int null_count = 0;
    for (int i = 0; i < evals.size(); ++i) {
        if (std::fabs(evals[i]) <= tol) {
            ++null_count;
        } else {
            EXPECT_GT(evals[i], -tol)
                << "eigenvalue " << i << " = " << evals[i];
        }
    }
    EXPECT_EQ(null_count, 6);
}

TEST(Validation, PositiveSemiDefiniteSpectrum) {
    Mesh mesh = MakeCubeMesh(1);
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const Eigen::MatrixXd K = AssembleGlobalMatrix(mesh, material);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    ASSERT_EQ(solver.info(), Eigen::Success);

    const auto& evals = solver.eigenvalues();
    const double tol = 1e-8 * evals.cwiseAbs().maxCoeff();
    for (int i = 0; i < evals.size(); ++i) {
        EXPECT_GE(evals[i], -tol) << "eigenvalue " << i;
    }
}

TEST(Validation, EnergyConsistency) {
    Mesh mesh = MakeCubeMesh(2);
    const auto material = UniformMaterial(mesh.NumElements(), 1.0e5, 0.5e5);

    const Matrix3 A = DefaultStrainMatrix();
    const auto u = MakeLinearField(mesh, A, Vector3::Zero());

    std::vector<Vector3> Ku;
    ApplyGlobalOperator(mesh, u, material, Ku);

    const double uKu = GlobalDot(u, Ku);
    const double strain_energy = IndependentStrainEnergy(mesh, u, material);

    // u.Ku = 2 * strain_energy for linear elasticity.
    EXPECT_NEAR(uKu, 2.0 * strain_energy, 1e-6 * std::max(1.0, std::fabs(uKu)));
}

TEST(Validation, RefinementInvarianceLinearField) {
    const Matrix3 A = DefaultStrainMatrix();
    const Vector3 b = Vector3::Zero();
    const double lambda = 1.0e5;
    const double mu = 0.5e5;

    double energies[3];
    for (int d = 0; d < 3; ++d) {
        const int divisions = d + 1;
        Mesh mesh = MakeCubeMesh(divisions);
        const auto material = UniformMaterial(mesh.NumElements(), lambda, mu);
        const auto u = MakeLinearField(mesh, A, b);
        std::vector<Vector3> Ku;
        ApplyGlobalOperator(mesh, u, material, Ku);
        energies[d] = 0.5 * GlobalDot(u, Ku);
    }

    EXPECT_NEAR(
        energies[0], energies[1], 1e-6 * std::max(1.0, std::fabs(energies[0]))
    );
    EXPECT_NEAR(
        energies[1], energies[2], 1e-6 * std::max(1.0, std::fabs(energies[1]))
    );
}

TEST(Validation, AnalyticalUniaxialStretchEnergy) {
    // u = (eps * x, 0, 0) on the unit cube => V = 1.
    // Energy = 0.5 * (lambda + 2*mu) * eps^2 * V.
    const double eps = 0.01;
    const double lambda = 1.0e5;
    const double mu = 0.5e5;

    Mesh mesh = MakeCubeMesh(2);
    const auto material = UniformMaterial(mesh.NumElements(), lambda, mu);

    Matrix3 A = Matrix3::Zero();
    A(0, 0) = eps;
    const auto u = MakeLinearField(mesh, A, Vector3::Zero());

    std::vector<Vector3> Ku;
    ApplyGlobalOperator(mesh, u, material, Ku);
    const double energy = 0.5 * GlobalDot(u, Ku);

    const double expected =
        0.5 * (lambda + 2.0 * mu) * eps * eps * 1.0;  // V = 1
    EXPECT_NEAR(energy, expected, 1e-6 * std::max(1.0, expected));
}
