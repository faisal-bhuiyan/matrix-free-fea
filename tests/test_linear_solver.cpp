/**
 * @file test_linear_solver.cpp
 * @brief Unit tests for the (P)CG solver and the Jacobi preconditioner,
 * exercising them directly rather than only through the end-to-end pipeline.
 *
 * Covers: symmetry / positive-definiteness of the Dirichlet-reduced
 * operator, the "PCG with M = I is exactly CG" equivalence, Jacobi PCG on a
 * heterogeneous-material problem, the SolveDirichletSystem driver against an
 * independent equilibrium residual, the iteration-cap non-convergence path,
 * and the p^T A p breakdown guard.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "boundary_conditions.hpp"
#include "linear_solver.hpp"
#include "mesh.hpp"
#include "test_helpers.hpp"
#include "test_meshes.hpp"
#include "types.hpp"

namespace matrix_free_fea {
namespace test {
namespace {

constexpr double kLambda{1.0e5};
constexpr double kMu{0.5e5};
constexpr double kFaceTol{1e-9};

/**
 * @brief BC fixing the x=0 and x=1 faces of a cube mesh to zero.
 *
 *     x=0 face                     x=1 face
 *   (u = 0, pinned)              (u = 0, pinned)
 *      +===========================+
 *      |      free interior +      |   pinning two opposite faces removes all
 *      |      side-face nodes      |   6 rigid-body modes -> K_FF is SPD
 *      +===========================+
 */
DirichletBC FixXFacesToZero(const Mesh& mesh) {
    DirichletBC bc;
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        const double x{mesh.node_coords[static_cast<std::size_t>(n)].x()};
        if (std::abs(x) < kFaceTol || std::abs(x - 1.0) < kFaceTol) {
            bc.constrained_nodes.push_back(n);
            bc.prescribed_values.push_back(Vector3::Zero());
        }
    }
    return bc;
}

std::vector<Vector3> ZeroField(const Mesh& mesh) {
    return std::vector<Vector3>(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );
}

std::vector<Vector3> ConstantLoad(const Mesh& mesh, const Vector3& f) {
    return std::vector<Vector3>(static_cast<std::size_t>(mesh.NumNodes()), f);
}

}  // namespace

//---------------------------------------------------------------------------
// Reduced operator is SPD -- the precondition CG's validity rests on
//---------------------------------------------------------------------------
//
//   symmetry:  <u, A v> == <v, A u>     (K_FF principal submatrix of sym K)
//   pos.def.:  <u, A u> >  0  for u != 0 on the free set

TEST(LinearSolver, ConstrainedOperatorIsSymmetricPositiveDefinite) {
    const Mesh mesh{MakeCubeMesh(2)};
    const auto material{UniformMaterial(mesh.NumElements(), kLambda, kMu)};
    const DirichletBC bc{FixXFacesToZero(mesh)};
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    std::vector<Vector3> u{MakeRandomField(mesh, /*seed=*/11)};
    std::vector<Vector3> v{MakeRandomField(mesh, /*seed=*/29)};
    ApplyZeroMask(mask, u);
    ApplyZeroMask(mask, v);

    std::vector<Vector3> au;
    std::vector<Vector3> av;
    apply_op(u, au);
    apply_op(v, av);

    const double u_av{GlobalDot(u, av)};
    const double v_au{GlobalDot(v, au)};
    const double scale{std::max({std::abs(u_av), std::abs(v_au), 1.0})};
    EXPECT_NEAR(u_av, v_au, 1e-9 * scale) << "operator is not symmetric";

    EXPECT_GT(GlobalDot(u, au), 0.0) << "operator is not positive definite";
}

//---------------------------------------------------------------------------
// PCG with the identity preconditioner is bit-for-bit plain CG
//---------------------------------------------------------------------------

TEST(LinearSolver, IdentityPreconditionedCGMatchesPlainCG) {
    const Mesh mesh{MakeCubeMesh(2)};
    const auto material{UniformMaterial(mesh.NumElements(), kLambda, kMu)};
    const DirichletBC bc{FixXFacesToZero(mesh)};
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    const std::vector<Vector3> load{ConstantLoad(mesh, Vector3(1.0e3, 0., 0.))};
    const std::vector<Vector3> rhs{
        ComputeDirichletRHS(mesh, material, mask, load)
    };
    const std::vector<Vector3> x0{ZeroField(mesh)};

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    const CGResult plain{ConjugateGradient(apply_op, rhs, x0, 1e-10, 500)};
    const CGResult identity_pcg{PreconditionedConjugateGradient(
        apply_op, IdentityPreconditioner{}, rhs, x0, 1e-10, 500
    )};

    EXPECT_EQ(plain.iterations, identity_pcg.iterations);
    ASSERT_EQ(
        plain.residual_history.size(), identity_pcg.residual_history.size()
    );
    for (std::size_t k = 0; k < plain.residual_history.size(); ++k) {
        EXPECT_DOUBLE_EQ(
            plain.residual_history[k], identity_pcg.residual_history[k]
        );
    }
    for (std::size_t i = 0; i < plain.solution.size(); ++i) {
        ExpectVector3Near(plain.solution[i], identity_pcg.solution[i], 1e-14);
    }
}

//---------------------------------------------------------------------------
// Jacobi PCG: same answer as CG, in no more iterations
//---------------------------------------------------------------------------
//
//   heterogeneous material -> per-DOF stiffness magnitudes vary across the
//   mesh -> diag(K_FF) scaling clusters the spectrum -> PCG <= CG iters.

TEST(LinearSolver, JacobiPCGConvergesNoSlowerThanPlainCG) {
    const Mesh mesh{MakeCubeMesh(2)};
    const auto material{VaryingMaterial(mesh.NumElements())};
    const DirichletBC bc{FixXFacesToZero(mesh)};
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    const std::vector<Vector3> load{ConstantLoad(mesh, Vector3(0., 1.0e3, 0.))};
    const std::vector<Vector3> rhs{
        ComputeDirichletRHS(mesh, material, mask, load)
    };
    const std::vector<Vector3> x0{ZeroField(mesh)};
    const std::vector<Vector3> jacobi_diagonal{
        BuildJacobiDiagonal(mesh, material, mask)
    };

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    const CGResult cg{ConjugateGradient(apply_op, rhs, x0, 1e-12, 2000)};
    const CGResult pcg{PreconditionedConjugateGradient(
        apply_op, JacobiPreconditioner{jacobi_diagonal}, rhs, x0, 1e-12, 2000
    )};

    PrintCGConvergence("plain CG (varying material)", cg);
    PrintCGConvergence("Jacobi PCG (varying material)", pcg);
    PrintCGComparison(
        "varying material, divisions=2", "CG", cg, "Jacobi PCG", pcg
    );

    ASSERT_TRUE(cg.converged);
    ASSERT_TRUE(pcg.converged);
    EXPECT_LE(pcg.iterations, cg.iterations);

    for (std::size_t i = 0; i < cg.solution.size(); ++i) {
        ExpectVector3Near(cg.solution[i], pcg.solution[i], 1e-8);
    }
}

//---------------------------------------------------------------------------
// SolveDirichletSystem driver vs. an independent equilibrium residual
//---------------------------------------------------------------------------
//
//   +===========================+   body load f = (0,0,2e3) at every node,
//   |            | f             |   x=0 / x=1 faces pinned to 0. After the
//   |            v               |   solve, r = f - K u must vanish at every
//   +===========================+   FREE node (constrained rows are the BC's).

TEST(LinearSolver, SolveDirichletSystemDriverReachesEquilibrium) {
    const Mesh mesh{MakeCubeMesh(2)};
    const auto material{UniformMaterial(mesh.NumElements(), kLambda, kMu)};
    const DirichletBC bc{FixXFacesToZero(mesh)};
    const std::vector<Vector3> load{ConstantLoad(mesh, Vector3(0., 0., 2.0e3))};

    const DirichletSolveResult out{
        SolveDirichletSystem(mesh, material, bc, load, 1e-12, 2000)
    };

    PrintCGConvergence("SolveDirichletSystem driver (body load +z)", out.cg);
    ASSERT_TRUE(out.cg.converged);

    std::vector<Vector3> ku;
    ApplyGlobalLinearElasticityOperator(mesh, out.displacement, material, ku);

    const DirichletMask mask{BuildDirichletMask(mesh, bc)};
    double free_residual_sq{0.};
    for (std::size_t i = 0; i < ku.size(); ++i) {
        if (!mask.is_constrained[i]) {
            free_residual_sq += (load[i] - ku[i]).squaredNorm();
        }
    }
    EXPECT_LT(std::sqrt(free_residual_sq), 1e-5)
        << "solved field is not in equilibrium at the free nodes";
}

//---------------------------------------------------------------------------
// Iteration cap -> reported as non-converged, state still sane
//---------------------------------------------------------------------------

TEST(LinearSolver, IterationCapReportsNonConvergence) {
    const Mesh mesh{MakeCubeMesh(2)};
    const auto material{UniformMaterial(mesh.NumElements(), kLambda, kMu)};
    const DirichletBC bc{FixXFacesToZero(mesh)};
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    const std::vector<Vector3> load{ConstantLoad(mesh, Vector3(1.0e3, 0., 0.))};
    const std::vector<Vector3> rhs{
        ComputeDirichletRHS(mesh, material, mask, load)
    };
    const std::vector<Vector3> x0{ZeroField(mesh)};

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    const CGResult r{ConjugateGradient(
        apply_op, rhs, x0, /*tolerance=*/1e-12,
        /*max_iterations=*/1
    )};

    EXPECT_FALSE(r.converged);
    EXPECT_EQ(r.iterations, 1);
    EXPECT_EQ(r.residual_history.size(), 2u);
    EXPECT_TRUE(std::isfinite(r.final_residual_norm));
}

//---------------------------------------------------------------------------
// Breakdown guard: p^T A p <= 0 stops cleanly, no NaNs
//---------------------------------------------------------------------------
//
// Synthetic operator A(p) = p with the (node 0, x) component forced to 0 --
// a 1-D null space, the exact shape of a Dirichlet operator whose BCs left
// one rigid-body mode unpinned. With rhs in that null space the first
// search direction p satisfies A p = 0, so p^T A p = 0 and CG must bail via
// the guard rather than divide by zero.

TEST(LinearSolver, BreakdownGuardStopsOnSingularSearchDirection) {
    const std::size_t n{5};
    std::vector<Vector3> rhs(n, Vector3::Zero());
    rhs[0] = Vector3(1.0, 0., 0.);
    const std::vector<Vector3> x0(n, Vector3::Zero());

    const auto singular_op = [](const std::vector<Vector3>& p,
                                std::vector<Vector3>& y) {
        y = p;
        y[0].x() = 0.;
    };

    const CGResult r{ConjugateGradient(singular_op, rhs, x0, 1e-12, 50)};

    EXPECT_FALSE(r.converged);
    EXPECT_EQ(r.iterations, 0);
    for (const Vector3& xi : r.solution) {
        EXPECT_TRUE(xi.allFinite());
    }
}

}  // namespace test
}  // namespace matrix_free_fea
