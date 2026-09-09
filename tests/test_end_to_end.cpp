/**
 * @file test_end_to_end.cpp
 * @brief End-to-end pipeline test: mesh -> Dirichlet BCs -> CG solve,
 * validated against a closed-form elasticity solution.
 *
 * Problem: a unit cube, fixed to the EXACT closed-form displacement at its
 * x=0 and x=1 faces, all other nodes free and unloaded. The chosen field is
 * a genuine uniaxial-STRESS state (not uniaxial strain, and not merely a
 * prescribed field the way test_validation.cpp's analytical check works):
 * a bar stretched along x, free to contract laterally via the Poisson
 * effect, with zero traction on the four side faces satisfied
 * automatically because sigma is diagonal with sigma_yy = sigma_zz = 0.
 *
 * u_exact(x, y, z) = (delta*x, -nu*delta*y, -nu*delta*z)
 *
 * with nu = lambda / (2*(lambda + mu)) chosen so sigma_yy = sigma_zz = 0
 * exactly (derivation: epsilon = diag(delta, -nu*delta, -nu*delta) for this
 * linear field; sigma_yy = lambda*delta*(1-2nu) - 2*mu*nu*delta = 0 solves
 * to nu = lambda/(2*(lambda+mu)), the standard Lame-to-Poisson-ratio
 * identity). Because the field is LINEAR, it is exactly representable by
 * the P2 basis (see the corner/edge shape-function properties in
 * test_element_kernel.cpp), so a correct pipeline should reproduce it at
 * every free node to CG's convergence tolerance, regardless of mesh
 * refinement -- this is a much stronger check than matching an aggregate
 * quantity, since every one of the mesh's interior/free DOFs is actually
 * solved for, not prescribed.
 */
#include <cmath>
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
constexpr double kNu{kLambda / (2.0 * (kLambda + kMu))};
constexpr double kDelta{0.01};
constexpr double kFaceTol{1e-9};

Vector3 ExactField(const Vector3& coords) {
    return Vector3(
        kDelta * coords.x(), -kNu * kDelta * coords.y(),
        -kNu * kDelta * coords.z()
    );
}

// clang-format off
// Problem schematic -- uniaxial tension of the unit cube along x:
//
//     z
//     |__ y
//    /
//   x
//
//     x=0 face                       x=1 face
//   (pinned to u_exact)            (pinned to u_exact)
//     +=============================+   --->  axial stretch:  u_x = delta*x
//     |                             |
//     |   free interior + side-face |   <-->  Poisson pull-in: u_y = -nu*delta*y
//     |   nodes: every DOF SOLVED   |                          u_z = -nu*delta*z
//     |                             |
//     +=============================+   side faces y,z = 0,1 are traction-free
//                                       (sigma_yy = sigma_zz = 0)
//
//   u_exact(x,y,z) = ( delta*x, -nu*delta*y, -nu*delta*z ),
//                    nu = lambda / (2*(lambda + mu))
//
// Only the x=0 / x=1 node displacements are prescribed; the field is linear
// so a correct P2 pipeline must reproduce u_exact at every solved node.
// clang-format on

/**
 * @brief Builds the DirichletBC fixing every x=0 and x=1 node to the exact
 * field's value there (NOT to zero -- most such nodes have nonzero y/z
 * displacement from the Poisson effect).
 */
DirichletBC MakeUniaxialTensionBC(const Mesh& mesh) {
    DirichletBC bc;
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        const Vector3& coords{mesh.node_coords[n]};
        const bool on_x0{std::abs(coords.x() - 0.0) < kFaceTol};
        const bool on_x1{std::abs(coords.x() - 1.0) < kFaceTol};
        if (on_x0 || on_x1) {
            bc.constrained_nodes.push_back(n);
            bc.prescribed_values.push_back(ExactField(coords));
        }
    }
    return bc;
}

/**
 * @brief Runs the full pipeline on a cube mesh with the given refinement
 * and checks the solved field against ExactField() at every node.
 */
void RunUniaxialTensionCase(int divisions) {
    const Mesh mesh{MakeCubeMesh(divisions)};
    const auto material{
        UniformMaterial(mesh.NumElements(), kLambda, kMu)
    };
    const DirichletBC bc{MakeUniaxialTensionBC(mesh)};
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    ASSERT_GT(bc.constrained_nodes.size(), 0u)
        << "no x=0/x=1 nodes found -- mesh face detection is broken";
    ASSERT_LT(bc.constrained_nodes.size(), mask.is_constrained.size())
        << "every node got constrained -- nothing left to solve for";

    const std::vector<Vector3> applied_load(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );
    const std::vector<Vector3> rhs{
        ComputeDirichletRHS(mesh, material, mask, applied_load)
    };

    // x0 = 0 everywhere -> already zero at every constrained node, the
    // invariant ApplyConstrainedOperator's derivation depends on.
    const std::vector<Vector3> x0(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    const CGResult result{ConjugateGradient(
        apply_op, rhs, x0, /*tolerance=*/1e-12,
        /*max_iterations=*/2000
    )};

    PrintCGConvergence(
        ("uniaxial tension, divisions=" + std::to_string(divisions)).c_str(),
        result
    );

    ASSERT_TRUE(result.converged)
        << "CG did not converge in " << result.iterations
        << " iterations, final residual " << result.final_residual_norm;

    const std::vector<Vector3> u{
        ApplyPrescribedValues(mask, result.solution)
    };

    for (int n = 0; n < mesh.NumNodes(); ++n) {
        ExpectVector3Near(
            u[static_cast<std::size_t>(n)], ExactField(mesh.node_coords[n]),
            1e-8, ("node " + std::to_string(n)).c_str()
        );
    }
}

}  // namespace

//---------------------------------------------------------------------------
// Trivial edge case: every node constrained, nothing to solve for
//---------------------------------------------------------------------------
//
//   +-----------+     every node pinned  ->  free set is empty
//   | x x x x x |     ->  initial residual is already 0
//   | x x x x x |     ->  CG returns at iteration 0, converged
//   +-----------+
//   (x = constrained node)

TEST(EndToEnd, FullyConstrainedProblemConvergesTrivially) {
    const Mesh mesh{MakeReferenceTetMesh()};
    const auto material{UniformMaterial(mesh.NumElements(), kLambda, kMu)};

    DirichletBC bc;
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        bc.constrained_nodes.push_back(n);
        bc.prescribed_values.push_back(Vector3(0.01, 0.02, -0.01));
    }
    const DirichletMask mask{BuildDirichletMask(mesh, bc)};

    const std::vector<Vector3> zero(
        static_cast<std::size_t>(mesh.NumNodes()), Vector3::Zero()
    );
    const std::vector<Vector3> rhs{
        ComputeDirichletRHS(mesh, material, mask, zero)
    };

    const auto apply_op = [&](const std::vector<Vector3>& p,
                              std::vector<Vector3>& y) {
        ApplyConstrainedOperator(mesh, p, material, mask, y);
    };

    const CGResult result{ConjugateGradient(apply_op, rhs, zero, 1e-12, 100)};

    PrintCGConvergence("fully constrained (empty free set)", result);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual_history.size(), 1u);
}

//---------------------------------------------------------------------------
// Uniaxial tension: the real end-to-end check
//---------------------------------------------------------------------------

TEST(EndToEnd, UniaxialTensionMatchesClosedForm_Coarse) {
    RunUniaxialTensionCase(/*divisions=*/1);
}

TEST(EndToEnd, UniaxialTensionMatchesClosedForm_Finer) {
    RunUniaxialTensionCase(/*divisions=*/2);
}

}  // namespace test
}  // namespace matrix_free_fea
