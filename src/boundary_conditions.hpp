#pragma once

#include <cstddef>
#include <vector>

#include "assembly.hpp"
#include "mesh.hpp"
#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief A set of Dirichlet (prescribed-displacement) boundary conditions.
 *
 * @ref constrained_nodes and @ref prescribed_values are parallel arrays:
 * constrained_nodes[i]'s displacement is fixed to prescribed_values[i].
 * All other nodes are free (unknowns solved for).
 */
struct DirichletBC {
    std::vector<int> constrained_nodes;
    std::vector<Vector3> prescribed_values;
};

/**
 * @brief Dense per-node lookup form of a DirichletBC, built once per solve.
 *
 * A DirichletBC's sparse (node, value) list is inconvenient to query inside
 * the O(NumNodes()) loops the operator/RHS builders below need, so this
 * expands it into two arrays indexed directly by global node id.
 */
struct DirichletMask {
    std::vector<bool> is_constrained;
    std::vector<Vector3> prescribed;
};

/**
 * @brief Builds a DirichletMask from a DirichletBC for the given mesh.
 */
inline DirichletMask BuildDirichletMask(
    const Mesh& mesh, const DirichletBC& bc
) {
    DirichletMask mask;
    const auto n = static_cast<std::size_t>(mesh.NumNodes());
    mask.is_constrained.assign(n, false);
    mask.prescribed.assign(n, Vector3::Zero());
    for (std::size_t i = 0; i < bc.constrained_nodes.size(); ++i) {
        const auto node = static_cast<std::size_t>(bc.constrained_nodes[i]);
        mask.is_constrained[node] = true;
        mask.prescribed[node] = bc.prescribed_values[i];
    }
    return mask;
}

//---------------------------------------------------------------------------
// Matrix-free Dirichlet elimination
//---------------------------------------------------------------------------
//
// K is symmetric but only positive SEMI-definite (rigid-body modes give it
// a nontrivial null space -- see test_validation.cpp's null-space-rank
// check). Enough Dirichlet constraints pin down every rigid-body mode, and
// the resulting *restricted* system is SPD, which is what CG requires.
//
// The standard technique for enforcing this in a matrix-free setting keeps
// the CG iterate's constrained-node entries at exactly zero throughout the
// whole solve (never at the prescribed value), and only ever asks the full
// operator K to act on vectors that are zero at every constrained node.
// Concretely, for any such vector p:
//
//     (K * p)_F = K_FF * p_F          (free-node output; K's real action)
//     (K * p)_C = K_CF * p_F          (constrained-node output; discarded)
//
// Discarding those rows -- @ref ApplyZeroMask below -- is what turns "apply
// full K, then zero the constrained rows" into exactly K_FF, the free-free
// principal submatrix of a symmetric K. A principal submatrix of a
// symmetric matrix is itself symmetric, so K_FF is SPD and CG is valid.
// (Naively overwriting those rows with p_C's own value instead of zero, or
// running the operator on a vector that carries the prescribed values at
// constrained nodes, both break this symmetry -- see conversation history.)
//
// The prescribed values enter only once, through the right-hand side:
//
//     b_F = f_F - K_FC * u_D
//
// computed by @ref ComputeDirichletRHS as one extra operator application
// before the CG loop starts, not per iteration. After CG returns the
// solution's free part, @ref ApplyPrescribedValues fills in the known
// values at constrained nodes to recover the full displacement field.

/**
 * @brief Zeroes every entry of @p v at constrained nodes, in place.
 */
inline void ApplyZeroMask(
    const DirichletMask& mask, std::vector<Vector3>& v
) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (mask.is_constrained[i]) {
            v[i] = Vector3::Zero();
        }
    }
}

/**
 * @brief The reduced operator K_FF that CG actually sees.
 *
 * @param p Must be zero at every constrained node (the CG loop maintains
 *          this invariant automatically; see the note above).
 * @param y Output: K*p with constrained-node rows discarded (zeroed).
 */
inline void ApplyConstrainedOperator(
    const Mesh& mesh, const std::vector<Vector3>& p,
    const std::vector<std::array<LinearElasticMaterial, 4>>& material,
    const DirichletMask& mask, std::vector<Vector3>& y
) {
    ApplyGlobalLinearElasticityOperator(mesh, p, material, y);
    ApplyZeroMask(mask, y);
}

/**
 * @brief Builds the Dirichlet-lifted right-hand side b_F = f_F - K_FC*u_D.
 *
 * @param applied_load External load f, one Vector3 per global node. Entries
 *                      at constrained nodes are ignored (masked out below).
 */
inline std::vector<Vector3> ComputeDirichletRHS(
    const Mesh& mesh,
    const std::vector<std::array<LinearElasticMaterial, 4>>& material,
    const DirichletMask& mask, const std::vector<Vector3>& applied_load
) {
    const auto n = static_cast<std::size_t>(mesh.NumNodes());

    std::vector<Vector3> u_d_only(n, Vector3::Zero());
    for (std::size_t i = 0; i < n; ++i) {
        if (mask.is_constrained[i]) {
            u_d_only[i] = mask.prescribed[i];
        }
    }

    std::vector<Vector3> k_ud;
    ApplyGlobalLinearElasticityOperator(mesh, u_d_only, material, k_ud);

    std::vector<Vector3> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = applied_load[i] - k_ud[i];
    }
    ApplyZeroMask(mask, rhs);
    return rhs;
}

/**
 * @brief Combines a CG solution's free-node part with the prescribed values
 * to form the full displacement field.
 */
inline std::vector<Vector3> ApplyPrescribedValues(
    const DirichletMask& mask, std::vector<Vector3> u
) {
    for (std::size_t i = 0; i < u.size(); ++i) {
        if (mask.is_constrained[i]) {
            u[i] = mask.prescribed[i];
        }
    }
    return u;
}

}  // namespace matrix_free_fea
