#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "types.hpp"

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Vector helpers over the global DOF space
//---------------------------------------------------------------------------

/**
 * @brief Global inner product sum_i a[i] . b[i] over all nodes.
 *
 * Same computational shape as the element loop's per-node accumulation --
 * a flat reduction over Vector3 entries -- just at global rather than
 * per-element scope.
 */
inline double GlobalDot(
    const std::vector<Vector3>& a, const std::vector<Vector3>& b
) {
    double sum{0.};
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += Dot(a[i], b[i]);
    }
    return sum;
}

//---------------------------------------------------------------------------
// Preconditioners
//---------------------------------------------------------------------------

/**
 * @brief The no-op preconditioner M = I, so z = M^{-1} r = r.
 *
 * Passing this to @ref PreconditionedConjugateGradient recovers textbook
 * unpreconditioned CG exactly -- same iterates, same arithmetic (the copy
 * is the only overhead). @ref ConjugateGradient below is the convenience
 * wrapper that does this.
 */
struct IdentityPreconditioner {
    void operator()(
        const std::vector<Vector3>& r, std::vector<Vector3>& z
    ) const {
        z = r;
    }
};

//---------------------------------------------------------------------------
// Conjugate Gradient
//---------------------------------------------------------------------------

/**
 * @brief Outcome of a (Preconditioned) ConjugateGradient solve.
 *
 * @ref residual_history holds the true Euclidean residual norm ||b - A x_k||
 * at every step: entry 0 is the initial residual ||b - A x0||, entry k is
 * the residual after k iterations, and residual_history.back() equals
 * @ref final_residual_norm. So residual_history.size() == iterations + 1
 * always. It is recorded from the residual *recurrence* (r -= alpha*Ap), not
 * a fresh b - A x_k each step, matching what the stopping test sees.
 */
struct CGResult {
    std::vector<Vector3> solution;
    int iterations{0};
    double final_residual_norm{0.};
    bool converged{false};
    std::vector<double> residual_history;
};

/**
 * @brief Preconditioned Conjugate Gradient for a symmetric positive-definite
 * matrix-free operator A(p) -> y with preconditioner M, applied as
 * z = M^{-1} r.
 *
 * Standard textbook PCG: neither A nor M is formed, only applied. The two
 * scalars the unpreconditioned formulation happened to share are kept
 * distinct here:
 *
 *   - <r, z>          drives alpha and beta (the M-inner product of the
 *                      residual; NOT a norm once M != I)
 *   - sqrt(<r, r>)    is the plain residual norm, and the ONLY quantity the
 *                      stopping test looks at
 *
 * Convergence is declared once ||r|| < tolerance * max(||rhs||, 1), a
 * relative-with-a-floor criterion so a near-zero RHS does not demand an
 * unreachable absolute tolerance.
 *
 * If the operator turns out not to be SPD on the space CG explores (for a
 * Dirichlet solve: BCs that fail to pin every rigid-body mode), p^T A p can
 * be zero or negative; alpha would then divide by ~0 and produce NaNs.
 * Instead this stops at that iteration and returns @ref CGResult::converged
 * false with the last good iterate -- see the breakdown guard below.
 *
 * @tparam ApplyOperatorFn      Callable void(const std::vector<Vector3>& p,
 *                              std::vector<Vector3>& out). Taken as a
 *                              template parameter (monomorphised at compile
 *                              time) rather than std::function, so this
 *                              stays a thin, zero-overhead wrapper portable
 *                              to a device-side rewrite later.
 * @tparam ApplyPreconditionerFn Callable void(const std::vector<Vector3>& r,
 *                              std::vector<Vector3>& z) writing z = M^{-1} r.
 *                              @ref IdentityPreconditioner for plain CG.
 * @param apply_operator     The operator A, e.g. @ref ApplyConstrainedOperator
 *                            for a Dirichlet-constrained elasticity solve.
 * @param apply_preconditioner The preconditioner application M^{-1}.
 * @param rhs                Right-hand side b, one Vector3 per node.
 * @param x0                 Initial guess, one Vector3 per node -- for a
 *                            Dirichlet-constrained solve this must already be
 *                            zero at every constrained node (see
 *                            boundary_conditions.hpp).
 * @param tolerance          Relative residual-norm stopping tolerance.
 * @param max_iterations     Iteration cap.
 */
template <typename ApplyOperatorFn, typename ApplyPreconditionerFn>
CGResult PreconditionedConjugateGradient(
    ApplyOperatorFn apply_operator, ApplyPreconditionerFn apply_preconditioner,
    const std::vector<Vector3>& rhs, std::vector<Vector3> x0, double tolerance,
    int max_iterations
) {
    const std::size_t n{rhs.size()};
    std::vector<Vector3> x{std::move(x0)};

    std::vector<Vector3> Ax(n);
    apply_operator(x, Ax);

    std::vector<Vector3> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = rhs[i] - Ax[i];
    }

    // z = M^{-1} r -- the preconditioned residual. For IdentityPreconditioner
    // this is just a copy of r and everything below reduces to plain CG.
    std::vector<Vector3> z(n);
    apply_preconditioner(r, z);

    std::vector<Vector3> p{z};

    double rz_old{GlobalDot(r, z)};  // CG scalar: <r, z>
    double res_sq{GlobalDot(r, r)};  // stopping quantity: ||r||^2

    const double rhs_norm{std::sqrt(GlobalDot(rhs, rhs))};
    const double abs_tol{tolerance * std::max(rhs_norm, 1.0)};

    CGResult result;
    result.residual_history.push_back(std::sqrt(res_sq));

    std::vector<Vector3> Ap(n);
    int iter{0};
    bool broke_down{false};
    for (; iter < max_iterations; ++iter) {
        if (std::sqrt(res_sq) < abs_tol) {
            break;
        }

        apply_operator(p, Ap);
        const double pAp{GlobalDot(p, Ap)};

        // Breakdown guard. For an SPD operator and a nonzero p, pAp > 0
        // always; pAp <= 0 (or NaN) means the restricted operator is not
        // SPD on this search direction, so alpha would blow up. Stop here
        // and report non-convergence rather than spraying NaNs into x.
        if (!(pAp > 0.0)) {
            broke_down = true;
            break;
        }

        const double alpha{rz_old / pAp};
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += p[i] * alpha;
            r[i] -= Ap[i] * alpha;
        }

        res_sq = GlobalDot(r, r);
        result.residual_history.push_back(std::sqrt(res_sq));

        apply_preconditioner(r, z);
        const double rz_new{GlobalDot(r, z)};
        const double beta{rz_new / rz_old};
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = z[i] + p[i] * beta;
        }
        rz_old = rz_new;
    }

    const double final_residual_norm{std::sqrt(res_sq)};
    result.solution = std::move(x);
    result.iterations = iter;
    result.final_residual_norm = final_residual_norm;
    result.converged = !broke_down && final_residual_norm < abs_tol;
    return result;
}

/**
 * @brief Unpreconditioned Conjugate Gradient -- @ref
 * PreconditionedConjugateGradient with @ref IdentityPreconditioner.
 *
 * Kept as a named entry point so call sites that do not (yet) have a
 * preconditioner read plainly, and so the "PCG with M = I is exactly CG"
 * equivalence has something to test against.
 */
template <typename ApplyOperatorFn>
CGResult ConjugateGradient(
    ApplyOperatorFn apply_operator, const std::vector<Vector3>& rhs,
    std::vector<Vector3> x0, double tolerance, int max_iterations
) {
    return PreconditionedConjugateGradient(
        std::move(apply_operator), IdentityPreconditioner{}, rhs, std::move(x0),
        tolerance, max_iterations
    );
}

}  // namespace matrix_free_fea
