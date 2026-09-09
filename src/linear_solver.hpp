#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
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
// Conjugate Gradient
//---------------------------------------------------------------------------

/**
 * @brief Outcome of a ConjugateGradient solve.
 */
struct CGResult {
    std::vector<Vector3> solution;
    int iterations;
    double final_residual_norm;
    bool converged;
};

/**
 * @brief Unpreconditioned Conjugate Gradient for a symmetric positive-
 * definite matrix-free operator A(p) -> y.
 *
 * Standard textbook CG: A is never formed, only applied. Convergence is
 * declared once ||r|| < tolerance * max(||rhs||, 1), a relative-with-a-
 * floor stopping criterion so a near-zero RHS does not demand an
 * unreachable absolute tolerance.
 *
 * No preconditioner is applied here -- see the repo's TODO/Future work
 * notes on Jacobi/multigrid preconditioning for ill-conditioned problems.
 *
 * @tparam ApplyOperatorFn Callable with signature
 *         void(const std::vector<Vector3>& p, std::vector<Vector3>& out).
 *         Taken as a template parameter (monomorphized at compile time)
 *         rather than std::function, so this stays a thin, zero-overhead
 *         wrapper portable to a device-side rewrite later, not a design
 *         that depends on host-only type erasure.
 * @param apply_operator The operator A, e.g. @ref ApplyConstrainedOperator
 *                        for a Dirichlet-constrained elasticity solve.
 * @param rhs             Right-hand side b, one Vector3 per node.
 * @param x0               Initial guess, one Vector3 per node -- for a
 *                          Dirichlet-constrained solve this must already be
 *                          zero at every constrained node (see
 *                          boundary_conditions.hpp).
 * @param tolerance         Relative residual-norm stopping tolerance.
 * @param max_iterations    Iteration cap.
 */
template <typename ApplyOperatorFn>
CGResult ConjugateGradient(
    ApplyOperatorFn apply_operator, const std::vector<Vector3>& rhs,
    std::vector<Vector3> x0, double tolerance, int max_iterations
) {
    const std::size_t n{rhs.size()};
    std::vector<Vector3> x{std::move(x0)};

    std::vector<Vector3> Ax(n);
    apply_operator(x, Ax);

    std::vector<Vector3> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = rhs[i] - Ax[i];
    }

    std::vector<Vector3> p{r};
    double rs_old{GlobalDot(r, r)};

    const double rhs_norm{std::sqrt(GlobalDot(rhs, rhs))};
    const double abs_tol{tolerance * std::max(rhs_norm, 1.0)};

    std::vector<Vector3> Ap(n);
    int iter{0};
    for (; iter < max_iterations; ++iter) {
        if (std::sqrt(rs_old) < abs_tol) {
            break;
        }

        apply_operator(p, Ap);
        const double pAp{GlobalDot(p, Ap)};
        const double alpha{rs_old / pAp};

        for (std::size_t i = 0; i < n; ++i) {
            x[i] += p[i] * alpha;
            r[i] -= Ap[i] * alpha;
        }

        const double rs_new{GlobalDot(r, r)};
        const double beta{rs_new / rs_old};
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = r[i] + p[i] * beta;
        }
        rs_old = rs_new;
    }

    const double final_residual_norm{std::sqrt(rs_old)};
    return CGResult{
        std::move(x), iter, final_residual_norm, final_residual_norm < abs_tol
    };
}

}  // namespace matrix_free_fea
