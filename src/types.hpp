#pragma once

#include <Eigen/Dense>

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Type aliases
//---------------------------------------------------------------------------

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;

//---------------------------------------------------------------------------
// Free-function helpers
//---------------------------------------------------------------------------

/**
 * @brief Computes the Euclidean dot product of two 3-vectors.
 *
 * Thin wrapper around Eigen's member `.dot()` so call sites inside this
 * namespace read uniformly as free functions rather than mixing member and
 * non-member notation.
 *
 * @param a First vector
 * @param b Second vector
 * @return The scalar a · b = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
 */
inline double Dot(const Vector3& a, const Vector3& b) {
    return a.dot(b);
}

/**
 * @brief Assembles a 3x3 matrix from three column vectors.
 *
 * Constructs the matrix M = [c0 | c1 | c2], i.e. the j-th column of M is
 * the j-th argument. This is the canonical way to build the geometric
 * Jacobian J = [x1-x0 | x2-x0 | x3-x0] for an affinely-mapped tetrahedron.
 *
 * @param c0 Vector placed in column 0
 * @param c1 Vector placed in column 1
 * @param c2 Vector placed in column 2
 * @return A Matrix3 whose j-th column equals cj
 */
inline Matrix3 FromColumns(
    const Vector3& c0, const Vector3& c1, const Vector3& c2
) {
    Matrix3 M{Matrix3::Zero()};
    M.col(0) = c0;  // column 0 -> c0
    M.col(1) = c1;  // column 1 -> c1
    M.col(2) = c2;  // column 2 -> c2
    return M;
}

/**
 * @brief Symmetric double-dot (Frobenius) contraction A:B = sum_ij (A_ij*B_ij).
 *
 * Computes the full componentwise inner product of two 3x3 matrices. For
 * symmetric tensors (e.g. stress and strain) this equals the elastic energy
 * density factor sigma:epsilon. The result is the same regardless of
 * argument order.
 *
 * @param A First matrix (typically the stress tensor sigma)
 * @param B Second matrix (typically the strain tensor epsilon)
 * @return The scalar contraction A:B
 */
inline double DoubleDot(const Matrix3& A, const Matrix3& B) {
    return A.cwiseProduct(B).sum();
}

}  // namespace matrix_free_fea
