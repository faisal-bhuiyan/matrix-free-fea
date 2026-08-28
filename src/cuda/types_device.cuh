#pragma once

//---------------------------------------------------------------------------
// Host/device portability
//---------------------------------------------------------------------------

/**
 * @brief Marks a function callable from both host and device code. Expands to
 *        nothing under a non-CUDA compiler so this file's math functions still
 *        build and can be exercised on the host.
 */
#if defined(__CUDACC__)
#define MFFEA_HOST_DEVICE __host__ __device__
#else
#define MFFEA_HOST_DEVICE
#endif

namespace matrix_free_fea {

//---------------------------------------------------------------------------
// Vector and matrix types
//---------------------------------------------------------------------------

/**
 * @brief A 3-component vector of doubles, usable on host and device.
 *
 * The interface is kept deliberately narrow -> only the operations the
 * element operator actually needs. Keeping it to plain scalar members with no
 * dynamic storage means the whole type lives in registers inside the kernel.
 */
struct Vector3 {
    double x{0.};  ///< First component
    double y{0.};  ///< Second component
    double z{0.};  ///< Third component

    Vector3() = default;

    MFFEA_HOST_DEVICE Vector3(double x_in, double y_in, double z_in)
        : x(x_in), y(y_in), z(z_in) {}

    /// @brief Returns the zero vector
    MFFEA_HOST_DEVICE static Vector3 Zero() { return Vector3(0., 0., 0.); }

    MFFEA_HOST_DEVICE Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    MFFEA_HOST_DEVICE Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    MFFEA_HOST_DEVICE Vector3 operator*(double scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    MFFEA_HOST_DEVICE Vector3& operator+=(const Vector3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /// @brief Component access by index -> 0 = x, 1 = y, 2 = z
    MFFEA_HOST_DEVICE double operator[](int index) const {
        return index == 0 ? x : (index == 1 ? y : z);
    }
};

/**
 * @brief A 3x3 matrix of doubles, usable on host and device.
 *
 * Entries are stored in a fixed row-major array and accessed through
 * operator(), so call sites read in conventional (row, col) form, e.g.
 * grad_displacement(dim_1, dim_2) += ...
 */
struct Matrix3 {
    /// Row-major entries -> m[row][col]
    double m[3][3]{
        {0., 0., 0.},  // row 0
        {0., 0., 0.},  // row 1
        {0., 0., 0.},  // row 2
    };

    /// @brief Returns the 3x3 zero matrix
    MFFEA_HOST_DEVICE static Matrix3 Zero() { return Matrix3(); }

    /// @brief Returns the 3x3 identity matrix
    MFFEA_HOST_DEVICE static Matrix3 Identity() {
        Matrix3 identity;
        identity.m[0][0] = 1.;
        identity.m[1][1] = 1.;
        identity.m[2][2] = 1.;
        return identity;
    }

    MFFEA_HOST_DEVICE double operator()(int row, int col) const {
        return m[row][col];
    }
    MFFEA_HOST_DEVICE double& operator()(int row, int col) {
        return m[row][col];
    }

    MFFEA_HOST_DEVICE Vector3 operator*(const Vector3& v) const {
        return Vector3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }

    MFFEA_HOST_DEVICE Matrix3 operator+(const Matrix3& other) const {
        Matrix3 result;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                result.m[row][col] = m[row][col] + other.m[row][col];
            }
        }
        return result;
    }

    MFFEA_HOST_DEVICE Matrix3 operator*(double scalar) const {
        Matrix3 result;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                result.m[row][col] = m[row][col] * scalar;
            }
        }
        return result;
    }

    /// @brief Returns the transpose of this matrix
    MFFEA_HOST_DEVICE Matrix3 Transpose() const {
        Matrix3 result;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                result.m[row][col] = m[col][row];
            }
        }
        return result;
    }

    /// @brief Returns the sum of the diagonal entries
    MFFEA_HOST_DEVICE double Trace() const {
        return m[0][0] + m[1][1] + m[2][2];
    }

    /// @brief Returns the determinant via cofactor expansion along row 0
    MFFEA_HOST_DEVICE double Determinant() const {
        const double minor_00{m[1][1] * m[2][2] - m[1][2] * m[2][1]};
        const double minor_01{m[1][0] * m[2][2] - m[1][2] * m[2][0]};
        const double minor_02{m[1][0] * m[2][1] - m[1][1] * m[2][0]};
        return m[0][0] * minor_00 - m[0][1] * minor_01 + m[0][2] * minor_02;
    }

    /**
     * @brief Returns the inverse via the adjugate (cofactor) formula.
     *
     * Closed-form rather than a factorisation because this is a 3x3 computed
     * once per element, not in a hot inner loop. A singular matrix is not
     * checked for -> det(J) > 0 is assumed for a non-degenerate element.
     */
    MFFEA_HOST_DEVICE Matrix3 Inverse() const {
        const double inv_det{1. / Determinant()};
        Matrix3 result;
        result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        result.m[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
        result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        result.m[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
        result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        result.m[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
        result.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        result.m[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
        result.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        return result;
    }
};

//---------------------------------------------------------------------------
// Free-function helpers
//---------------------------------------------------------------------------

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
MFFEA_HOST_DEVICE inline Matrix3 FromColumns(
    const Vector3& c0, const Vector3& c1, const Vector3& c2
) {
    Matrix3 matrix{Matrix3::Zero()};
    for (int row = 0; row < 3; ++row) {
        matrix(row, 0) = c0[row];  // column 0 -> c0
        matrix(row, 1) = c1[row];  // column 1 -> c1
        matrix(row, 2) = c2[row];  // column 2 -> c2
    }
    return matrix;
}

}  // namespace matrix_free_fea
