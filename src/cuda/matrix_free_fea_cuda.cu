/**
 * @file matrix_free_fea_cuda.cu
 * @brief CUDA matrix-free P2 tetrahedral linear-elasticity operator.
 *
 * ---------------------------------------------------------------------------
 * Overview
 * ---------------------------------------------------------------------------
 * Evaluates y = K * u over a quadratic (10-node) tetrahedral mesh without
 * explicitly forming an element or global stiffness matrix. Every application
 * of the operator re-derives each element's contribution (on the fly) from the
 * mesh geometry and material data, so the working set shrinks from an assembled
 * matrix's non-zero elements (O(n^2)) to little more than the solution
 * vectors (O(n)).
 *
 * The cost is recomputing the element integrals on each time the operator
 * is applied, which is the tradeoff a matrix-free formulation makes. On
 * bandwidth-bound hardware, and GPUs in particular, that recomputation is
 * typically cheaper than the memory fetches it replaces.
 *
 * ---------------------------------------------------------------------------
 * Design choices
 * ---------------------------------------------------------------------------
 * Two design choices are worth calling out explicitly, both driven by the
 * target hardware:
 *
 *   - Vector3 and Matrix3 are hand-rolled rather than taken from a
 *     general-purpose linear algebra library such as Eigen. Device support in
 *     such libraries is typically unofficial and has a history of breaking
 *     across nvcc releases.
 *
 *   - The element loop is a CUDA kernel, one thread per element. Each thread
 *     runs its element's 4 quadrature points and 30 local DOFs serially -> at
 *     this size there is little to gain from finer parallelism, since the
 *     speedup comes from thousands of element threads running concurrently,
 *     as opposed to from splitting one element. The scatter-add pattern uses
 *     atomicAdd because elements sharing a face, edge, or vertex write to the
 *     same global node concurrently. Mesh coloring patterns would remove the
 *     atomics at the cost of real preprocessing complexity, and are
 *     deliberately out of scope for this implementation.
 *
 * Precision is double throughout -> a future implementation may support single
 * precision for reduced memory footprint via a template parameter.
 *
 * ---------------------------------------------------------------------------
 * Out of scope
 * ---------------------------------------------------------------------------
 * - No linear solver (this is a single Krylov matrix-vector product, not an
 * actual linear solve)
 * - No mechanism for applying boundary conditions
 * - No mesh I/O (the mesh is assumed already resident on the device in
 * flattened form via a custom mesh format)
 * - No degenerate-element check (det(J) > 0 is assumed)
 */

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

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Represents a single quadrature point and its weight in the reference
 * tetrahedron.
 *
 * Holds the barycentric coordinates (xi, eta, zeta) packed into a Vector3
 * and the associated integration weight for one point in the Hammer-Stroud
 * quadrature rule. The weight already incorporates the reference-tetrahedron
 * volume (1/6), so the integral of a function f over the reference tet is
 * approximated as -> sum_q (weight_q * f(coords_q)) with no further scaling.
 */
struct QuadraturePoint {
    Vector3 coords;  ///< Barycentric coordinates (xi, eta, zeta) of the point
    double weight;   ///< Integration weight (includes reference-tet volume 1/6)
};

/// @brief Number of points in the quadrature rule below
constexpr int kQuadraturePoints{4};

//---------------------------------------------------------------------------
// Quadrature Rule
//---------------------------------------------------------------------------

/**
 * @brief Writes the 4-point Hammer-Stroud quadrature rule for the reference
 *        tetrahedron into @p quadrature_rule.
 *
 * The rule is exact for polynomials up to total degree 2. The P2/quadratic
 * tetrahedron's shape-function gradients are degree-1 polynomials in
 * (xi, eta, zeta) -> integrand epsilon(v):sigma(u) -> a product of two such
 * gradients -> is degree 2 and is integrated exactly.
 *
 * Material properties lambda and mu are evaluated pointwise at each quadrature
 * point rather than interpolated -> they do not raise the polynomial degree.
 *
 * The reference tetrahedron has volume 1/6; the 4 points share equal weight,
 * giving each weight = (1/6) / 4 = 1/24.
 *
 * The rule fills a caller-supplied array from literal constants rather than
 * returning a reference to a function-local static. That keeps the function
 * pure and stateless -> there is no device-side static whose initialisation
 * order and thread-safety across concurrently launched threads would need
 * reasoning about. The constants a and b below are (5 + 3*sqrt(5))/20 and
 * (5 - sqrt(5))/20 respectively.
 *
 * @param quadrature_rule Output: the 4 QuadraturePoint values of the rule
 */
MFFEA_HOST_DEVICE inline void TetrahedronQuadratureRule(
    QuadraturePoint quadrature_rule[kQuadraturePoints]
) {
    constexpr double a{0.5854101966249685};
    constexpr double b{0.1381966011250105};
    constexpr double weight{1. / 24.};

    quadrature_rule[0] = QuadraturePoint{Vector3(a, b, b), weight};
    quadrature_rule[1] = QuadraturePoint{Vector3(b, a, b), weight};
    quadrature_rule[2] = QuadraturePoint{Vector3(b, b, a), weight};
    quadrature_rule[3] = QuadraturePoint{Vector3(b, b, b), weight};
}

//---------------------------------------------------------------------------
// Constants and data types
//---------------------------------------------------------------------------

/// @brief Number of nodes per P2/quadratic tetrahedron element
constexpr int kNodesPerTetElement{10};

/// @brief Number of corner nodes per tetrahedron -> only these four enter the
/// affine geometric map, while all 10 carry the quadratic displacement basis
constexpr int kCornersPerTetElement{4};

/**
 * @brief Holds shape function values and reference-space gradients at a single
 *        quadrature point.
 *
 * Holds the output of @ref EvaluateShapeFunctions for the standard quadratic
 * (P2) 10-node tetrahedron. Node ordering matches common conventions
 * (e.g. Abaqus C3D10).
 *
 *   Corners:    0, 1, 2, 3
 *   Edge nodes: 4 = mid(0,1)   5 = mid(1,2)   6 = mid(0,2)
 *               7 = mid(0,3)   8 = mid(1,3)   9 = mid(2,3)
 */
struct ShapeFunctionData {
    /// N_i(xi, eta, zeta) -> shape function values
    double shape_func_values[kNodesPerTetElement];

    /// dN_i/d(xi, eta, zeta) -> Reference-space shape function derivatives
    Vector3 shape_func_derivatives[kNodesPerTetElement];
};

//---------------------------------------------------------------------------
// Evaluation
//---------------------------------------------------------------------------

/**
 * @brief Evaluates all 10 P2/quadratic shape functions and their reference
 * space gradients at a given point in the reference tetrahedron.
 *
 * The reference tetrahedron has vertices at (0,0,0), (1,0,0), (0,1,0) and
 * (0,0,1) in (xi, eta, zeta).  The four barycentric coordinates are
 *
 *   L0 = 1 - xi - eta - zeta   (corner 0, the origin)
 *   L1 = xi                    (corner 1)
 *   L2 = eta                   (corner 2)
 *   L3 = zeta                  (corner 3)
 *
 * and the two families of shape functions are
 *
 *   Corner nodes (a = 0..3):  N_a = L_a * (2*L_a - 1)
 *   Edge nodes   (p,q pair):  N   = 4 * L_p * L_q
 *
 * The dL and edge_pairs tables are plain locals rather than function-local
 * statics, for the same reason as @ref TetrahedronQuadratureRule. Both are
 * tiny (4 and 6 entries), so rebuilding them per call costs less than a
 * device-side static would.
 *
 * @param xi   First barycentric coordinate (>= 0)
 * @param eta  Second barycentric coordinate (>= 0)
 * @param zeta Third barycentric coordinate (>= 0, xi+eta+zeta <= 1)
 * @return A ShapeFunctionData holding all N_i values and dN_i/d(xi, eta, zeta)
 * vectors
 */
MFFEA_HOST_DEVICE inline ShapeFunctionData EvaluateShapeFunctions(
    double xi, double eta, double zeta
) {
    const double L[4] = {
        1. - xi - eta - zeta,  // L0
        xi,                    // L1
        eta,                   // L2
        zeta                   // L3
    };

    // dL_a/d(xi,eta,zeta), a = 0, 1, 2, 3 -> constant across the element
    // since the geometric map is affine
    const Vector3 dL[4] = {
        Vector3(-1., -1., -1.),  // dL0
        Vector3(1., 0., 0.),     // dL1
        Vector3(0., 1., 0.),     // dL2
        Vector3(0., 0., 1.),     // dL3
    };

    ShapeFunctionData shape_func{};

    // Corner nodes (4): N_a = L_a * (2*L_a - 1), a = 0, 1, 2, 3
    for (int node_idx = 0; node_idx < kCornersPerTetElement; ++node_idx) {
        // shape function values for corner nodes
        shape_func.shape_func_values[node_idx] =
            L[node_idx] * (2. * L[node_idx] - 1.);

        // reference-space gradients for corner nodes -> dL_a/dxi
        shape_func.shape_func_derivatives[node_idx] =
            dL[node_idx] * (4. * L[node_idx] - 1.);
    }

    // Edge nodes (6): N = 4*L_p*L_q -> pairing matches the node-ordering above
    const int edge_pairs[6][2]{
        {0, 1},  // edge 0
        {1, 2},  // edge 1
        {0, 2},  // edge 2
        {0, 3},  // edge 3
        {1, 3},  // edge 4
        {2, 3}   // edge 5
    };

    for (int edge_idx = 0; edge_idx < 6; ++edge_idx) {
        const int edge_node_1{edge_pairs[edge_idx][0]};
        const int edge_node_2{edge_pairs[edge_idx][1]};
        const int node_idx{4 + edge_idx};

        // shape function values for edge nodes -> N = 4 * L_p * L_q
        shape_func.shape_func_values[node_idx] =
            4. * L[edge_node_1] * L[edge_node_2];

        // reference-space gradients for edge nodes ->
        // dN/dxi = 4 * L_q * (dL_node_1/dxi) + 4 * L_p * (dL_node_2/dxi)
        shape_func.shape_func_derivatives[node_idx] =
            dL[edge_node_1] * (4. * L[edge_node_2]) +
            dL[edge_node_2] * (4. * L[edge_node_1]);
    }
    return shape_func;
}

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Pre-computed affine geometry for one tetrahedral element.
 *
 * Encapsulates the geometric Jacobian J = dx/d(xi,eta,zeta) and its inverse
 * for the affine map from the reference tetrahedron to the physical element.
 * Because the map is linear in the corner-node coordinates (independent of
 * the quadratic displacement basis), all three quantities are constant across
 * the element and need only be computed once per element, not once per
 * quadrature point.
 */
struct ElementGeometry {
    Matrix3 jacobian;          ///< Jacobian dx/d(xi, eta, zeta) ->
                               ///< columns = [x1-x0 | x2-x0 | x3-x0]
    Matrix3 inverse_jacobian;  ///< Inverse of Jacobian -> used to pull physical
                               ///< space gradients back to reference space
    double det_jacobian;       ///< det(J) -> positive for a non-inverted,
                               ///< right-handed element
};

//---------------------------------------------------------------------------
// Functions
//---------------------------------------------------------------------------

/**
 * @brief Computes the affine element geometry from the 4 corner-node
 * coordinates.
 *
 * The affine map is x(xi,eta,zeta) = x0 + xi*(x1-x0) + eta*(x2-x0) +
 * zeta*(x3-x0), so J = [x1-x0 | x2-x0 | x3-x0].  J is assembled with
 * @ref FromColumns, inverted with the closed-form 3x3 adjugate, and its
 * determinant is stored for use in the quadrature weight scaling. A
 * non-degenerate, non-inverted element (det(J) > 0) is assumed and not
 * checked.
 *
 * @param corner_nodes Physical (x,y,z) coordinates of the 4 corner nodes, in
 * the same ordering as @ref EvaluateShapeFunctions (corner node 0 maps to the
 * reference tet's origin)
 * @return A fully populated ElementGeometry with J, inverse_jacobian, and
 *         det_jacobian
 */
MFFEA_HOST_DEVICE inline ElementGeometry ComputeElementGeometry(
    const Vector3 corner_nodes[kCornersPerTetElement]
) {
    ElementGeometry element_geometry{};
    element_geometry.jacobian = FromColumns(
        corner_nodes[1] - corner_nodes[0],  // x1 - x0
        corner_nodes[2] - corner_nodes[0],  // x2 - x0
        corner_nodes[3] - corner_nodes[0]   // x3 - x0
    );
    element_geometry.det_jacobian = element_geometry.jacobian.Determinant();
    element_geometry.inverse_jacobian = element_geometry.jacobian.Inverse();
    return element_geometry;
}

/**
 * @brief Transforms reference-space shape-function gradients to physical space.
 *
 * Applies the pull-back dN/dx = J^{-T} * dN/dxi for all kNodesPerTetElement
 * nodes at once. J^{-T} is constant across the element (affine map) but
 * must still be applied per quadrature point because dN_dxi varies with the
 * evaluation point (quadratic P2/basis).
 *
 * @param J_inv_transpose      Transpose of J^{-1} -> constant for element
 * @param dN_dxi     Reference-space gradients dN_i/d(xi,eta,zeta) -> one per
 *                   node (P2/quadratic basis)
 * @param dN_dx      Output: physical-space gradients dN_i/dx -> one per node
 */
MFFEA_HOST_DEVICE inline void PhysicalGradients(
    const Matrix3& J_inv_transpose, const Vector3 dN_dxi[kNodesPerTetElement],
    Vector3 dN_dx[kNodesPerTetElement]
) {
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        dN_dx[node_idx] = J_inv_transpose * dN_dxi[node_idx];
    }
}

//---------------------------------------------------------------------------
// Constants
//---------------------------------------------------------------------------

/// @brief Spatial dimension
constexpr int kDimensions{3};

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Lamé material parameters at a single quadrature point.
 *
 * Lambda and mu are provided pointwise rather than interpolated by a
 * polynomial basis, so they do not raise the polynomial degree of the
 * integrand and the 4-point quadrature rule remains exact.
 */
struct LinearElasticMaterial {
    double lambda;  ///< First Lamé parameter (related to bulk modulus)
    double mu;      ///< Second Lamé parameter (shear modulus)
};

//---------------------------------------------------------------------------
// Compute element operator
//---------------------------------------------------------------------------

/**
 * @brief Computes y_e = K_e * u_e for one element without forming K_e.
 *
 * Implements the matrix-free evaluation of the local stiffness action via
 * the weak-form identity:
 *
 *   (y_e)_{i,d} = integral_Omega  ( sigma(u) * grad(N_i) )_d  dOmega
 *
 * derived from a(v, u) = integral epsilon(v):sigma(u) with v = N_i * e_d.
 * At each quadrature point the algorithm:
 *
 *   1. Evaluates dN/dx = J^{-T} * dN/dxi  (physical-space gradients)
 *   2. Assembles grad(u) = sum_i u_i (x) dN_i/dx  (B * u_e, never explicit)
 *   3. Computes epsilon = sym(grad(u)) and sigma = lambda*tr(eps)*I + 2*mu*eps
 *   4. Accumulates y_i += w_q * det(J) * sigma * grad(N_i) -> (B^T * sigma,
 *      never explicit)
 *
 * The geometric Jacobian J is constant across the element (affine map) and is
 * therefore computed once before the quadrature loop.
 *
 * This is the unit of work one CUDA thread performs, in full, for one element
 * -> see @ref GlobalLinearElasticOperatorKernel  below.
 *
 * @param corner_nodes   Physical coordinates of the 4 corner nodes
 * @param u_local        Nodal displacements, one Vector3 per node in
 *                       EvaluateShapeFunctions ordering
 * @param material_at_QP Lamé parameters at each of the 4 quadrature points,
 *                       in TetrahedronQuadratureRule() order
 * @param y_local        Output: local internal-force vector, one Vector3 per
 *                       node.  Zero-initialised by this function
 */
MFFEA_HOST_DEVICE inline void ComputeElementLinearElasticOperator(
    const Vector3 corner_nodes[kCornersPerTetElement],
    const Vector3 u_local[kNodesPerTetElement],
    const LinearElasticMaterial material_at_QP[kQuadraturePoints],
    Vector3 y_local[kNodesPerTetElement]
) {
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        y_local[node_idx] = Vector3::Zero();
    }

    // Affine map -> J, inverse_jacobian, det_jacobian are constant across
    // the element -> compute once here rather than inside quadrature loop
    const ElementGeometry element_geometry{
        ComputeElementGeometry(corner_nodes)
    };
    const Matrix3 J_inverse_transpose{
        element_geometry.inverse_jacobian.Transpose()
    };

    QuadraturePoint quadrature_rule[kQuadraturePoints];
    TetrahedronQuadratureRule(quadrature_rule);

    for (int qp_idx = 0; qp_idx < kQuadraturePoints; ++qp_idx) {
        const QuadraturePoint& qp{quadrature_rule[qp_idx]};

        // Shape-function gradients DO vary per quadrature point even
        // though J does NOT -> quadratic P2 basis vs. linear geometry
        const ShapeFunctionData shape_func{
            EvaluateShapeFunctions(qp.coords.x, qp.coords.y, qp.coords.z)
        };

        Vector3 dN_dx[kNodesPerTetElement];
        PhysicalGradients(
            J_inverse_transpose, shape_func.shape_func_derivatives, dN_dx
        );

        // Strain: epsilon(u) = 0.5*(grad(u) + grad(u)^T)
        // grad(u)_{jk} = sum_i (u_local[i][j] * dN_dx[i][k]) ->
        //  (= B*u_e, implicit)
        Matrix3 grad_displacement{Matrix3::Zero()};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            const Vector3& u_i{u_local[node_idx]};
            const Vector3& dN_i_dx{dN_dx[node_idx]};
            for (int dim_1 = 0; dim_1 < kDimensions; ++dim_1) {
                for (int dim_2 = 0; dim_2 < kDimensions; ++dim_2) {
                    grad_displacement(dim_1, dim_2) +=
                        u_i[dim_1] * dN_i_dx[dim_2];
                }
            }
        }
        const Matrix3 epsilon{
            (grad_displacement + grad_displacement.Transpose()) * 0.5
        };

        // Stress: sigma = lambda*tr(eps)*I + 2*mu*eps
        const double lambda{material_at_QP[qp_idx].lambda};
        const double mu{material_at_QP[qp_idx].mu};
        const Matrix3 sigma{
            Matrix3::Identity() * (lambda * epsilon.Trace()) +
            epsilon * (2. * mu)
        };

        // Accumulate: y_i += w_q * det(J) * sigma * grad(N_i) ->
        // (= B^T*sigma -> implicit)
        const double scale{qp.weight * element_geometry.det_jacobian};
        for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
            y_local[node_idx] += (sigma * dN_dx[node_idx]) * scale;
        }
    }
}

//---------------------------------------------------------------------------
// Data types
//---------------------------------------------------------------------------

/**
 * @brief Device-resident, flattened view of the mesh.
 *
 * Owning containers such as std::vector do not survive a trip to the device,
 * so the mesh is described here in flattened form: raw pointers into device
 * memory plus the two sizes. Building it -> copying each array across with
 * cudaMemcpy -> is the caller's job and out of this file's scope. This struct
 * only describes the layout the kernel expects to already find in device
 * memory.
 */
struct MeshView {
    /// Physical (x,y,z) coordinate of each global node -> [num_nodes]
    const Vector3* node_coords;

    /// Per-element connectivity, row-major ->
    /// [num_elements * kNodesPerTetElement]
    const int* connectivity;

    /// Lamé parameters per element per quadrature point, row-major ->
    /// [num_elements * kQuadraturePoints]
    const LinearElasticMaterial* material;

    int num_nodes;     ///< Number of global nodes
    int num_elements;  ///< Number of elements
};

//---------------------------------------------------------------------------
// Global matrix-free linear elasticity operator
//---------------------------------------------------------------------------

/**
 * @brief Applies the global stiffness operator y = K * u without ever forming
 * K -> one CUDA thread per element.
 *
 * Each thread handles exactly one element and runs three phases:
 *
 *   - Gather -> copy the element's 4 corner coordinates and 10 nodal
 *     displacements out of the global arrays into thread-local buffers, using
 *     the element's connectivity row as the index map.
 *
 *   - Apply -> call @ref ComputeElementLinearElasticOperator  to evaluate
 *     y_local = K_e * u_local by quadrature -> K_e is likewise never formed.
 *
 *   - Scatter-add -> accumulate the 10 local force vectors back into y_global
 *     at those same connectivity slots.
 *
 * The scatter is the only phase that needs synchronisation. Elements sharing
 * a face, edge, or vertex reach the same global node from different threads,
 * so plain += would race and each component goes through atomicAdd instead.
 * atomicAdd on double requires compute capability 6.0 or newer.
 *
 * @param mesh      Device-resident mesh -> see @ref MeshView
 * @param u_global  Device array, input displacement field -> [num_nodes]
 * @param y_global  Output: device array, global internal-force vector ->
 *                  [num_nodes]. The caller must zero this before launch (e.g.
 *                  cudaMemset), since the kernel accumulates and concurrent
 *                  threads cannot each independently reset a shared array
 */
__global__ void GlobalLinearElasticOperatorKernel(
    MeshView mesh, const Vector3* u_global, Vector3* y_global
) {
    const int elem{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (elem >= mesh.num_elements) {
        return;
    }

    // Maps the element's 10 local slots -> global node indices
    const int* connectivity{mesh.connectivity + elem * kNodesPerTetElement};

    // (1) Gather -> pull element's data out of the global arrays so the
    // element operator can walk it contiguously
    Vector3 corner_nodes[kCornersPerTetElement];
    for (int node_idx = 0; node_idx < kCornersPerTetElement; ++node_idx) {
        corner_nodes[node_idx] = mesh.node_coords[connectivity[node_idx]];
    }

    // Copy element's 10 nodal displacements out of the global array
    // into a local contiguous buffer
    Vector3 u_local[kNodesPerTetElement];
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        u_local[node_idx] = u_global[connectivity[node_idx]];
    }

    // (2) Apply -> {y_local} = [K_e] * {u_local} by quadrature -> [K_e] is
    // never formed
    Vector3 y_local[kNodesPerTetElement];
    ComputeElementLinearElasticOperator(
        corner_nodes, u_local, mesh.material + elem * kQuadraturePoints, y_local
    );

    // (3) Scatter-add -> fold local forces back into the global vector.
    // Shared nodes accumulate one term per adjacent element -> concurrent
    // threads may target the same node, so += becomes atomicAdd
    for (int node_idx = 0; node_idx < kNodesPerTetElement; ++node_idx) {
        const int global_node{connectivity[node_idx]};
        atomicAdd(&(y_global[global_node].x), y_local[node_idx].x);
        atomicAdd(&(y_global[global_node].y), y_local[node_idx].y);
        atomicAdd(&(y_global[global_node].z), y_local[node_idx].z);
    }
}

/**
 * @brief Host-side launch wrapper for @ref GlobalLinearElasticOperatorKernel.
 *
 * The block size is a multiple of the 32-thread warp size, as full-warp
 * utilisation requires, and is kept modest because each thread carries 30
 * local DOFs of live state -> a larger block would only cut occupancy
 * further.
 *
 * @param mesh      Device-resident mesh -> see @ref MeshView
 * @param u_global  Device array, input displacement field -> [num_nodes]
 * @param y_global  Output: device array, global internal-force vector ->
 *                  [num_nodes]. Must already be zeroed by the caller, since
 *                  the kernel accumulates rather than assigns
 */
inline void LaunchGlobalLinearElasticOperatorKernel(
    const MeshView& mesh, const Vector3* u_global, Vector3* y_global
) {
    constexpr int kBlockSize{128};
    const int grid_size{(mesh.num_elements + kBlockSize - 1) / kBlockSize};

    GlobalLinearElasticOperatorKernel<<<grid_size, kBlockSize>>>(
        mesh, u_global, y_global
    );
}

}  // namespace matrix_free_fea

/**
 * ---------------------------------------------------------------------------
 * Validation and verification strategy
 * ---------------------------------------------------------------------------
 * This operator admits a strong set of black-box checks that hold exactly, or
 * to floating-point tolerance, independent of mesh or material. They are listed
 * here for reference; a recommended unit testing structure exercising each one
 * follows below.
 *
 * Properties (exact i.e. machine epsilon, or to floating-point tolerance):
 *
 * 1. Rigid-body translation -> zero nodal force. A constant displacement field
 * gives grad(u) = 0, hence epsilon = 0, sigma = 0, y_i = 0 at every node. A
 * single-element check is sufficient.
 *
 * 2. Rigid-body rotation -> zero nodal force. A FINITE rotation is
 * nonlinear in the coordinates, so plugging it into the small-strain formula
 * leaves an O(theta^2) residual -> not a clean check. Instead let's use the
 * infinitesimal (velocity) form u = omega x (x - x_ref): grad(u) is then
 * exactly skew-symmetric, so epsilon = sym(grad(u)) vanishes exactly, with no
 * residual.
 *
 * 3. Internal force self-equilibrium -> sum_i y_i = 0 for ANY displacement
 * field and ANY (even spatially-varying) material. Follows from partition of
 * unity: sum_i grad(N_i) = grad(sum_i N_i) = grad(1) = 0, so summing y_i =
 * sigma . grad(N_i) over i cancels regardless of sigma.
 *
 * 4. Positive semi-definite spectrum. All eigenvalues of the assembled operator
 * must be >= 0, since elastic strain energy u.(K*u)/2 cannot be negative for
 * any physically valid material.
 *
 * 5. Energy consistency against an INDEPENDENTLY implemented quadrature pass.
 * Compare u.(K*u) (via this operator) against the same energy computed by a
 * second, separately-written integration routine that does not call
 * ComputeElementLinearElasticOperator .
 *
 * 6. Refinement invariance for an exact linear field. The quadratic P2 basis
 * reproduces any linear field exactly, so the true strain energy has no
 * discretization error and must be IDENTICAL across mesh refinements (e.g. 1,
 * 2, 3 cube divisions).
 *
 * 7. Closed-form analytical benchmark: uniaxial stretch. For epsilon =
 * diag(eps0, 0, 0), sigma reduces to a diagonal tensor and the energy density
 * collapses to a single term, giving the closed form U = 0.5*(lambda +
 * 2*mu)*eps0^2*V. Prescribing this field exactly and comparing against the
 * formula is the strongest check available, since nothing on the reference side
 * of the comparison shares any code with the implementation under test.
 *
 * ---------------------------------------------------------------------------
 * Out of scope
 * ---------------------------------------------------------------------------
 * Solving an actual boundary-value problem (e.g. a cantilever beam under an end
 * load) against its closed-form deflection, or against a third-party FE solver.
 * That would be a stronger, more standard benchmark than (7) since it exercises
 * bending rather than uniform stretch, which is where element formulations most
 * commonly fail (shear locking). But this approach requires a linear solver and
 * boundary-condition handling, both explicitly out of scope for this project.
 *
 * ---------------------------------------------------------------------------
 * Unit testing strategy
 * ---------------------------------------------------------------------------
 * A per-concern test file structure, mirroring the structure of the source
 * code.
 *
 * - test_vector3_matrix3: Vector3/Matrix3 arithmetic, Identity/Zero,
 * Transpose/Trace/Determinant/Inverse against hand-computed values.
 *
 * - test_quadrature: weights sum to the reference-tet volume (1/6), points lie
 * inside the reference tetrahedron.
 *
 * - test_shape_functions: partition of unity (sum_i N_i = 1) and its gradient
 * (sum_i grad(N_i) = 0), and Kronecker-delta property (N_i(node_j) = delta_ij).
 *
 * - test_geometry: J, det(J), and J^{-1} against a hand-computed reference
 * tetrahedron (J = identity, det(J) = 1).
 *
 * - test_element_operator: properties (1), (2), (3), and (7) above. All
 * single-element checks, run directly against
 * ComputeElementLinearElasticOperator with no assembly involved.
 *
 * - test_assembly: properties (3), (4), (5), and (6) above, on both a small
 * hand-built mesh and a programmatically generated mesh (N x N x N cube
 * subdivisions) for (6), which needs multiple refinement levels.
 *
 * - test_kernel_launch: confirms the kernel produces the same result at varying
 * grid/block configurations (e.g. block sizes 32, 64, 128, 256) on a fixed
 * mesh, since the physics must be invariant to how work is partitioned across
 * threads.
 */
