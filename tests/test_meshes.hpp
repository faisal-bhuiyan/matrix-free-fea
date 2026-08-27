#pragma once

#include <map>
#include <utility>
#include <vector>

#include "mesh.hpp"
#include "types.hpp"

namespace matrix_free_fea {
namespace test {

//---------------------------------------------------------------------------
// Edge pairing matching EvaluateShapeFunctions
//---------------------------------------------------------------------------

// Corner-node pairs for local edge slots 4..9.
inline constexpr int kEdgePairs[6][2] = {
    {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
};

/**
 * @brief Append the 6 edge-midpoint nodes for a tet given its 4 corners,
 *        and return a full 10-node ElementConnectivity.
 *
 * Midpoints are looked up (or created) in @p edge_map keyed on the sorted
 * global corner-index pair, so shared edges between adjacent tets collapse
 * to a single global node.
 */
inline ElementConnectivity AppendEdgeNodes(
    Mesh& mesh, const int corners[4],
    std::map<std::pair<int, int>, int>& edge_map
) {
    ElementConnectivity connectivity{};
    for (int i = 0; i < 4; ++i) {
        connectivity[i] = corners[i];
    }
    for (int e = 0; e < 6; ++e) {
        int a = corners[kEdgePairs[e][0]];
        int b = corners[kEdgePairs[e][1]];
        if (a > b) {
            std::swap(a, b);
        }
        const auto key = std::make_pair(a, b);
        auto it = edge_map.find(key);
        if (it == edge_map.end()) {
            const int mid_idx = mesh.NumNodes();
            mesh.node_coords.push_back(
                (mesh.node_coords[a] + mesh.node_coords[b]) * 0.5
            );
            edge_map[key] = mid_idx;
            connectivity[4 + e] = mid_idx;
        } else {
            connectivity[4 + e] = it->second;
        }
    }
    return connectivity;
}

//---------------------------------------------------------------------------
// Single-element factories
//---------------------------------------------------------------------------

/**
 * @brief Single P2 tet on the reference tetrahedron (J == I, det == 1).
 */
inline Mesh MakeReferenceTetMesh() {
    Mesh mesh;
    const Vector3 corners[4] = {
        Vector3(0., 0., 0.),
        Vector3(1., 0., 0.),
        Vector3(0., 1., 0.),
        Vector3(0., 0., 1.),
    };
    for (int i = 0; i < 4; ++i) {
        mesh.node_coords.push_back(corners[i]);
    }
    std::map<std::pair<int, int>, int> edge_map;
    const int corner_idx[4] = {0, 1, 2, 3};
    mesh.elements.push_back(AppendEdgeNodes(mesh, corner_idx, edge_map));
    return mesh;
}

/**
 * @brief Single P2 tet with a deliberately non-symmetric Jacobian.
 *
 * Corner placement ensures J is not symmetric, so a J^{-1} vs J^{-T}
 * transpose mistake is visible in gradient pull-back tests.
 */
inline Mesh MakeSkewedTetMesh() {
    Mesh mesh;
    const Vector3 corners[4] = {
        Vector3(0., 0., 0.),
        Vector3(2., 0., 0.),
        Vector3(0.5, 1.5, 0.),
        Vector3(0.3, 0.2, 1.0),
    };
    for (int i = 0; i < 4; ++i) {
        mesh.node_coords.push_back(corners[i]);
    }
    std::map<std::pair<int, int>, int> edge_map;
    const int corner_idx[4] = {0, 1, 2, 3};
    mesh.elements.push_back(AppendEdgeNodes(mesh, corner_idx, edge_map));
    return mesh;
}

/**
 * @brief Two P2 tets that share no nodes, isolating scatter-add.
 */
inline Mesh MakeDisconnectedPairMesh() {
    Mesh mesh;
    // Element 0: reference-like tet in the positive octant.
    {
        const Vector3 corners[4] = {
            Vector3(0., 0., 0.),
            Vector3(1., 0., 0.),
            Vector3(0., 1., 0.),
            Vector3(0., 0., 1.),
        };
        for (int i = 0; i < 4; ++i) {
            mesh.node_coords.push_back(corners[i]);
        }
        std::map<std::pair<int, int>, int> edge_map;
        const int corner_idx[4] = {0, 1, 2, 3};
        mesh.elements.push_back(AppendEdgeNodes(mesh, corner_idx, edge_map));
    }
    // Element 1: translated copy far away, fresh nodes.
    {
        const int base = mesh.NumNodes();
        const Vector3 offset(10., 10., 10.);
        const Vector3 corners[4] = {
            Vector3(0., 0., 0.) + offset,
            Vector3(1., 0., 0.) + offset,
            Vector3(0., 1., 0.) + offset,
            Vector3(0., 0., 1.) + offset,
        };
        for (int i = 0; i < 4; ++i) {
            mesh.node_coords.push_back(corners[i]);
        }
        std::map<std::pair<int, int>, int> edge_map;
        const int corner_idx[4] = {base, base + 1, base + 2, base + 3};
        mesh.elements.push_back(AppendEdgeNodes(mesh, corner_idx, edge_map));
    }
    return mesh;
}

//---------------------------------------------------------------------------
// Two-element patch (moved from src/mesh.hpp)
//---------------------------------------------------------------------------

/**
 * @brief Build the canonical 2-element patch mesh used by the assembly tests.
 *
 * The mesh is a bipyramid formed by two P2 tetrahedra that share a
 * triangular face ABC:
 *
 *   Element 1: corners (A, B, C, D),  apex D = (0, 0,  1)
 *   Element 2: corners (A, C, B, E),  apex E = (0, 0, -1)
 *
 * Element 2's corner order is deliberately permuted (A, C, B instead of
 * A, B, C) so that both elements have a positive Jacobian determinant even
 * though E sits on the opposite side of the shared face from D.  One
 * consequence: the two elements' local edge-node slots 4..9 do not
 * correspond one-to-one across the shared face — only the global node that
 * each local slot maps to must agree, which is what the connectivity arrays
 * below encode.
 *
 * The mesh has 14 global nodes:
 *   0=A  1=B  2=C  3=D  4=E
 *   5=mid(A,B)  6=mid(B,C)  7=mid(A,C)     (shared-face edges)
 *   8=mid(A,D)  9=mid(B,D)  10=mid(C,D)    (element-1-only edges)
 *   11=mid(A,E) 12=mid(B,E) 13=mid(C,E)    (element-2-only edges)
 *
 * @return A fully populated Mesh with 2 elements and 14 nodes.
 */
inline Mesh MakeTwoElementPatchMesh() {
    Mesh mesh;

    const Vector3 A(0., 0., 0.);
    const Vector3 B(1., 0., 0.);
    const Vector3 C(0., 1., 0.);
    const Vector3 D(0., 0., 1.);
    const Vector3 E(0., 0., -1.);

    mesh.node_coords = {
        A,
        B,
        C,
        D,
        E,
        (A + B) * 0.5,  // 5  = mid(A,B)
        (B + C) * 0.5,  // 6  = mid(B,C)
        (A + C) * 0.5,  // 7  = mid(A,C)
        (A + D) * 0.5,  // 8  = mid(A,D)
        (B + D) * 0.5,  // 9  = mid(B,D)
        (C + D) * 0.5,  // 10 = mid(C,D)
        (A + E) * 0.5,  // 11 = mid(A,E)
        (B + E) * 0.5,  // 12 = mid(B,E)
        (C + E) * 0.5,  // 13 = mid(C,E)
    };

    // Element 1: local corners (A,B,C,D). Local edge slots land on
    // global nodes 5..10 via the standard {0,1}{1,2}{0,2}{0,3}{1,3}{2,3}
    // pairing from EvaluateShapeFunctions.
    mesh.elements.push_back({0, 1, 2, 3, 5, 6, 7, 8, 9, 10});

    // Element 2: local corners (A,C,B,E) -- B and C swapped to preserve
    // a positive Jacobian. Local edge slots therefore map to:
    //   local4=(0,1)=mid(A,C)->7   local5=(1,2)=mid(C,B)->6
    //   local6=(0,2)=mid(A,B)->5   local7=(0,3)=mid(A,E)->11
    //   local8=(1,3)=mid(C,E)->13  local9=(2,3)=mid(B,E)->12
    mesh.elements.push_back({0, 2, 1, 4, 7, 6, 5, 11, 13, 12});

    return mesh;
}

//---------------------------------------------------------------------------
// Structured cube mesh (Kuhn / Freudenthal subdivision)
//---------------------------------------------------------------------------

/**
 * @brief Unit cube split into divisions^3 sub-cubes, each into 6 tets.
 *
 * Uses the standard Kuhn/Freudenthal subdivision (all 6 tets share the
 * main diagonal) so every element has consistent positive orientation.
 * Corner nodes sit on a regular (divisions+1)^3 lattice; edge midpoints
 * are deduplicated via a sorted-pair map.
 *
 * @param divisions Number of subdivisions along each axis (>= 1).
 * @return A fully populated Mesh of the unit cube [0,1]^3.
 */
inline Mesh MakeCubeMesh(int divisions) {
    Mesh mesh;
    const int n = divisions + 1;  // corner nodes per axis

    auto corner_index = [n](int i, int j, int k) {
        return i + n * (j + n * k);
    };

    // Lattice of corner nodes
    mesh.node_coords.reserve(static_cast<std::size_t>(n * n * n));
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                mesh.node_coords.emplace_back(
                    static_cast<double>(i) / divisions,
                    static_cast<double>(j) / divisions,
                    static_cast<double>(k) / divisions
                );
            }
        }
    }

    // Six tets per cube, all sharing the main diagonal 000-111.
    // Corner order chosen so det(J) > 0.
    static constexpr int kTetCorners[6][4] = {
        {0, 1, 3, 7},  // 000, 100, 110, 111
        {0, 3, 2, 7},  // 000, 110, 010, 111
        {0, 2, 6, 7},  // 000, 010, 011, 111
        {0, 6, 4, 7},  // 000, 011, 001, 111
        {0, 4, 5, 7},  // 000, 001, 101, 111
        {0, 5, 1, 7},  // 000, 101, 100, 111
    };

    std::map<std::pair<int, int>, int> edge_map;
    mesh.elements.reserve(
        static_cast<std::size_t>(6 * divisions * divisions * divisions)
    );

    for (int k = 0; k < divisions; ++k) {
        for (int j = 0; j < divisions; ++j) {
            for (int i = 0; i < divisions; ++i) {
                // 8 corners of this sub-cube, numbered 0..7 as binary xyz.
                const int cube[8] = {
                    corner_index(i, j, k),              // 0: 000
                    corner_index(i + 1, j, k),          // 1: 100
                    corner_index(i, j + 1, k),          // 2: 010
                    corner_index(i + 1, j + 1, k),      // 3: 110
                    corner_index(i, j, k + 1),          // 4: 001
                    corner_index(i + 1, j, k + 1),      // 5: 101
                    corner_index(i, j + 1, k + 1),      // 6: 011
                    corner_index(i + 1, j + 1, k + 1),  // 7: 111
                };
                for (int t = 0; t < 6; ++t) {
                    const int tet_corners[4] = {
                        cube[kTetCorners[t][0]],
                        cube[kTetCorners[t][1]],
                        cube[kTetCorners[t][2]],
                        cube[kTetCorners[t][3]],
                    };
                    mesh.elements.push_back(
                        AppendEdgeNodes(mesh, tet_corners, edge_map)
                    );
                }
            }
        }
    }
    return mesh;
}

}  // namespace test
}  // namespace matrix_free_fea
