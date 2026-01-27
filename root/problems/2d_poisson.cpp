#include "headers/2d_poisson.h"

template <typename TMatrix>
void create2dPoissonSystem
        (
                TMatrix& m,
                Vector& rhs,
                std::size_t numElemsPerDim,
                double eps_x,
                double eps_y
        )
{
    // number of INTERIOR elements per dimension
    const std::size_t n = numElemsPerDim;
    const double h = 1.0 / (n + 1);
    const double h_minus2 = (n + 1) * (n + 1);

    // square matrix for interior unknowns
    m.resize(n*n, n*n);
    rhs.resize(n*n, 2.0); // -Laplace u = 2

    // Helper to get boundary value
    auto get_boundary = [](double x, double y) {
        return 1.0 - 0.5*x*x - 0.5*y*y;
    };

    // fill matrix using pure 5-point stencil (Dirichlet 0 implicit)
    // We will adjust RHS for non-zero Dirichlet
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t idx = i * n + j;
            m(idx, idx) = (2.0 * (eps_x + eps_y)) * h_minus2;

            double x = (i + 1) * h;
            double y = (j + 1) * h;

            // Left Neighbor (j-1)
            if (j > 0) {
                const std::size_t idx_left = i * n + (j - 1);
                m(idx, idx_left) = -eps_x * h_minus2;
            } else {
                // Boundary x, y-h
                rhs[idx] += eps_x * h_minus2 * get_boundary(x, 0.0);
            }

            // Right Neighbor (j+1)
            if (j + 1 < n) {
                const std::size_t idx_right = i * n + (j + 1);
                m(idx, idx_right) = -eps_x *h_minus2;
            } else {
                // Boundary x, 1.0
                rhs[idx] += eps_x * h_minus2 * get_boundary(x, 1.0);
            }

            // Bottom Neighbor (i-1)
            if (i > 0) {
                const std::size_t idx_bottom = (i - 1) * n + j;
                m(idx, idx_bottom) = -eps_y * h_minus2;
            } else {
                // Boundary 0.0, y
                rhs[idx] += eps_y * h_minus2 * get_boundary(0.0, y);
            }

            // Top Neighbor (i+1)
            if (i + 1 < n) {
                const std::size_t idx_top = (i + 1) * n + j;
                m(idx, idx_top) = -eps_y * h_minus2;
            } else {
                // Boundary 1.0, y
                rhs[idx] += eps_y * h_minus2 * get_boundary(1.0, y);
            }
        }
    }
}

template void create2dPoissonSystem<SparseMatrix>
        (
                SparseMatrix& m,
                Vector& rhs,
                std::size_t numElemsPerDim,
                double eps_x,
                double eps_y
        );

template void create2dPoissonSystem<Matrix>
        (
                Matrix& m,
                Vector& rhs,
                std::size_t numElemsPerDim,
                double eps_x,
                double eps_y
        );