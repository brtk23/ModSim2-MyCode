#include "headers/multigrid_solver.h"
#include "../data structures/headers/sparse_matrix.h"
#include "../data structures/headers/matrix.h"
#include "../problems/headers/2d_poisson_settings.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <type_traits>

// Build a standard 5-point Laplace stencil matrix on a square grid with
// num_elements_per_dim unknowns per dimension. This mirrors the assumption
// used throughout the multigrid implementation.
template <typename TMatrix>
TMatrix build_poisson_matrix(std::size_t num_elements_per_dim) {
    double eps_x = PoissonSettings::get_EPS_X();
    double eps_y = PoissonSettings::get_EPS_Y();
    const std::size_t n = num_elements_per_dim;
    const std::size_t n_unknowns = n * n;

    TMatrix A = [] (std::size_t rows, std::size_t cols) {
        if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
            return TMatrix(rows, cols, 5);
        } else {
            return TMatrix(rows, cols, 0.0);
        }
    }(n_unknowns, n_unknowns);

    const double h_minus2 = (n + 1) * (n + 1);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t idx = i * n + j;
            A(idx, idx) = (2.0 * (eps_x + eps_y)) * h_minus2;;

            if (j > 0) { // Left Neighbor (j-1)
                const std::size_t idx_left = i * n + (j - 1);
                A(idx, idx_left) = -eps_x * h_minus2;
            }
            if (j + 1 < n) { // Right Neighbor (j+1)
                const std::size_t idx_right = i * n + (j + 1);
                A(idx, idx_right) = -eps_x * h_minus2;
            }
            if (i > 0) { // Bottom Neighbor (i-1)
                const std::size_t idx_bottom = (i - 1) * n + j;
                A(idx, idx_bottom) = -eps_y * h_minus2;
            }
            if (i + 1 < n) { // Top Neighbor (i+1)
                const std::size_t idx_top = (i + 1) * n + j;
                A(idx, idx_top) = -eps_y * h_minus2;
            }
        }
    }

    return A;
}

template <typename TMatrix>
MultiGridSolver<TMatrix>::MultiGridSolver(IterativeSolver<TMatrix> &smoother, 
                                          LUSolver<TMatrix> &base_solver)
    : smoother(smoother),
      base_solver(base_solver),
      num_pre_smooth(2),
      num_post_smooth(2),
      num_recursions(1),
      base_elements_per_dim(2),
      max_iterations(50),
      min_defect(1e-15),
      min_reduction(1e-8),
      bVerbose(false),
      bUseRAP(false),
      A_finest(nullptr) {
    // Members initialized in initializer list
}

/**
 * @brief Perform a single multigrid cycle, assumes that original grid is square
 * and num_elements_per_dim is the number of elements per dimension. Also
 * assumes that the system matrix A corresponds to a 5-point stencil discretization
 * of the Laplace operator on a square grid.
 * 
 * Also assumes that the hierarchy for the given num_elements_per_dim has already been built.
 * 
 * @tparam TMatrix 
 * @param A System matrix at current level
 * @param x Solution vector at current level
 * @param b Right-hand side vector at current level
 * @param num_elements_per_dim Number of elements per dimension at current level,
 *                             assumed square grid!
 */
template <typename TMatrix>
void MultiGridSolver<TMatrix>::cycle(const TMatrix &A, Vector &x, const Vector &b,
                                       std::size_t num_elements_per_dim) {
    // Always bind the smoother to the matrix of the current level
    smoother.set_matrix(&A);
    smoother.init(x); // x is never used but for interface compliance

    if (num_elements_per_dim <= base_elements_per_dim) {
        // Directly use the base solver at the coarsest level
        // Matrix A has already been set in initialization
        base_solver.solve(x, b);
        return;
    }
    
    // Pre-smoothing to reduce high-frequency errors
    for (int i = 0; i < num_pre_smooth; ++i) {
        // Fused mat-vec and subtraction to avoid an extra temporary
        Vector defect = A * x;
        defect -= b;
        Vector correction(defect.size());
        smoother.get_corrector()->apply(correction, defect);
        x -= correction;
    }
    
    // Compute residual r = b - A*x without extra temporaries
    Vector r = A * x;
    r *= -1.0;
    r += b;
    
    // Get cached operators for this grid transition
    const GridLevel<TMatrix>& level = get_grid_level(num_elements_per_dim);
    
    // Restrict residual to coarse grid, low frequency errors become high frequency
    Vector r_coarse = level.restriction * r;
    
    // Recursive call to cycle on coarser grid
    Vector e_coarse(r_coarse.size(), 0.0);
    for (int recursion = 1; recursion <= num_recursions; ++recursion) {
        this->cycle(level.A_coarse, e_coarse, r_coarse, num_elements_per_dim / 2);
    }

    // Prolongate error to fine grid and correct
    Vector e_fine = level.prolongation * e_coarse;
    x += e_fine;

    // Post-smoothing
    smoother.set_matrix(&A);
    smoother.init(x); // x is never used but for interface compliance
    for (int i = 0; i < num_post_smooth; ++i) {
        // Fused mat-vec and subtraction to avoid an extra temporary
        Vector defect = A * x;
        defect -= b;
        Vector correction(defect.size());
        smoother.get_corrector()->apply(correction, defect);
        x -= correction;
    }
}

/**
 * @brief Solve A*x = b using multigrid approach. Assumes that A corresponds to
 * a 5-point stencil discretization of the Laplace operator on a square grid.
 * 
 * @tparam TMatrix 
 * @param x The solution vector (initial guess on input)
 * @param b The right-hand side vector
 * @param num_elements_per_dim The number of elements per dimension (assumed square grid)
 * @return std::tuple<bool, size_t> Tuple of (converged, iterations).
 */
template <typename TMatrix>
std::tuple<bool, size_t> MultiGridSolver<TMatrix>::solve(
                        Vector &x, const Vector &b,
                        std::size_t num_elements_per_dim) {
    if (!A_finest) {
        throw std::runtime_error("MultiGridSolver: Matrix not set!");
    }
    const TMatrix& A = *A_finest;
    if (!initialized) {
        throw std::runtime_error("MultiGridSolver: Not initialized! Call init() before solve().");
    }

    // Compute initial defect norm (fused A*x - b)
    Vector residual = A * x;
    residual -= b;
    double residual_norm = residual.norm();
    double prev_residual_norm = 0.0;
    double b_norm = b.norm();
    
    const double initial_residual_norm = residual_norm;
    
    // Check if already converged
    if (residual_norm / b_norm < min_reduction || residual_norm < min_defect) {
        return {true, 0};
    }
    

    if(bVerbose) {
        std::cout << "## MultiGridSolver #############################################################" << std::endl;
        std::cout << "Iter\tDefect\t\tRate\t\tReduction" << std::endl;
    }

    // Multigrid iteration loop
    double rate = 0.0;
    double reduction = 0.0;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        prev_residual_norm = residual_norm;
        
        // Perform one cycle
        cycle(A, x, b, num_elements_per_dim);
        
        // Compute residual norm (fused A*x - b)
        residual = A * x;
        residual -= b;
        residual_norm = residual.norm();
        if(bVerbose) {
            if(iter > 0) rate = residual_norm / prev_residual_norm;
            if(iter > 0) reduction = residual_norm / initial_residual_norm;
            std::cout << iter << "\t" << std::scientific << std::setprecision(6) 
                      << residual_norm << "\t";
            if(iter==0) std::cout << "--------\t"; 
            else std::cout << std::fixed << std::setprecision(6) << rate << "\t";
            if(iter==0) std::cout << "-------\n"; 
            else std::cout << std::scientific << std::setprecision(6) << reduction << std::endl;
        }
        // Check convergence: absolute defect or relative reduction
        if (residual_norm < min_defect || residual_norm / initial_residual_norm < min_reduction) {
            if(bVerbose) {
                std::cout << "MultiGridSolver: Converged in " << (iter+1) << " iterations." << std::endl;
            }
            return {true, iter+1}; // Converged
        }
    }
    
    // Did not converge within max_iterations
    if(bVerbose) {
        std::cout << "MultiGridSolver: Not converged after " << max_iterations << " iterations." << std::endl;
    }
    return {false, max_iterations};
}

template<typename TMatrix>
bool MultiGridSolver<TMatrix>::init(const Vector &x) {
    // Build hierarchy
    auto start = std::chrono::high_resolution_clock::now();
    const std::size_t finest_elements_per_dim = static_cast<std::size_t>(std::sqrt(x.size()));
    build_hierarchy_test(finest_elements_per_dim);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(bVerbose) {
        std::cout << "Hierarchy build time: " << duration.count() << " ms" << std::endl;
    }
    // Initialize base solver at coarsest level
    const TMatrix* A_base = &hierarchy_cache.begin()->second.A_coarse;
    if (bVerbose) {
        int rows = std::sqrt(A_base->num_rows());
        int cols = std::sqrt(A_base->num_cols());
        std::cout << "Initializing base solver... (" 
                << rows << " x " << cols << ") => ("
                << A_base->num_rows() << " x " << A_base->num_cols() << ")"
                << std::endl;
    }
    base_solver.set_matrix(A_base);
    bool base_initialized = base_solver.init(x);
    if (!base_initialized) {
        return false;
    }

    initialized = true;
    return true;
}

/**
 * @brief Builds hierarchy using 4 point averaging restriction and
 * injection prolongation operators, with rediscretization for coarse grid operators.
 * 
 * @tparam TMatrix 
 * @param finest_elements_per_dim top level square grid size
 */
template <typename TMatrix>
void MultiGridSolver<TMatrix>::build_hierarchy(std::size_t finest_elements_per_dim) {
    // Clear cache before building new hierarchy
    hierarchy_cache.clear();
    
    // Build all levels from finest down to base
    std::size_t current_size = finest_elements_per_dim;

    const TMatrix* A_current = A_finest;
    if (bUseRAP && !A_current) {
         throw std::runtime_error("MultiGridSolver: Matrix not set, cannot build hierarchy with RAP!");
    }
    
    while (current_size > base_elements_per_dim) {
        std::size_t n_fine = current_size;
        std::size_t n_coarse = current_size / 2;
        std::size_t rows = n_coarse * n_coarse;
        std::size_t cols = n_fine * n_fine;
        
        GridLevel<TMatrix> level;
        
        // Build restriction operator (R: fine -> coarse)
        if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
            level.restriction = TMatrix(rows, cols, 4);
        } else {
            level.restriction = TMatrix(rows, cols);
        }
        // 4 point averaging - each coarse grid point gets contributions from 4 fine grid points
        for (std::size_t i_coarse = 0; i_coarse < n_coarse; ++i_coarse) {
            for (std::size_t j_coarse = 0; j_coarse < n_coarse; ++j_coarse) {
                std::size_t coarse_idx = i_coarse * n_coarse + j_coarse;
                for (std::size_t di = 0; di < 2; ++di) {
                    for (std::size_t dj = 0; dj < 2; ++dj) {
                        std::size_t i_fine = 2 * i_coarse + di;
                        std::size_t j_fine = 2 * j_coarse + dj;
                        std::size_t fine_idx = i_fine * n_fine + j_fine;
                        level.restriction(coarse_idx, fine_idx) = 0.25;
                    }
                }
            }
        }

        // Build prolongation operator (P: coarse -> fine)
        if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
            level.prolongation = TMatrix(cols, rows, 1);
                } else {
            level.prolongation = TMatrix(cols, rows);
        }
        // Coarse cell spreads equally to 4 fine cells
        for (std::size_t i_coarse = 0; i_coarse < n_coarse; ++i_coarse) {
            for (std::size_t j_coarse = 0; j_coarse < n_coarse; ++j_coarse) {
                std::size_t coarse_idx = i_coarse * n_coarse + j_coarse;
                for (std::size_t di = 0; di < 2; ++di) {
                    for (std::size_t dj = 0; dj < 2; ++dj) {
                        std::size_t i_fine = 2 * i_coarse + di;
                        std::size_t j_fine = 2 * j_coarse + dj;
                        std::size_t fine_idx = i_fine * n_fine + j_fine;
                        level.prolongation(fine_idx, coarse_idx) = 1.0;
                    }
                }
            }
        }
        
        // Build coarse grid operator by rediscretization (standard for structured grids)
        // This is superior to Galerkin product for finite difference discretizations
        // because it preserves stencil structure and conditioning across levels
        if (bUseRAP) {
            // Galerkin Product: A_coarse = R * A_fine * P
            TMatrix RA = level.restriction * (*A_current);
            level.A_coarse = RA * level.prolongation;
        } else {
            level.A_coarse = build_poisson_matrix<TMatrix>(n_coarse);
        }
        level.h_minus2 = (n_coarse + 1) * (n_coarse + 1);
        
        // Cache this level by its fine grid size
        hierarchy_cache[n_fine] = level;

        if (bUseRAP) {
             A_current = &hierarchy_cache[n_fine].A_coarse;
        }
        
        // Move to next coarser level
        current_size = n_coarse;
    }
}

/**
 * @brief Builds hierarchy using state-of-the-art restriction/prolongation
 * operators (Bilinear/Full Weighting) with rediscretization for coarse grid operators.
 * 
 * @tparam TMatrix 
 * @param finest_elements_per_dim the top level square grid size
 */
template <typename TMatrix>
void MultiGridSolver<TMatrix>::build_hierarchy_test(std::size_t finest_elements_per_dim) {
    // Clear cache before building new hierarchy
    hierarchy_cache.clear();
    
    // Build all levels from finest down to base
    std::size_t current_size = finest_elements_per_dim;

    const TMatrix* A_current = A_finest; // only needed if using RAP
    if (bUseRAP && !A_current) {
         throw std::runtime_error("MultiGridSolver: Matrix not set, cannot build hierarchy with RAP!");
    }
    
    while (current_size > base_elements_per_dim) {
        std::size_t n_fine = current_size;
        std::size_t n_coarse = current_size / 2;
        if (n_coarse < 2 && base_elements_per_dim >=2) {
            n_coarse = 2; // ensure we don't go below base grid (2x2)
        }
        std::size_t rows = n_coarse * n_coarse;
        std::size_t cols = n_fine * n_fine;
        
        GridLevel<TMatrix> level;
        
        // Allocate matrices
        // R: coarse x fine (max 16 entries per row for 2D cell-centered linear interpolation)
        // P: fine x coarse (max 4 entries per row)
        if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
            level.restriction = TMatrix(rows, cols, 16);
            level.prolongation = TMatrix(cols, rows, 4);
        } else {
            level.restriction = TMatrix(rows, cols);
            level.prolongation = TMatrix(cols, rows);
        }

        // Cell-Centered Linear Interpolation
        // Weights derived from 1D linear interpolation on cell centers:
        // Fine_L = 0.75 * Coarse_Self + 0.25 * Coarse_Left
        // Fine_R = 0.75 * Coarse_Self + 0.25 * Coarse_Right
        //
        // 2D Tensor product weights:
        // Self-Self (0.75*0.75) = 9/16
        // Self-Neigh (0.75*0.25) = 3/16
        // Neigh-Neigh (0.25*0.25) = 1/16

        for (std::size_t i_fine = 0; i_fine < n_fine; ++i_fine) {
            for (std::size_t j_fine = 0; j_fine < n_fine; ++j_fine) {
                std::size_t fine_idx = i_fine * n_fine + j_fine;
                
                // Parent coarse cell index
                std::int64_t i_c = i_fine / 2;
                std::int64_t j_c = j_fine / 2;
                
                // Position within parent (0=left/bottom, 1=right/top)
                std::int64_t di = i_fine % 2;
                std::int64_t dj = j_fine % 2;
                
                // Identify the "secondary" coarse neighbor we pull from
                // If di=0 (left), we pull from Left (i_c - 1)
                // If di=1 (right), we pull from Right (i_c + 1)
                std::int64_t i_neigh = (di == 0) ? i_c - 1 : i_c + 1;
                std::int64_t j_neigh = (dj == 0) ? j_c - 1 : j_c + 1;
                
                // Contributors with weights
                struct Contributor { std::int64_t i; std::int64_t j; double w; };
                Contributor contributors[] = {
                    {i_c,     j_c,     9.0/16.0}, // Center
                    {i_neigh, j_c,     3.0/16.0}, // Horizontal neighbor
                    {i_c,     j_neigh, 3.0/16.0}, // Vertical neighbor
                    {i_neigh, j_neigh, 1.0/16.0}  // Diagonal neighbor
                };

                for (const auto& c : contributors) {
                    std::int64_t target_i = c.i;
                    std::int64_t target_j = c.j;
                    double weight = c.w;

                    // Apply homogenous Dirichlet BCs (u=0 at boundary) using reflection/ghost points
                    // u_minus_1 = -u_0
                    // u_N = -u_N_minus_1
                    
                    bool valid = true;
                    
                    if (target_i < 0) {
                        target_i = 0;
                        weight *= -1.0;
                    } else if (target_i >= (std::int64_t)n_coarse) {
                        target_i = n_coarse - 1;
                        weight *= -1.0;
                    }

                    if (target_j < 0) {
                        target_j = 0;
                        weight *= -1.0;
                    } else if (target_j >= (std::int64_t)n_coarse) {
                        target_j = n_coarse - 1;
                        weight *= -1.0;
                    }
                    
                    // Note: If we reflect twice (corner ghost), weight becomes positive again (-1 * -1),
                    // which is correct for u_-1,-1 = -u_0,-1 = -(-u_0,0) = u_0,0 ??
                    // Actually, at a corner, u(x,y) near 0,0 is ~ c*x*y.
                    // u(-h, -h) ~ c*(-h)*(-h) = c*h*h = u(h,h).
                    // So corner reflection should be positive. Logic holds.

                    std::size_t coarse_idx = target_i * n_coarse + target_j;
                    
                    // P: Add contribution
                    // Note: sparse matrix usually sums duplicates if insertion repeated, 
                    // but here we have unique (fine, coarse) pairs for this fine loop?
                    // Actually, re-mapping ghost points to internal might cause duplicates 
                    // if multiple contributors map to the same boundary node?
                    // Example: At corner fine (0,0).
                    // i_c=0, j_c=0. i_neigh=-1, j_neigh=-1.
                    // Contribs:
                    // (0,0) w=9/16 -> (0,0) w=9/16
                    // (-1,0) w=3/16 -> (0,0) w=-3/16
                    // (0,-1) w=3/16 -> (0,0) w=-3/16
                    // (-1,-1) w=1/16 -> (0,0) w=1/16
                    // Total P(0,0 from 0,0) = 9-3-3+1 = 4/16 = 0.25.
                    // Correct! u_F[0] = 0.5 * 0.5 * u_C[0] = 0.25 u_C[0].
                    // So we MUST use += to accumulate weights.
                    
                    level.prolongation(fine_idx, coarse_idx) += weight;
                    
                    // R: Add contribution (P^T scaled by 0.25)
                    // We can accumulate R as well.
                    level.restriction(coarse_idx, fine_idx) += weight * 0.25;
                }
            }
        }
        
        // Build coarse grid operator
        if (bUseRAP) {
            // Galerkin Product: A_coarse = R * A_fine * P
             TMatrix RA = level.restriction * (*A_current);
             level.A_coarse = RA * level.prolongation;
        } else {
            // Rediscretization
             level.A_coarse = build_poisson_matrix<TMatrix>(n_coarse);
        }
        level.h_minus2 = (n_coarse + 1) * (n_coarse + 1);
        
        hierarchy_cache[n_fine] = level;

        if (bUseRAP) {
             A_current = &hierarchy_cache[n_fine].A_coarse;
        }
        current_size = n_coarse;
    }
}

template <typename TMatrix>
const GridLevel<TMatrix>& MultiGridSolver<TMatrix>::get_grid_level(std::size_t num_elements_per_dim) {
    auto it = hierarchy_cache.find(num_elements_per_dim);
    if (it != hierarchy_cache.end()) {
        return it->second;
    }
    // Should not happen if hierarchy was built correctly
    throw std::runtime_error("Grid level not found in hierarchy cache");
}



template class MultiGridSolver<Matrix>;
template class MultiGridSolver<SparseMatrix>;
