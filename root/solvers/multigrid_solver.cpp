#include "headers/multigrid_solver.h"
#include "../data structures/headers/sparse_matrix.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>

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
      bVerbose(false) {
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
    for (int i = 0; i < num_post_smooth; ++i) {
        // Fused mat-vec and subtraction to avoid an extra temporary
        Vector defect = A * x;
        defect -= b;
        Vector correction(defect.size());
        smoother.get_corrector()->apply(correction, defect);
        x -= correction;
    }
}

template<typename TMatrix>
bool MultiGridSolver<TMatrix>::init(const Vector &x) {
    auto start = std::chrono::high_resolution_clock::now();
    build_hierarchy(static_cast<std::size_t>(std::sqrt(x.size())));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(bVerbose) {
        std::cout << "Hierarchy build time: " << duration.count() << " ms" << std::endl;
    }
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
    bool initialized = base_solver.init(x);
    return initialized;
}

template <typename TMatrix>
void MultiGridSolver<TMatrix>::build_hierarchy(std::size_t finest_elements_per_dim) {
    // Clear cache before building new hierarchy
    hierarchy_cache.clear();
    
    // Build all levels from finest down to base
    std::size_t current_size = finest_elements_per_dim;
    
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
        
        // Build coarse grid operator (rediscretized Poisson)
        if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
            level.A_coarse = TMatrix(rows, rows, 5);
        } else {
            level.A_coarse = TMatrix(rows, rows);
        }
        
        level.h_minus2 = (n_coarse + 1) * (n_coarse + 1);
        // Just make a 5-point stencil from scratch
        for (std::size_t i = 0; i < n_coarse; ++i) {
            for (std::size_t j = 0; j < n_coarse; ++j) {
                std::size_t idx = i * n_coarse + j;
                level.A_coarse(idx, idx) = 4.0 * level.h_minus2;
                
                if (j > 0) { // left neighbor
                    std::size_t idx_left = i * n_coarse + (j - 1);
                    level.A_coarse(idx, idx_left) = -level.h_minus2;
                }
                if (j < n_coarse - 1) { // right neighbor
                    std::size_t idx_right = i * n_coarse + (j + 1);
                    level.A_coarse(idx, idx_right) = -level.h_minus2;
                }
                if (i > 0) { // bottom neighbor
                    std::size_t idx_bottom = (i - 1) * n_coarse + j;
                    level.A_coarse(idx, idx_bottom) = -level.h_minus2;
                }
                if (i < n_coarse - 1) { // top neighbor
                    std::size_t idx_top = (i + 1) * n_coarse + j;
                    level.A_coarse(idx, idx_top) = -level.h_minus2;
                }
            }
        }
        
        // Cache this level by its fine grid size
        hierarchy_cache[n_fine] = level;
        
        // Move to next coarser level
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

/**
 * @brief Solve A*x = b using multigrid approach. Assumes that A corresponds to
 * a 5-point stencil discretization of the Laplace operator on a square grid.
 * 
 * @tparam TMatrix 
 * @param A The system matrix
 * @param x The solution vector (initial guess on input)
 * @param b The right-hand side vector
 * @param num_elements_per_dim The number of elements per dimension (assumed square grid)
 * @return std::tuple<bool, size_t> Tuple of (converged, iterations).
 */
template <typename TMatrix>
std::tuple<bool, size_t> MultiGridSolver<TMatrix>::solve(
                        const TMatrix &A, Vector &x, const Vector &b,
                        std::size_t num_elements_per_dim) {
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
            if(iter==0) std::cout << "----------\t"; 
            else std::cout << std::fixed << std::setprecision(6) << rate << "\t";
            if(iter==0) std::cout << "-------------\n"; 
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

template class MultiGridSolver<Matrix>;
template class MultiGridSolver<SparseMatrix>;
