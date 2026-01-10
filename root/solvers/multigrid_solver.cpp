#include "headers/multigrid_solver.h"
#include "../data structures/headers/sparse_matrix.h"
#include <cmath>
#include <iomanip>

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

template <typename TMatrix>
void MultiGridSolver<TMatrix>::cycle(TMatrix &A, Vector &x, const Vector &b,
                                       std::size_t num_elements_per_dim) {
    // Always bind the smoother to the matrix of the current level
    smoother.set_matrix(&A);

    if (num_elements_per_dim <= base_elements_per_dim) {
        // Directly use the base solver
        base_solver.set_matrix(&A);
        base_solver.solve(x, b);
        return;
    }
    // Pre-smoothing to reduce high-frequency errors and thus interpolation errors
    for (int i = 0; i < num_pre_smooth; ++i) {
        // Apply one smoothing iteration: x := x - M^{-1} * (A*x - b)
        Vector defect = A * x - b;
        Vector correction(defect.size());
        smoother.get_corrector()->apply(correction, defect);
        x -= correction;
    }
    // Compute residual
    Vector r = b - A * x;
    // Restrict to coarse grid => turn low frequency errors to high frequency errors
    std::size_t n_fine = num_elements_per_dim;
    std::size_t n_coarse = num_elements_per_dim / 2;
    std::size_t rows = n_coarse * n_coarse;
    std::size_t cols = n_fine * n_fine;
    // If TMatrix is SparseMatrix, then row capacity is 4 (each coarse node connects to 4 fine nodes).
    TMatrix restriction_operator;
    if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
        restriction_operator = TMatrix(rows, cols, 4);
    } else {
        restriction_operator = TMatrix(rows, cols);
    }
    // Build 2D restriction operator (full-weighting with averaging)
    // Maps 4 fine grid cells to 1 coarse grid cell: (2i, 2j), (2i+1, 2j), (2i, 2j+1), (2i+1, 2j+1) -> (i, j)
    // Use weight 0.25 to average (not sum) the 4 fine values
    for (std::size_t i_coarse = 0; i_coarse < n_coarse; ++i_coarse) {
        for (std::size_t j_coarse = 0; j_coarse < n_coarse; ++j_coarse) {
            std::size_t coarse_idx = i_coarse * n_coarse + j_coarse;
            
            // Map to 4 fine grid points with weight 0.25 (averaging)
            for (std::size_t di = 0; di < 2; ++di) {
                for (std::size_t dj = 0; dj < 2; ++dj) {
                    std::size_t i_fine = 2 * i_coarse + di;
                    std::size_t j_fine = 2 * j_coarse + dj;
                    std::size_t fine_idx = i_fine * n_fine + j_fine;
                    restriction_operator(coarse_idx, fine_idx) = 0.25;
                }
            }
        }
    }

    Vector r_coarse = restriction_operator * r;
    
    // Build 2D prolongation operator (bilinear interpolation - replication)
    // Maps 1 coarse grid cell to 4 fine grid cells with weight 1.0 (replicate, not average)
    TMatrix prolongation_operator;
    if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
        prolongation_operator = TMatrix(cols, rows, 1); // transpose dimensions, 1 entry per row
    } else {
        prolongation_operator = TMatrix(cols, rows);
    }
    
    for (std::size_t i_coarse = 0; i_coarse < n_coarse; ++i_coarse) {
        for (std::size_t j_coarse = 0; j_coarse < n_coarse; ++j_coarse) {
            std::size_t coarse_idx = i_coarse * n_coarse + j_coarse;
            
            // Each coarse value is replicated to 4 fine grid points with weight 1.0
            for (std::size_t di = 0; di < 2; ++di) {
                for (std::size_t dj = 0; dj < 2; ++dj) {
                    std::size_t i_fine = 2 * i_coarse + di;
                    std::size_t j_fine = 2 * j_coarse + dj;
                    std::size_t fine_idx = i_fine * n_fine + j_fine;
                    prolongation_operator(fine_idx, coarse_idx) = 1.0;
                }
            }
        }
    }
    
    // Use rediscretization for coarse grid operator to preserve boundary conditions
    // Build the same Poisson discretization on coarser grid
    TMatrix A_coarse;
    if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
        A_coarse = TMatrix(rows, rows, 5);
    } else {
        A_coarse = TMatrix(rows, rows);
    }
    
    const double h_minus2 = (n_coarse + 1) * (n_coarse + 1);
    
    // Build 2D Poisson stencil on coarse grid
    for (std::size_t i = 0; i < n_coarse; ++i) {
        for (std::size_t j = 0; j < n_coarse; ++j) {
            std::size_t idx = i * n_coarse + j;
            
            // Diagonal entry (4-point stencil center)
            A_coarse(idx, idx) = 4.0 * h_minus2;
            
            // Off-diagonal entries (neighbors)
            if (j > 0) { // left neighbor
                std::size_t idx_left = i * n_coarse + (j - 1);
                A_coarse(idx, idx_left) = -h_minus2;
            }
            if (j < n_coarse - 1) { // right neighbor
                std::size_t idx_right = i * n_coarse + (j + 1);
                A_coarse(idx, idx_right) = -h_minus2;
            }
            if (i > 0) { // bottom neighbor
                std::size_t idx_bottom = (i - 1) * n_coarse + j;
                A_coarse(idx, idx_bottom) = -h_minus2;
            }
            if (i < n_coarse - 1) { // top neighbor
                std::size_t idx_top = (i + 1) * n_coarse + j;
                A_coarse(idx, idx_top) = -h_minus2;
            }
        }
    }

    // Recursive call to cycle on coarser grid
    Vector e_coarse(r_coarse.size(), 0.0); // Initialize error on coarse grid
    for (int recursion = 1; recursion <= num_recursions; ++recursion) { // num. of recursion cycles
        this->cycle(A_coarse, e_coarse, r_coarse, num_elements_per_dim / 2);
    }

    // Prolongate error to fine grid and correct
    Vector e_fine = prolongation_operator * e_coarse;
    x += e_fine;

    // Post-smoothing
    smoother.set_matrix(&A); // reset to current level after recursive call
    for (int i = 0; i < num_post_smooth; ++i) {
        // Apply one smoothing iteration: x := x - M^{-1} * (A*x - b)
        Vector defect = A * x - b;
        Vector correction(defect.size());
        smoother.get_corrector()->apply(correction, defect);
        x -= correction;
    }
}

template <typename TMatrix>
std::tuple<bool, size_t> MultiGridSolver<TMatrix>::solve(TMatrix &A, Vector &x, const Vector &b,
                                     std::size_t num_elements_per_dim) {
    // Compute initial defect norm
    Vector residual = A * x - b;
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
        
        // Compute residual norm
        residual = A * x - b;
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
