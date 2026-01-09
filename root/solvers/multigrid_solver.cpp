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
    std::size_t rows = (num_elements_per_dim / 2) * (num_elements_per_dim / 2);
    std::size_t cols = num_elements_per_dim * num_elements_per_dim;
    // If TMatrix is SparseMatrix, then row capacity is 2.
    TMatrix restriction_operator;
    if constexpr (std::is_same<TMatrix, SparseMatrix>::value) {
        restriction_operator = TMatrix(rows, cols, 2);
    } else {
        restriction_operator = TMatrix(rows, cols);
    }
    // Build restriction operator (injection - simple summation) 
    // -> transpose of prolongation (saves space but is not optimal accuracy)
    for (std::size_t i = 0; i < rows; ++i) {
        restriction_operator(i, 2*i) = 1.0;
        restriction_operator(i, 2*i + 1) = 1.0;
    }

    Vector r_coarse = restriction_operator * r;
    // A_coarse is approx. r * A * p (algebraic multigrid)
    TMatrix A_coarse = restriction_operator * A * restriction_operator.transpose(); //TODO: make transpose multiplication efficient

    // Recursive call to cycle on coarser grid
    Vector e_coarse(r_coarse.size(), 0.0); // Initialize error on coarse grid
    for (int recursion = 1; recursion <= num_recursions; ++recursion) { // num. of recursion cycles
        MultiGridSolver<TMatrix> mg_solver(smoother, base_solver); //TODO:: avoid creating new object each recursion
        mg_solver.set_parameters(num_pre_smooth, num_post_smooth,
                                 num_recursions, base_elements_per_dim);
        mg_solver.cycle(A_coarse, e_coarse, r_coarse, num_elements_per_dim / 2);
    }

    // Prolongate error to fine grid and correct
    Vector e_fine = restriction_operator.transpose() * e_coarse; //TODO: make transpose multiplication efficient
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
