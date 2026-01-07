#include "headers/multigrid_solver.h"
#include "../data structures/headers/sparse_matrix.h"

template <typename TMatrix>
MultiGridSolver<TMatrix>::MultiGridSolver(IterativeSolver<TMatrix> &smoother, 
                                          LUSolver<TMatrix> &base_solver,
                                          int num_pre_smooth, 
                                          int num_post_smooth, 
                                          int num_cycles,
                                          int base_elements_per_dim)
    : smoother(smoother),
      base_solver(base_solver),
      num_pre_smooth(num_pre_smooth),
      num_post_smooth(num_post_smooth),
      num_cycles(num_cycles),
      base_elements_per_dim(base_elements_per_dim) {
    // Members initialized in initializer list
}

template <typename TMatrix>
bool MultiGridSolver<TMatrix>::solve(TMatrix &A, Vector &x, const Vector &b,
                                     std::size_t num_elements_per_dim) {
    if (num_elements_per_dim <= base_elements_per_dim) {
        // Directly use the base solver
        base_solver.set_matrix(&A);
        base_solver.solve(x, b);
        return true;
    }
    // Pre-smoothing to reduce high-frequency errors and thus interpolation errors
    for (int i = 0; i < num_pre_smooth; ++i) {
        smoother.set_matrix(&A);
        smoother.solve(x, b);
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
    TMatrix A_coarse = restriction_operator * A * restriction_operator.transpose(); //TODO make transpose multiplication efficient

    // Recursive call to solve on coarser grid
    Vector e_coarse(r_coarse.size(), 0.0); // Initialize error on coarse grid
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        MultiGridSolver<TMatrix> mg_solver(smoother, base_solver,
                                        num_pre_smooth, num_post_smooth,
                                        num_cycles, base_elements_per_dim);
        mg_solver.solve(A_coarse, e_coarse, r_coarse, num_elements_per_dim / 2);
    }

    // Prolongate error to fine grid and correct
    Vector e_fine = restriction_operator.transpose() * e_coarse; //TODO make transpose multiplication efficient
    x += e_fine;

    // Post-smoothing
    for (int i = 0; i < num_post_smooth; ++i) {
        smoother.set_matrix(&A);
        smoother.solve(x, b);
    }

    return true;
}

// Explicit template instantiations.
template class MultiGridSolver<Matrix>;
template class MultiGridSolver<SparseMatrix>;
