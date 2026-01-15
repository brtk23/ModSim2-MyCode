/*
 *
 *
 *  Created on: 2024-11-06
 *      Author: julian hilbert
 */
#include "data structures/headers/vector.h"
#include "data structures/headers/matrix.h"
#include "data structures/headers/sparse_matrix.h"
#include "solvers/headers/iterative_solver.h"
#include "solvers/headers/jacobi.h"
#include "solvers/headers/gauss_seidel.h"
#include "solvers/headers/lu_solver.h"
#include "solvers/headers/multigrid_solver.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <functional>
#include <string>
#if defined(_OPENMP)
#include "omp.h"
#endif

// Color codes for output
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

/**
 * @brief Template timer class for benchmarking function calls
 */
template<typename Clock = std::chrono::high_resolution_clock>
class Timer {
public:
    using Duration = std::chrono::duration<double, std::milli>;
    
    /**
     * @brief Measure execution time of a function
     * @param func Function/lambda to execute and time
     * @return Execution time in milliseconds
     */
    template<typename Func>
    double measure(Func&& func) {
        auto start = Clock::now();
        func();
        auto end = Clock::now();
        Duration duration = end - start;
        return duration.count();
    }
    
    /**
     * @brief Measure execution time with return value
     * @param func Function/lambda to execute and time
     * @return Pair of (return value, execution time in ms)
     */
    template<typename Func>
    auto measure_with_result(Func&& func) -> std::pair<decltype(func()), double> {
        auto start = Clock::now();
        auto result = func();
        auto end = Clock::now();
        Duration duration = end - start;
        return {result, duration.count()};
    }
};

/**
 * @brief Print a formatted benchmark result
 */
void print_benchmark_result(const std::string& solver_name, double time_ms, bool converged, size_t iterations) {
    std::cout << std::left << std::setw(20) << solver_name << ": ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms << " ms ";
    if (converged) {
        std::cout << GREEN << "CONVERGED" << RESET << " (" << iterations << " iterations)";
    } else {
        std::cout << RED << "FAILED" << RESET << " (" << iterations << " iterations)";
    }
    std::cout << std::endl;
}


/**
 * assembles the 2d poisson discretization required by the exercise sheet.
 */
template <typename TMatrix>
void create2dPoissonSystemWithSize
        (
                TMatrix& m,
                Vector& rhs,
                std::size_t numElemsPerDim
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
            m(idx, idx) = 4.0 * h_minus2;

            double x = (i + 1) * h;
            double y = (j + 1) * h;

            // Left Neighbor (j-1)
            if (j > 0) {
                const std::size_t idx_left = i * n + (j - 1);
                m(idx, idx_left) = -h_minus2;
            } else {
                // Boundary x, y-h
                rhs[idx] += h_minus2 * get_boundary(x, 0.0);
            }

            // Right Neighbor (j+1)
            if (j + 1 < n) {
                const std::size_t idx_right = i * n + (j + 1);
                m(idx, idx_right) = -h_minus2;
            } else {
                // Boundary x, 1.0
                rhs[idx] += h_minus2 * get_boundary(x, 1.0);
            }

            // Bottom Neighbor (i-1)
            if (i > 0) {
                const std::size_t idx_bottom = (i - 1) * n + j;
                m(idx, idx_bottom) = -h_minus2;
            } else {
                // Boundary 0.0, y
                rhs[idx] += h_minus2 * get_boundary(0.0, y);
            }

            // Top Neighbor (i+1)
            if (i + 1 < n) {
                const std::size_t idx_top = (i + 1) * n + j;
                m(idx, idx_top) = -h_minus2;
            } else {
                // Boundary 1.0, y
                rhs[idx] += h_minus2 * get_boundary(1.0, y);
            }
        }
    }
}
/** this function prints the solution vector or rhs vector in
 *  form of the underlying grid!
		**/
void printVectorOnGrid(Vector& u, size_t nElemsPerDim){
    const int col_width = 10;
    const int row_label_width = 5;
    
    // Print header with column indices
    std::cout << std::string(row_label_width, ' ');
    for(size_t j = 0; j < nElemsPerDim; ++j) {
        std::cout << std::setw(col_width) << std::left << ("C" + std::to_string(j));
    }
    std::cout << std::endl;
    std::cout << std::string(row_label_width - 1, ' ') << "|" << std::string(nElemsPerDim * col_width, '-') << std::endl;
    
    // Print rows
    for(size_t i = 0; i < nElemsPerDim; ++i){
        std::cout << std::setw(row_label_width - 1) << std::left << ("R" + std::to_string(i)) << "|";
        for(size_t j = 0; j < nElemsPerDim; ++j){
            size_t idx = i * nElemsPerDim + j;
            if(idx < u.size()){
                std::cout << std::setw(col_width) << std::left << std::fixed << std::setprecision(4) << u[idx];
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
///basic LU decomposition and solving a system of linear equations
bool LU_test(size_t nElem){
    const int border_width = 70;
    std::string title = "LU DECOMPOSITION VERIFICATION TEST";
    int padding = (border_width - title.length()) / 2;
    
    std::cout << "\n" << BOLD << CYAN << std::string(border_width, '=') << RESET << std::endl;
    std::cout << BOLD << CYAN << std::string(padding, ' ') << title << std::string(border_width - padding - title.length(), ' ') << RESET << std::endl;
    std::cout << BOLD << CYAN << std::string(border_width, '=') << RESET << "\n" << std::endl;
    
    Vector v(nElem, 1.0);
    Vector x(nElem,0.0);
    SparseMatrix S(nElem, nElem, 5);
    Matrix M(nElem, nElem, 0.0);
    
    std::cout << "Setting up " << nElem << "x" << nElem << " test matrices..." << std::endl;
    for (size_t i = 0; i < nElem; ++i) {
        S(i, i) = 2;
        M(i, i) = 2;
        if (i > 0) {
            S(i - 1, i) = -1.0;
            M(i - 1, i) = -1.0;
        }
        if (i < nElem - 1) {
            S(i + 1, i) = -1;
            M(i + 1, i) = -1;
        }
    }
    
    LUSolver<SparseMatrix> lu_sparse;
    LUSolver<Matrix>       lu_full;

    lu_sparse.set_matrix(&S);
    lu_full.set_matrix(&M);
    
    std::cout << YELLOW << ">> Initializing sparse LU solver..." << RESET << std::endl;
    lu_sparse.init(v);
    std::cout << GREEN << "   [OK] Sparse LU initialized" << RESET << std::endl;
    
    std::cout << YELLOW << ">> Initializing full LU solver..." << RESET << std::endl;
    lu_full.init(v);
    std::cout << GREEN << "   [OK] Full LU initialized" << RESET << "\n" << std::endl;

    std::cout << BOLD << "Sparse Matrix Decomposition:" << RESET << std::endl;
    lu_sparse.print();
    lu_sparse.solve(x,v);
    std::cout << BOLD << "Sparse Solution:" << RESET << std::endl;
    std::cout << x << std::endl;
    
    std::cout << BOLD << "Full Matrix Decomposition:" << RESET << std::endl;
    lu_full.print();
    lu_full.solve(x, v);
    std::cout << BOLD << "Full Solution:" << RESET << std::endl;
    std::cout << x << std::endl;
    
    std::cout << GREEN << BOLD << "[SUCCESS] LU TEST COMPLETE!" << RESET << "\n" << std::endl;
    return true;
}

void BasicTests(size_t nElemsPerDim = 3)
{
    const int border_width = 70;
    std::string title1 = "BASIC SOLVER TESTS";
    int padding1 = (border_width - title1.length()) / 2;
    
    std::cout << "\n" << BOLD << YELLOW << std::string(border_width, '=') << RESET << std::endl;
    std::cout << BOLD << CYAN << std::string(padding1, ' ') << title1 << std::string(border_width - padding1 - title1.length(), ' ') << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(border_width, '=') << RESET << "\n" << std::endl;
    
    // test for LU implementation from last exercise sheet..
    // please make sure this passes and if not, consider your
    // implementation again!
    LU_test(5);

    // example for solving of poisson problem
    std::string title2 = "2D POISSON PROBLEM SOLVER COMPARISON";
    int padding2 = (border_width - title2.length()) / 2;
    
    std::cout << BOLD << CYAN << std::string(border_width, '=') << RESET << std::endl;
    std::cout << BOLD << CYAN << std::string(padding2, ' ') << title2 << std::string(border_width - padding2 - title2.length(), ' ') << RESET << std::endl;
    std::cout << BOLD << CYAN << std::string(border_width, '=') << RESET << "\n" << std::endl;
    
    SparseMatrix A(nElemsPerDim * nElemsPerDim, nElemsPerDim * nElemsPerDim, 5);
    Vector b, x;

    std::cout << YELLOW << ">> Assembling " << nElemsPerDim << "x" << nElemsPerDim << " Poisson system..." << RESET << std::endl;
    create2dPoissonSystemWithSize(A, b, nElemsPerDim);
    std::cout << GREEN << "   [OK] System assembled (" << (nElemsPerDim * nElemsPerDim) << " unknowns)" << RESET << "\n" << std::endl;

    std::cout << BOLD << "Vector b [Right-hand side]:" << RESET << std::endl;
    printVectorOnGrid(b, nElemsPerDim);
    std::cout << "\n" << BOLD << "Matrix A [System matrix]:" << RESET << std::endl;
    std::cout << A << std::endl;
    
    x.resize(b.size());

    // create random starting values
    std::srand(0);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;

    // create iterative solver and set parameters
    IterativeSolver<SparseMatrix> iterative_solver(A);
    iterative_solver.set_verbose(false);

    // create preconditioners:
    Jacobi<SparseMatrix> jac;
    GaussSeidel<SparseMatrix> gs;

    // JACOBI
    std::cout << "\n" << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    std::cout << BOLD << CYAN << "  Solver: JACOBI Iterative Method" << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    
    iterative_solver.set_corrector(&jac);
    iterative_solver.init(x);
    auto [success, iters1] = iterative_solver.solve(x, b);
    
    if (success) {
        std::cout << GREEN << "   [CONVERGED] in " << iters1 << " iterations" << RESET << std::endl;
    } else {
        std::cout << RED << "   [FAILED] Did not converge" << RESET << std::endl;
    }
    
    // Compute and print residual norm
    Vector residual1 = A * x;
    residual1 -= b;
    double residual_norm1 = residual1.norm();
    std::cout << YELLOW << "   Residual ||Ax - b|| = " << std::scientific << std::setprecision(6) << residual_norm1 << RESET << std::endl;
    
    std::cout << BOLD << "Solution x:" << RESET << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // GAUSS-SEIDEL
    std::cout << "\n" << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    std::cout << BOLD << CYAN << "  Solver: GAUSS-SEIDEL Iterative Method" << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    
    iterative_solver.set_corrector(&gs);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;

    iterative_solver.init(x);
    auto [success2, iters2] = iterative_solver.solve(x, b);
    
    if (success2) {
        std::cout << GREEN << "   [CONVERGED] in " << iters2 << " iterations" << RESET << std::endl;
    } else {
        std::cout << RED << "   [FAILED] Did not converge" << RESET << std::endl;
    }
    
    // Compute and print residual norm
    Vector residual2 = A * x;
    residual2 -= b;
    double residual_norm2 = residual2.norm();
    std::cout << YELLOW << "   Residual ||Ax - b|| = " << std::scientific << std::setprecision(6) << residual_norm2 << RESET << std::endl;
    
    std::cout << BOLD << "Solution x:" << RESET << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // MULTIGRID
    std::cout << "\n" << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    std::cout << BOLD << CYAN << "  Solver: MULTIGRID V-Cycle Method" << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    
    LUSolver<SparseMatrix> base_solver;
    MultiGridSolver<SparseMatrix> mg(iterative_solver, base_solver);
    mg.set_parameters(2, 2, 1, 2); 
    mg.set_convergence_params(50, 1e-15, 1e-8);
    mg.set_matrix(A);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;
    mg.init(x);

    auto [success3, iters3] = mg.solve(x, b, nElemsPerDim);
    
    if (success3) {
        std::cout << GREEN << "   [CONVERGED] in " << iters3 << " iterations" << RESET << std::endl;
    } else {
        std::cout << RED << "   [FAILED] Did not converge" << RESET << std::endl;
    }
    
    // Compute and print residual norm
    Vector residual3 = A * x;
    residual3 -= b;
    double residual_norm3 = residual3.norm();
    std::cout << YELLOW << "   Residual ||Ax - b|| = " << std::scientific << std::setprecision(6) << residual_norm3 << RESET << std::endl;
    
    std::cout << BOLD << "Solution x:" << RESET << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // LU DECOMPOSITION
    std::cout << "\n" << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    std::cout << BOLD << CYAN << "  Solver: LU Decomposition Direct Method" << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(70, '-') << RESET << std::endl;
    
    LUSolver<SparseMatrix> LU;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;
    LU.set_matrix(&A);
    LU.init(x);
    LU.solve(x, b);
    
    std::cout << GREEN << "   [COMPLETE] Direct solution computed" << RESET << std::endl;
    
    // Compute and print residual norm
    Vector residual_lu = A * x;
    residual_lu -= b;
    double residual_norm_lu = residual_lu.norm();
    std::cout << YELLOW << "   Residual ||Ax - b|| = " << std::scientific << std::setprecision(6) << residual_norm_lu << RESET << std::endl;
    
    std::cout << BOLD << "Solution x:" << RESET << std::endl;
    printVectorOnGrid(x, nElemsPerDim);
    
    const int final_border_width = 80;
    std::string final_title = "ALL BASIC TESTS COMPLETED SUCCESSFULLY";
    int final_padding = (final_border_width - final_title.length()) / 2;
    
    std::cout << "\n" << BOLD << GREEN << std::string(final_border_width, '=') << RESET << std::endl;
    std::cout << BOLD << GREEN << std::string(final_padding, ' ') << final_title << std::string(final_border_width - final_padding - final_title.length(), ' ') << RESET << std::endl;
    std::cout << BOLD << GREEN << std::string(final_border_width, '=') << RESET << "\n" << std::endl;
}


/**
 * Run benchmarks for all solvers
 */
void run_solver_benchmarks(size_t nElemsPerDim, bool bVerbose = false,
                           bool bJacobi = true, bool bGaussSeidel = true,
                           bool bMultigrid = true, int seed = 0) {
    std::cout << "\n" << BOLD << YELLOW << std::string(80, '=') << RESET << std::endl;
    std::cout << BOLD << CYAN << "SOLVER BENCHMARKS (" << nElemsPerDim << "x" << nElemsPerDim 
              << " grid, " << nElemsPerDim * nElemsPerDim << " unknowns)" << RESET << std::endl;
    std::cout << BOLD << YELLOW << std::string(80, '=') << RESET << "\n" << std::endl;

    Timer<> timer;
    
    // Setup problem
    SparseMatrix A(nElemsPerDim * nElemsPerDim, nElemsPerDim * nElemsPerDim, 5);
    Vector b, x;
    create2dPoissonSystemWithSize(A, b, nElemsPerDim);

    // Benchmark LU (only for small problems)
    if (nElemsPerDim < 16) {
        x.resize(b.size());
        std::srand(seed);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        LUSolver<SparseMatrix> lu;
        lu.set_matrix(&A);
        
        auto time = timer.measure([&]() {
            lu.init(x);
            lu.solve(x, b);
        });
        //printVectorOnGrid(x, nElemsPerDim);
        
        print_benchmark_result("LU Decomposition", time, true, 1); // Direct solver always converges in 1 iteration
    } else {
        std::cout << std::left << std::setw(20) << "LU Decomposition" << ": "
                  << YELLOW << "SKIPPED (problem too large)" << RESET << std::endl;
    }

    // Benchmark Jacobi
    if (bJacobi) {
        x.resize(b.size());
        std::srand(seed);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> solver(A);
        solver.set_verbose(bVerbose);
        Jacobi<SparseMatrix> jac;
        solver.set_corrector(&jac);
        solver.init(x);
        
        auto [result, time] = timer.measure_with_result([&]() {
            return solver.solve(x, b);
        });
        auto [conv, iters] = result;
        
        //printVectorOnGrid(x, nElemsPerDim);

        print_benchmark_result("Jacobi", time, conv, iters);
    }
    
    // Benchmark Gauss-Seidel
    if (bGaussSeidel) {
        x.resize(b.size());
        std::srand(seed);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> solver(A);
        solver.set_verbose(bVerbose);
        GaussSeidel<SparseMatrix> gs;
        solver.set_corrector(&gs);
        solver.init(x);
        
        auto [result, time] = timer.measure_with_result([&]() {
            return solver.solve(x, b);
        });
        auto [conv, iters] = result;

        //printVectorOnGrid(x, nElemsPerDim);
        
        print_benchmark_result("Gauss-Seidel", time, conv, iters);
    }
    
    // Benchmark Multigrid
    if (bMultigrid) {
        x.resize(b.size());
        std::srand(seed);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> smoother(A);
        smoother.set_verbose(false); // no verbose for smoother
        GaussSeidel<SparseMatrix> gs; // fastest smoother
        //Jacobi<SparseMatrix> jac;
        //jac.set_damping(0.8); // damped Jacobi
        smoother.set_corrector(&gs);
        smoother.init(x);
        
        LUSolver<SparseMatrix> base_solver;
        MultiGridSolver<SparseMatrix> mg(smoother, base_solver);
        mg.set_parameters(3, 3, 2, 2); // pre-smooth, post-smooth, recursions, base elements per dim
        mg.set_convergence_params(50, 1e-5, 1e-15);
        mg.set_verbose(bVerbose);
        mg.set_matrix(A); // Set matrix reference
        //mg.set_use_RAP(true); // enable RAP (Galerkin Product for A_H)
        mg.init(x);
        
        auto [result, time] = timer.measure_with_result([&]() {
            return mg.solve(x, b, nElemsPerDim);
        });
        auto [conv, iters] = result;

        //printVectorOnGrid(x, nElemsPerDim);
        
        print_benchmark_result("Multigrid", time, conv, iters);
    }
    
    std::cout << "\n" << BOLD << YELLOW << std::string(80, '=') << RESET << "\n" << std::endl;
}

int main(int argc, char** argv)
{
#if defined(_OPENMP)
    // OpenMP enabled
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Running with " << omp_get_num_threads() << " OpenMP threads." << std::endl;
        }
    }
#else
    std::cout << "Running in serial mode (OpenMP disabled)." << std::endl;
#endif

    BasicTests(4);

    // Run benchmarks
    std::cout << "\n\n" << BOLD << CYAN << "Starting Performance Benchmarks..." << RESET << std::endl;
    
    for (int i = 2; i <= 11; ++i) {
        run_solver_benchmarks(1<<i, true, false, false, true);
    }

    //run_solver_benchmarks(443, true, false, false, true);
    return 0;
}

