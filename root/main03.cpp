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
    numElemsPerDim++;
    const std::size_t sz = numElemsPerDim - 1;
    const double h_minus2 = numElemsPerDim * numElemsPerDim;
    const double h2_2 = 1.0 / (2*numElemsPerDim*numElemsPerDim);

    // square matrix
    m.resize(sz*sz, sz*sz);
    rhs.resize(sz*sz, 2.0);

    // fill matrix and rhs
    for (std::size_t r = 0; r < sz; ++r)
    {
        m(r*sz, r*sz) = 4.0 * h_minus2;
        for (std::size_t c = 1; c < sz; ++c)
        {   
            m(r*sz+c, r*sz+c) = 4.0 * h_minus2;
            m(r*sz+c, r*sz+c-1) = -h_minus2;
            m(r*sz+c-1, r*sz+c) = -h_minus2;
            m(c*sz+r, (c-1)*sz+r) = -h_minus2;
            m((c-1)*sz+r, c*sz+r) = -h_minus2;
        }

    }
    //create rhs by first writing entries in a matrix.. its easier that was..
    Matrix dummy(sz,sz,2.0);
    Vector test(sz*sz, 2.0);
    double dxy = 1.0/((double)numElemsPerDim - 2);
    for(int i = 0; i < sz; ++i){
        for(int j = 0; j < sz; ++j){
             if(i == 0 || j == 0 || i == sz-1 || j == sz -1){
                dummy(i,j) = (1 - 0.5*(i*dxy)*(i*dxy) - 0.5*(j*dxy)*(j*dxy));//* h_minus2;
            }

        }
    }
    Vector rhs_new(sz*sz, 0);
    size_t vector_index = 0;
    // now port the entries into rhs vector without iterators
    for(size_t i = 0; i < sz; ++i){
        for(size_t j = 0; j < sz; ++j){
            rhs[vector_index] = dummy(i, j);
            // if we have dirichlet boundary condition, substitute matrix
            // row with identity row
            if(rhs[vector_index] != 2.0){
                // Must clear the row first to remove old Poisson stencil entries!
                m.clear_row(vector_index);
                // Now set only the diagonal entry
                m(vector_index, vector_index) = 1.0;
            }
            ++vector_index;
        }
    }

}
/** this function prints the solution vector or rhs vector in
 *  form of the underlying grid!
		**/
void printVectorOnGrid(Vector& u, size_t nElemsPerDim){
    for(size_t i = 0; i < u.size(); ++i){

        if(i % (nElemsPerDim ) == 0){
            std::cout << std::endl;
        }
        std::cout << std::setw(6) << std::setfill(' ') << std::setprecision(2) << std::fixed << u[i] << "\t";

    }
    std::cout << std::endl;
}
///basic LU decomposition and solving a system of linear equations
bool LU_test(size_t nElem){
    std::cout << "BEGIN LU TESTING..." << std::endl;
    Vector v(nElem, 1.0);
    Vector x(nElem,0.0);
    SparseMatrix S(nElem, nElem, 5);
    Matrix M(nElem, nElem, 0.0);
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
    lu_sparse.init(v);
    std::cout << "Initialized sparse LU solver." << std::endl;
    lu_full.init(v);
    std::cout << "Initialized full LU solver." << std::endl;

    std::cout << "decomposition sparse:" << std::endl;
    lu_sparse.print();
    lu_sparse.solve(x,v);
    std::cout << "solution sparse: " << std::endl;
    std::cout << x << std::endl;
    std::cout << "decomposition full:" << std::endl;
    lu_full.print();
    lu_full.solve(x, v);
    std::cout << "solution full: " << std::endl;
    std::cout << x << std::endl;
    std::cout << "LU TEST COMPLETE :)" << std::endl;
    return true;
}

void BasicTests(size_t nElemsPerDim = 3)
{
    // test for LU implementation from last exercise sheet..
    // please make sure this passes and if not, consider your
    // implementation again!
    LU_test(5);

    // example for solving of poisson problem
    SparseMatrix A(nElemsPerDim * nElemsPerDim, nElemsPerDim * nElemsPerDim, 5);
    Vector b, x;

    create2dPoissonSystemWithSize(A, b, nElemsPerDim);

    std::cout << "Vector b [from Ax = b] looks like this:" << std::endl;
    printVectorOnGrid(b, nElemsPerDim);
    std::cout << "\nAssembled Matrix A [from Ax = b] looks like this:" << std::endl;
    std::cout << A << std::endl;
    x.resize(b.size());

    // create random starting values
    std::srand(0);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;

    // create iterative solver and set parameters
    IterativeSolver<SparseMatrix> iterative_solver(A);
    iterative_solver.set_convergence_params(500000, 1e-15, 1e-8);

    // with set_verbose we should get a print every ~ k iterations
    // with information about current reduction rate etc.
    iterative_solver.set_verbose(false);

    // create preconditioners:
    Jacobi<SparseMatrix> jac;
    GaussSeidel<SparseMatrix> gs;

    // now set jacobi as preconditioner of iterative solver:
    iterative_solver.set_corrector(&jac);
    iterative_solver.init(x);
    // start iteration scheme
    auto [success, iters1] = iterative_solver.solve(x, b);

    std::cout << "Jacobi finished with solution x = " << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // same for gauss seidel:
    iterative_solver.set_corrector(&gs);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;

    iterative_solver.init(x);
    auto [success2, iters2] = iterative_solver.solve(x, b);
    std::cout << "Gauss-Seidel finished with solution x = " << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // now multigrid with gauss seidel as smoother
    LUSolver<SparseMatrix> base_solver;
    MultiGridSolver<SparseMatrix> mg(iterative_solver, base_solver);
    mg.set_parameters(2, 2, 1, 2); 
    mg.set_convergence_params(50, 1e-15, 1e-8);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;

    mg.solve(A, x, b, nElemsPerDim);
    std::cout << "Multigrid finished with solution x =  " << std::endl;
    printVectorOnGrid(x, nElemsPerDim);

    // do it with LU now
    LUSolver<SparseMatrix> LU;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::rand() / (double)RAND_MAX;
    LU.set_matrix(&A);
    LU.init(x);
    LU.solve(x, b);
    std::cout << "\nLU decomposition finished with solution x = " << std::endl;
    printVectorOnGrid(x, nElemsPerDim);
}


/**
 * Run benchmarks for all solvers
 */
void run_solver_benchmarks(size_t nElemsPerDim, bool bVerbose = false,
                           bool bJacobi = true, bool bGaussSeidel = true,
                           bool bMultigrid = true) {
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
    if (nElemsPerDim < 20) {
        x.resize(b.size());
        std::srand(0);
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
        std::srand(0);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> solver(A);
        solver.set_convergence_params(50000, 1e-15, 1e-8);
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
        std::srand(0);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> solver(A);
        solver.set_convergence_params(50000, 1e-15, 1e-8);
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
        std::srand(0);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::rand() / (double) RAND_MAX;
        
        IterativeSolver<SparseMatrix> smoother(A);
        smoother.set_convergence_params(50000, 1e-15, 1e-8);
        smoother.set_verbose(false); // no verbose for smoother
        GaussSeidel<SparseMatrix> gs; // fastest smoother
        //Jacobi<SparseMatrix> jac;
        smoother.set_corrector(&gs);
        smoother.init(x);
        
        LUSolver<SparseMatrix> base_solver;
        MultiGridSolver<SparseMatrix> mg(smoother, base_solver);
        mg.set_parameters(4, 4, 2, 2);
        mg.set_convergence_params(5000, 1e-15, 1e-8);
        mg.set_verbose(bVerbose);
        
        auto [result, time] = timer.measure_with_result([&]() {
            return mg.solve(A, x, b, nElemsPerDim);
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

    //BasicTests(4);

    // Run benchmarks
    std::cout << "\n\n" << BOLD << CYAN << "Starting Performance Benchmarks..." << RESET << std::endl;
    
    for (int i = 2; i <= 10; ++i) {
        run_solver_benchmarks(1<<i, false, false, false, true);
    }
    // 1024x1024 grid
    // without saving hierarchy and without openmp:
    // multigrid: 13 sec (15 iter) 
    // with hierarchy and without openmp:
    // multigrid: 11 sec (15 iter)
    // with hierarchy and with openmp:
    // multigrid: ? sec (15 iter)

    return 0;
}

