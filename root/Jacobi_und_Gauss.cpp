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
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

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

    //DEBUG
    std::cout << "Matrix and RHS vector initialized." << std::endl;

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
    //DEBUG
    std::cout << "Matrix filled with stencil values." << std::endl;
    
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
                for(std::size_t current_col = 0; current_col < m.num_cols(); ++current_col){
                    m(vector_index, current_col) = (current_col == vector_index) ? 1.0 : 0.0;
                }
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


int main(int argc, char** argv)
{
    size_t nElemsPerDim = 4;
    SparseMatrix A;
    Vector b, u;

    create2dPoissonSystemWithSize(A, b, nElemsPerDim);

    std::cout << "RHS vector looks like this:" << std::endl;
    printVectorOnGrid(b, nElemsPerDim);
    std::cout << "\nassembled SparseMatrix A looks like this:" << std::endl;
    std::cout << A << std::endl;
    u.resize(b.size());


    // create random starting values
    std::srand(0);
    for (std::size_t i = 0;  i < u.size(); ++i)
        u[i] = std::rand() / (double) RAND_MAX;

    // create iterative solver and set parameters
    IterativeSolver<SparseMatrix> iterative_solver(A);
    iterative_solver.set_convergence_params(500000, 1e-15, 1e-8);
    
    // with set_verbose we should get a print every ~ k iterations
    // with information about current reduction rate etc.
    iterative_solver.set_verbose(true);

    
    // create preconditioners:
    Jacobi<SparseMatrix> jac;
    GaussSeidel<SparseMatrix> gs;

    

    // now set jacobi as preconditioner of iterative solver:
    iterative_solver.set_corrector(&jac);
    iterative_solver.init(u);
    // start iteration scheme
    bool success = iterative_solver.solve(u, b);

    std::cout << "Jacobi finished with soultion: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    // same for gauss seidel:
    iterative_solver.set_corrector(&gs);
    for (std::size_t i = 0;  i < u.size(); ++i)
        u[i] = std::rand() / (double) RAND_MAX;

    iterative_solver.init(u);
    success = iterative_solver.solve(u,b);
    std::cout << "Gauss-Seidel finished with soultion: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    return 0;
}

