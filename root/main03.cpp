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

// void TestSparseMatrixMatMulMinus() {
//     // Test SparseMatrix::operator* against SparseMatrix::MatMulMinus
//     std::cout << "\nTesting SparseMatrix::MatMulMinus against operator* ..." << std::endl;
//     const int iterations = 10000;
//     SparseMatrix A = SparseMatrix(1000, 1000, 10);
//     Vector x = Vector(1000, 1.0);
//     Vector b = Vector(1000, 2.0);
//     Vector d = Vector(1000, 0.0);
//     // Fill A with some values
//     for (std::size_t i = 0; i < 1000; ++i) {
//         A(i, i) = 4.0;
//         if (i > 0) {
//             A(i, i - 1) = -1.0;
//             A(i - 1, i) = -1.0;
//         }
//         if (i < 999) {
//             A(i, i + 1) = -1.0;
//             A(i + 1, i) = -1.0;
//         }
//     }
//     // Print A, x, b
//     // std::cout << "Matrix A:\n" << A << std::endl;
//     // std::cout << "Vector x:\n" << x << std::endl;
//     // std::cout << "Vector b:\n" << b << std::endl;

//     // Test standard operator*
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < iterations; ++i) {
//         d = b - A * x;
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_standard = end - start;
//     std::cout << "Standard operator* time: " << elapsed_standard.count() << 
//                 " seconds\n";

//     // Test MatMulMinus
//     start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < iterations; ++i) {
//         A.MatMulMinus(A, b, x, d);
//     }
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_matmulminus = end - start;
//     std::cout << "MatMulMinus time: " << elapsed_matmulminus.count()
//                 << " seconds\n";

//     // Speedup
//     double speedup = elapsed_standard.count() / elapsed_matmulminus.count();
//     std::cout << "Speedup: " << speedup << "x\n";
// }


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
    //now port the entries into rhs vector
    for(size_t i = 0; i < sz; ++i){

        Matrix::RowIterator it = dummy.begin(i);
        Matrix::RowIterator end= dummy.end(i);
        for(; it != end; ++it){
            rhs[vector_index] = it.value() ;
            //if we have dirichlet boundary condition, substitute matrix
            //row with identity row..
            if(rhs[vector_index] != 2.0){
                typename TMatrix::RowIterator Ait = m.begin(vector_index);
                typename TMatrix::RowIterator Aend= m.end(vector_index);
                for(; Ait != Aend; ++Ait){
                    size_t current_col = Ait.col_index();
                    if(current_col == vector_index){
                        m(vector_index, current_col) = 1.0;
                    }
                    else{
                        m(vector_index, current_col) = 0;
                    }

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
int main(int argc, char** argv)
{


    // test for LU implementation from last exercise sheet..
    // please make sure this passes and if not, consider your
    // implementation again!
    LU_test(5);

    // example for solving of poisson problem
    size_t nElemsPerDim = 3;
    SparseMatrix A;
    Vector b, u;

    create2dPoissonSystemWithSize(A, b, nElemsPerDim);

    std::cout << "RHS vector looks like this:" << std::endl;
    printVectorOnGrid(b, nElemsPerDim);
    std::cout << "\nassembled Matrix A looks like this:" << std::endl;
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

    std::cout << "Jacobi finished with solution: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    // same for gauss seidel:
    iterative_solver.set_corrector(&gs);
    for (std::size_t i = 0;  i < u.size(); ++i)
        u[i] = std::rand() / (double) RAND_MAX;

    iterative_solver.init(u);
    success = iterative_solver.solve(u,b);
    std::cout << "Gauss-Seidel finished with solution: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    // now multigrid with gauss seidel as smoother
    iterative_solver.set_verbose(false); // no need to print in smoother
    LUSolver<SparseMatrix> base_solver;
    MultiGridSolver<SparseMatrix> mg(iterative_solver, base_solver, 2, 2, 1, 2);
    for (std::size_t i = 0;  i < u.size(); ++i)
        u[i] = std::rand() / (double) RAND_MAX;
    
    mg.solve(A, u, b, nElemsPerDim);
    std::cout << "Multigrid finished with solution: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    // do it with LU now
    Matrix M;
    LUSolver<SparseMatrix> LU;

    create2dPoissonSystemWithSize(M, b, nElemsPerDim);

    LU.set_matrix(&A);
    LU.init(u);
    LU.solve(u, b);
    std::cout << "\nLU decomposition finished with solution: " << std::endl;
    printVectorOnGrid(u, nElemsPerDim);

    
    return 0;
}

