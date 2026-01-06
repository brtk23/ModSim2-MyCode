/*
 * lu_solver.h
 *
 *  Created on: 2019-05-03
 *      Author: mbreit
 */

#ifndef LU_SOLVER_H
#define LU_SOLVER_H

#include "../../data structures/headers/vector.h"
#include "linear_solver_interface.h"
#include <iostream>

template <typename TMatrix>
class LUSolver
 : public ILinearSolver<TMatrix>
{
	public:
		typedef Vector vector_type;
		typedef TMatrix matrix_type;

	public:
		// constructor
		LUSolver();

		/// constructor with matrix
		LUSolver(const matrix_type& mat);

		/** @brief Perform LU decomposition
		 * This method performs the LU decomposition of the underlying matrix.
		 * The decomposed matrix is held as a member in an object of this class.
		 * This method has to be executed once prior to any call to solve().
		 *
		 * @return  true iff decomposition succeeded
		 */
		virtual bool init(const vector_type& x);

		/// set matrix
		virtual void set_matrix(const matrix_type* mat);

		/** @brief Solve a linear system using the LU decomposition
		 * This method will use the LU decomposition computed by init() to solve the
		 * system A*x = b,
		 * where A is the underlying matrix passed in the constructor.
		 * It first solves L*v = b and then U*x = v. As both L and U are triangular matrices,
		 * this can be done efficiently.
		 *
		 * @param x  output: solution
		 * @param b  input: right-hand side
		 * @return   false on any failure; true otherwise
		 */
		bool solve(vector_type& x, const vector_type& b) const;

        void print(){
            std::cout << m_decomp << std::endl;
        }

    private:
        TMatrix m_decomp;
        bool m_initialized = false;
};

#endif // LU_SOLVER_H
