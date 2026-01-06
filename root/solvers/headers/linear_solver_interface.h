/*
 * linear_solver_interface.h
 *
 *  Created on: 2017-05-22
 *      Author: avogel
 */

#ifndef LINEAR_SOLVER_INTERFACE_H
#define LINEAR_SOLVER_INTERFACE_H

#include "../../data structures/headers/vector.h"
#include "../../data structures/headers/matrix.h"

/** @brief Interface class for linear solver
 * Each class (LU, CG, LinearIteration, ...) implementing this interface has to
 * provide the method solve() which will compute the solution.
 */
template <typename TMatrix>
class ILinearSolver
{
	typedef Vector vector_type;
	typedef TMatrix matrix_type;

	public:
		/// constructor
		ILinearSolver() : m_pA(NULL) {};

		/// destructor
		virtual ~ILinearSolver() {};

		/// set matrix
		virtual void set_matrix(const matrix_type* mat) = 0;

		/// init with current solution (IDK WHAT THIS DOES?) maybe for optimization?
		// e.g. precompute stuff
		// but for now I just ignore it...
		virtual bool init(const vector_type& x) = 0;

		/** @brief Solves a linear system
		 * Solve A*x = b for x.
		 *
		 * @param x  output: solution
		 * @param b  input: right-hand side
		 * @return   false on any failure; true otherwise
		 */
		virtual bool solve(vector_type& x, const vector_type& b) const = 0;

	protected:
		/// underlying matrix
		const matrix_type* m_pA;
};

#endif // LINEAR_SOLVER_INTERFACE_H
