/*
 * preconditioner_interface.h
 *
 *  Created on: 2019-05-03
 *      Author: mbreit
 */

#ifndef PRECONDITIONER_INTERFACE_H
#define PRECONDITIONER_INTERFACE_H

#include "../../data structures/headers/vector.h"
#include "../../data structures/headers/matrix.h"


/** @brief Interface class for iterative corrector methods.
 * Each class (Jacobi, Gauss-Seidel etc.) implementing this interface has to
 * provide the method apply() which will perform one step of the method
 * resulting in a correction vector that can be used, e.g., in an IterativeSolver.
 */
template <typename TMatrix>
class IPreconditioner
{
	typedef Vector vector_type;
	typedef TMatrix matrix_type;

	public:
		/// constructor
		IPreconditioner() : m_A(NULL) {};

		/// destructor
		virtual ~IPreconditioner() {};

		/**
		 * @brief Init with current solution
		 * This method will assemble level matrices and rhs's
		 * as well as init the base solver (LU, typically).
		 *
		 * @param x the solution to use (not needed in linear case)
		 */
		virtual bool init(const vector_type& x) = 0;

		/// set matrix
		virtual void set_matrix(const matrix_type* mat) {m_A = mat;}

		/** @brief Apply preconditioner
		 * This method applies the preconditioner and has to be provided by all classes
		 * that implement this interface. It is given a defect d and computes
		 * a correction c = M^(-1) * d
		 * with M being an approximation of the underlying matrix A.
		 *
		 * @param c  computed correction vector
		 * @param d  input defect
		 * @return   false on any failure; true otherwise
		 */
		virtual bool apply(vector_type& c, const vector_type& d) const = 0;

	protected:
		/// underlying matrix
		const matrix_type* m_A;
};



#endif // PRECONDITIONER_INTERFACE_H
