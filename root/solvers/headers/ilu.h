/*
 * lu_solver.h
 *
 *  Created on: 2019-05-03
 *      Author: mbreit
 */

#ifndef ILU_SOLVER_H
#define ILU_SOLVER_H

#include "../../data structures/headers/vector.h"
#include "../../data structures/headers/sparse_matrix.h"
#include "preconditioner_interface.h"

#include <iostream>
#include <vector>


class ILUSolver
 : public IPreconditioner<SparseMatrix>
{
	public:
		typedef Vector vector_type;
		typedef SparseMatrix matrix_type;

	private:
		// Cache for column-wise access: for each column, store which rows have non-zeros there
		struct ColumnCache {
			std::vector<size_t> entries;
		};
		std::vector<ColumnCache> m_col_cache;

	public:
		// constructor
		ILUSolver();

		// constructor with matrix
		ILUSolver(const matrix_type& mat);

		/** @brief Perform LU decomposition
		 * This method performs the LU decomposition of the underlying matrix.
		 * The decomposed matrix is held as a member in an object of this class.
		 * This method has to be executed once prior to any call to solve().
		 *
		 * @return  true if decomposition succeeded
		 */
		virtual bool init(const vector_type& x);

		// set matrix
		virtual void set_matrix(const matrix_type* mat);

		/** @brief Apply the ILU preconditioner to a defect
         * This method will use the LU decomposition computed by init() to solve the
         * system M*c = d,
		 * where M is the approximate underlying matrix (ignore epsilon)
         * passed in the constructor.
		 * It first solves L*v = d and then U*c = v.
         *
         * @param c  output: correction
         * @param d  input: defect
         * @return   false on any failure; true otherwise
         * 
        */
		virtual bool apply(vector_type& c, const vector_type& d) const;

        void print(){
            std::cout << m_decomp << std::endl;
        }

    private:
        matrix_type m_decomp;
        bool m_initialized = false;
        double m_tolerance = 1e-8; // Tolerance for ignoring small values
};

#endif // LU_SOLVER_H
