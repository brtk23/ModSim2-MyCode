/*
 * jacobi.cpp
 *
 *  Created on: 2019-05-03
 *      Author: 
 */

#include <iostream>
#include "headers/jacobi.h"
#include "../data structures/headers/matrix.h"
#include "../data structures/headers/sparse_matrix.h"

// Threshold for enabling OpenMP parallelization
// Only parallelize for very large independent problems, not within multigrid smoothing
static constexpr std::size_t OMP_MIN_ROWS = 1000;


template <typename TMatrix>
Jacobi<TMatrix>::Jacobi()
{
	m_damp = 1.0;
}


template <typename TMatrix>
bool Jacobi<TMatrix>::init(const vector_type& x)
{
	return true;
}


template <typename TMatrix>
bool Jacobi<TMatrix>::apply(vector_type& c, const vector_type& d) const
{
	if (!this->m_A) {
        std::cerr << "Jacobi::apply: m_A is null - did you forget to set the matrix?" << std::endl;
        return false;
    }
	const matrix_type& A = *(this->m_A);
	std::size_t n = A.num_rows();
	// Solve M * c = d <=> c = M^-1 * d where M is diag(A), so M^-1 = 1/diag(A).
	// Calculate element-wise for vector c:
	// => c[i] = damp * d[i] / A(i,i)
	// Fully parallelizable: no dependencies between iterations
	bool has_error = false;
	#pragma omp parallel for schedule(static) if(n > OMP_MIN_ROWS)
	for(std::size_t i = 0; i < n; ++i){
		// direct access to diagonal
		double diag = A(i, i);
		if(diag < 1e-15) {
			// zero diagonal entry - cannot proceed
			std::cerr << "Jacobi: Zero or almost zero diagonal entry at row " << i << std::endl;
			// Note: error handling in parallel region is tricky
			#pragma omp critical
			has_error = true;
		}
		// compute correction c[i] = damp * d[i] / diag
		c[i] = m_damp * d[i] / diag;
	}
	
	if (has_error) return false;
	return true;
}



// explicit template declarations
template class Jacobi<Matrix>;
template class Jacobi<SparseMatrix>;
