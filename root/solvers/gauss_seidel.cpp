/*
 * gauss_seidel.cpp
 *
 *  Created on: 2019-05-03
 *      Author: 
 */

#include "headers/gauss_seidel.h"
#include "../data structures/headers/matrix.h"
#include "../data structures/headers/sparse_matrix.h"


template <typename TMatrix>
GaussSeidel<TMatrix>::GaussSeidel()
{
}


template <typename TMatrix>
bool GaussSeidel<TMatrix>::init(const vector_type& x)
{
	return true;
}


template <typename TMatrix>
bool GaussSeidel<TMatrix>::apply(vector_type& c, const vector_type& d) const {
	// Update c using Gauss-Seidel method
	if(this->m_A == nullptr){
		return false; // matrix not set
	}

	const TMatrix& A = *(this->m_A);
	std::size_t n = A.num_rows();

	if(c.size() != n || d.size() != n){
		return false; // size mismatch
	}

	// Solve M * c = d for c where M is the lower triangular part of A
	// Calculate element-wise for vector c:
	// d[i] = c[i] * A(i,i) + sum_{j<i} c[j] * A(i,j)
	// => c[i] = (d[i] - sum_{j<i} c[j] * A(i,j)) / A(i,i)
	for(std::size_t i = 0; i < n; ++i){
		double sigma = 0.0;
		// accumulate lower triangular part using direct indexing
		for(std::size_t j = 0; j < i; ++j){
			sigma += A(i, j) * c[j];
		}
		double diag = A(i, i);
		if(diag < 1e-15){
			return false; // singular matrix
		}
		c[i] = (d[i] - sigma) / diag;
	}

	return true;
}


// explicit template declarations
template class GaussSeidel<Matrix>;
template class GaussSeidel<SparseMatrix>;

