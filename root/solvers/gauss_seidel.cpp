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
	// Update c using Gauss-Seidel method (sequential due to data dependencies)
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
	// Sequential loop (cannot parallelize due to dependencies)
	for(std::size_t i = 0; i < n; ++i){
		double sigma = 0.0;
		// For SparseMatrix: iterate only stored elements in row i
		if constexpr(std::is_same<TMatrix, SparseMatrix>::value) {
			const auto& values = A.get_values();
			const auto& col_inds = A.get_col_inds();
			const auto& row_len = A.get_row_len();
			std::size_t base = i * A.row_capacity();
			std::size_t len = row_len[i];
			// Use SIMD for reduction within single row (safe: each j only touches c[j] already computed)
			#pragma omp simd reduction(+:sigma)
			for(std::size_t k = 0; k < len; ++k) {
				std::size_t j = col_inds[base + k];
				if(j < i) {
					sigma += values[base + k] * c[j];
				}
			}
		} else {
			// For dense matrices: iterate all j < i
			for(std::size_t j = 0; j < i; ++j){
				sigma += A(i, j) * c[j];
			}
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

