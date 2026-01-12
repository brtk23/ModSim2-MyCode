/*
 * lu_solver.cpp
 *
 *  Created on: 2019-05-03
 *      Author: 
 */
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include "headers/lu_solver.h"
#include "../data structures/headers/matrix.h"
#include "../data structures/headers/sparse_matrix.h"


template <typename TMatrix>
LUSolver<TMatrix>::LUSolver()
{

}


template <typename TMatrix>
LUSolver<TMatrix>::LUSolver(const matrix_type& mat)
{
	set_matrix(&mat);
}


template <typename TMatrix>
void LUSolver<TMatrix>::set_matrix(const matrix_type* A)
{
	this->m_pA = A;
	m_initialized = false;
}


template <typename TMatrix>
bool LUSolver<TMatrix>::init(const vector_type& x)
{
	if(this->m_pA == nullptr) {
		return false;
	}
	
	const matrix_type& A = *(this->m_pA);
	std::size_t n = A.num_rows();
	
	if(n != A.num_cols()) {
		return false; // Matrix must be square
	}
	
	if (std::is_same<TMatrix, SparseMatrix>::value) {
		// If matrix is sparse: create with worst-case capacity and copy non-zeros via indexed access
		m_decomp = TMatrix(n, n, n);
		for(std::size_t i = 0; i < n; ++i) {
			for(std::size_t j = 0; j < n; ++j) {
				double val = A(i, j);
				if(val != 0.0) {
					m_decomp(i, j) = val;
				}
			}
		}
	} else {
		// Dense matrix: direct copy via indexed access
		m_decomp = TMatrix(n, n);
		for(std::size_t i = 0; i < n; ++i) {
			for(std::size_t j = 0; j < n; ++j) {
				m_decomp(i, j) = A(i, j);
			}
		}
	}
	
	// Perform LU decomposition without pivoting
	for(std::size_t k = 0; k < n; ++k) {
		if(std::abs(m_decomp(k, k)) < 1e-15) {
			return false; // Pivot too small
		}
		
		for(std::size_t i = k + 1; i < n; ++i) {
			m_decomp(i, k) /= m_decomp(k, k);
			for(std::size_t j = k + 1; j < n; ++j) {
				m_decomp(i, j) -= m_decomp(i, k) * m_decomp(k, j);
			}
		}
	}
	
	m_initialized = true;
	return true;
}


template <typename TMatrix>
std::tuple<bool, size_t> LUSolver<TMatrix>::solve(vector_type& x, const vector_type& b) const
{

	if(!m_initialized) {
		std::cerr << "LUSolver::solve: Solver not initialized - did you forget to call init()?" << std::endl;
		return {false, 0};
	}
	
	std::size_t n = m_decomp.num_rows();
	
	if(x.size() != n || b.size() != n) {
		return {false, 0};
	}
	
	// Forward substitution: solve L*v = b
	vector_type v(n);
	for(std::size_t i = 0; i < n; ++i) {
		v[i] = b[i];
		for(std::size_t j = 0; j < i; ++j) {
			v[i] -= m_decomp(i, j) * v[j];
		}
	}
	
	// Backward substitution: solve U*x = v
	for(std::size_t i = n; i-- > 0; ) {
		x[i] = v[i];
		for(std::size_t j = i + 1; j < n; ++j) {
			x[i] -= m_decomp(i, j) * x[j];
		}
		x[i] /= m_decomp(i, i);
	}
	
	return {true, 1}; // Direct solver always completes in 1 "iteration"
}



// explicit template declarations
template class LUSolver<Matrix>;
template class LUSolver<SparseMatrix>;

