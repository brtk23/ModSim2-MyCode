/*
 * Iterative_solver.cpp
 *
 *  Created on: 2019-05-03
 *      Author: 
 */

#include <iostream>
#include <iomanip>
#include "headers/iterative_solver.h"
#include "../data structures/headers/matrix.h"
#include "../data structures/headers/sparse_matrix.h"



template <typename TMatrix>
IterativeSolver<TMatrix>::IterativeSolver(const matrix_type& mat)
{
	this->m_pA = &mat;
	m_corrector = nullptr;
	m_nit = 5000;
	m_minDef = 1e-15;
	m_minRed = 1e-8;
	m_bVerbose = false;
	m_bInited = false;
}


template <typename TMatrix>
IterativeSolver<TMatrix>::IterativeSolver()
{
	m_corrector = nullptr;
	m_nit = 5000;
	m_minDef = 1e-15;
	m_minRed = 1e-8;
	m_bVerbose = false;
	m_bInited = false;
}


template <typename TMatrix>
void IterativeSolver<TMatrix>::set_corrector(corrector_type* stepMethod)
{
	m_corrector = stepMethod;
	if(m_corrector != nullptr && this->m_pA != nullptr){
		m_corrector->set_matrix(this->m_pA);
	}
}


template <typename TMatrix>
bool IterativeSolver<TMatrix>::init(const vector_type& x)
{
	if(m_corrector != nullptr){
		if(!m_corrector->init(x)){
			return false;
		}
	}
	m_bInited = true;
	return true;
}


template <typename TMatrix>
void IterativeSolver<TMatrix>::set_matrix(const matrix_type* A)
{
	this->m_pA = A;
	if(m_corrector != nullptr){
		m_corrector->set_matrix(A);
	}
}


template <typename TMatrix>
void IterativeSolver<TMatrix>::set_convergence_params
(
	std::size_t nIter,
	double minDef,
	double minRed
)
{
	m_nit = nIter;
	m_minDef = minDef;
	m_minRed = minRed;
}


template <typename TMatrix>
std::tuple<bool, size_t> IterativeSolver<TMatrix>::solve(vector_type& x, const vector_type& b) const
{
	if(!m_bInited){
		return {false, 0};
	}

	if(this->m_pA == nullptr){
		return {false, 0};
	}

	vector_type d(b.size());
	vector_type c(b.size());
	double norm_di = 0.0;
	double norm_di_prev = 0.0;
	double norm_d0 = 0.0;
	double rate = 0.0;
	double reduction = 0.0;

	if(m_bVerbose) {
		std::cout << "## Iterative solver #############################################################" << std::endl;
		std::cout << "Iter\tDefect\t\tRequired\tRate\t\tReduction\tRequired" << std::endl;
	}

	std::size_t iter = 0;
	do {
		// compute defect d_i = b - A*x_i (already parallelized in Vector::operator-)
		d = b - (*(this->m_pA)) * (x); // TODO MatMulMinus
		if(iter == 0){
			norm_d0 = d.norm();
		}
		norm_di_prev = norm_di;
		norm_di = d.norm();

		// compute correction c_i = M^(-1) * d_i
		if(m_corrector != nullptr){
			bool bResult = m_corrector->apply(c, d);
			if(!bResult){
				std::cerr << "IterativeSolver: Corrector application failed!" << std::endl;
				return {false, iter};
			}
		} else {
			std::cerr << "IterativeSolver: No corrector method set!" << std::endl;
			return {false, iter};
		}

		// update solution x_i+1 = x_i + c_i (parallelized with SIMD)
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < x.size(); ++i) {
			x[i] += c[i];
		}

		if(m_bVerbose) {
			if(iter > 0) rate = norm_di / norm_di_prev;
			if(iter > 0) reduction = norm_di / norm_d0;
			std::cout << iter << "\t" << std::scientific << std::setprecision(6) 
					  << norm_di << "\t" << std::setprecision(3) << m_minDef << "\t";
			if(iter==0) std::cout << "----------"; else std::cout << std::setprecision(3) << rate;
			std::cout << "\t";
			if(iter==0) std::cout << "-------------"; else std::cout << std::setprecision(6) << reduction;
			std::cout << "\t" << std::setprecision(3) << m_minRed << std::endl;
		}

		++iter;
	} while((norm_di > m_minDef) && (norm_di > m_minRed * norm_d0) && (iter < m_nit));

	if(norm_di <= m_minDef || norm_di <= m_minRed * norm_d0){
		if (m_bVerbose) {
			std::cout << "IterativeSolver: Converged in " << iter << " iterations." << std::endl;
		}
		return {true, iter};
	} else {
		if (m_bVerbose) {
			std::cout << "IterativeSolver: Not converged after " << iter << " iterations." << std::endl;
		}
		return {false, iter};
	}
}



// explicit template declarations
template class IterativeSolver<Matrix>;
template class IterativeSolver<SparseMatrix>;

