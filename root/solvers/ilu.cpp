#include "headers/ilu.h"


ILUSolver::ILUSolver()
{

}


ILUSolver::ILUSolver(const matrix_type& mat)
{
	set_matrix(&mat);
}


void ILUSolver::set_matrix(const matrix_type* A)
{
	this->m_A = A;
	m_initialized = false;
}


bool ILUSolver::init(const vector_type& x)
{
	if(this->m_A == nullptr) {
		return false;
	}
	
	const matrix_type& A = *(this->m_A);
	std::size_t n = A.num_rows();
	
	if(n != A.num_cols()) {
		return false; // Matrix must be square
	}
	
	// ILU(0): Copy original sparsity pattern from A
	// We preserve only the nonzeros that exist in A
	m_decomp = matrix_type(n, n, A.row_capacity()); 
	for(std::size_t i = 0; i < n; ++i) {
		size_t base = i * A.row_capacity();
		size_t row_len = A.get_row_len()[i];
		for(std::size_t idx = 0; idx < row_len; ++idx) {
			size_t j = A.get_col_inds()[base + idx];
			double val = A(i, j);
			if(std::abs(val) > 0.0) {  // Only copy actual nonzeros
				m_decomp(i, j) = val;
			}
		}
	}
	
	// Build column caches: for each column k, store all (row, offset) pairs
	// where m_decomp(row, k) is a non-zero
	m_col_cache.resize(n);
	for(std::size_t i = 0; i < n; ++i) {
		size_t base = i * m_decomp.row_capacity();
		size_t row_len = m_decomp.get_row_len()[i];
		for(std::size_t idx = 0; idx < row_len; ++idx) {
			size_t j = m_decomp.get_col_inds()[base + idx];
			// Store row in the cache for column j
			m_col_cache[j].entries.push_back(i);
		}
	}
	
	// ILU(0) factorization: perform elimination but drop any fill
	// that would occur outside the original sparsity pattern
	for(std::size_t k = 0; k < n; ++k) {
		double pivot = m_decomp(k, k);
		if(std::abs(pivot) < 1e-15) {
			return false; // Pivot too small
		}
		
		// For each row i that has a non-zero at (i,k), use the column cache
		for(const auto& i : m_col_cache[k].entries) {
			if(i <= k) continue;  // Only process rows below pivot
			
			// Compute multiplier and store in L part
            // (how much of row k to subtract from row i to eliminate)
			double l_ik = m_decomp(i, k) / pivot;
			m_decomp(i, k) = l_ik;
			
			// Update row i in U: subtract l_ik * row_k from row_i
			// Iterate through non-zeros in row k that are to the right of diagonal
			size_t k_base = k * m_decomp.row_capacity();
			size_t k_row_len = m_decomp.get_row_len()[k];
			for(std::size_t k_idx = 0; k_idx < k_row_len; ++k_idx) {
				size_t j = m_decomp.get_col_inds()[k_base + k_idx];
				if(j <= k) continue;  // Only U part (j > k)
				
				// Check if (i, j) exists (can't rule out using iterators)
				if(!m_decomp.has_entry(i, j)) {
					continue;  // Skip fill-ins
				}
				
				double u_kj = m_decomp.get_values()[k_base + k_idx];
				double& current_val = m_decomp(i, j);
				current_val = current_val - l_ik * u_kj;
			}
		}
	}
	
	m_initialized = true;
	return true;
}


bool ILUSolver::apply(vector_type& c, const vector_type& d) const
{

	if(!m_initialized) {
		std::cerr << "ILUSolver::apply: Solver not initialized - did you forget to call init()?" << std::endl;
		return false;
	}
	
	std::size_t n = m_decomp.num_rows();
	
	if(c.size() != n || d.size() != n) {
		return false;
	}
	
	// Forward substitution: solve L*v = b
	vector_type v(n);
	for(std::size_t i = 0; i < n; ++i) {
		v[i] = d[i];
		// Only subtract if L(i,j) exists
		size_t base = i * m_decomp.row_capacity();
		size_t row_len = m_decomp.get_row_len()[i];
		for(std::size_t idx = 0; idx < row_len; ++idx) { 
			size_t j = m_decomp.get_col_inds()[base + idx];
			if(j >= i) {
				continue; // Only L part (j < i)
			}
			v[i] -= m_decomp(i, j) * v[j];
		}
	}
	
	// Backward substitution: solve U*x = v
	for(std::size_t i = n; i-- > 0; ) {
		c[i] = v[i];
		// Only subtract if U(i,j) exists
		size_t base = i * m_decomp.row_capacity();
		size_t row_len = m_decomp.get_row_len()[i];
		for(std::size_t idx = 0; idx < row_len; ++idx) {
			size_t j = m_decomp.get_col_inds()[base + idx];
			if(j <= i) {
				continue; // Only U part (j > i)
			}
			c[i] -= m_decomp(i, j) * c[j];
		}
		c[i] /= m_decomp(i, i); 
	}
	
	return true;
}
