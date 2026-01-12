/*
 * sparse_matrix.cpp
 *
 *  Created on: 2019-04-28
 *      Author: 
 *  Redesigned for SIMD optimization with OpenMP
 */

#include "headers/sparse_matrix.h"
#include "omp.h"
#include <iostream>
#include <iomanip>

// Threshold for enabling OpenMP parallelization
// Below this size, overhead of thread creation outweighs benefit
// Increased significantly for multigrid: only parallelize at coarsest level or for standalone large ops
static constexpr std::size_t OMP_MIN_ROWS = 10000;
static constexpr std::size_t OMP_MIN_SIZE = 10000000;  // For matrix-matrix operations

SparseMatrix::SparseMatrix()
	: SparseMatrix(0, 0)
{
}


SparseMatrix::SparseMatrix(std::size_t r, std::size_t c, std::size_t rowCapacity)
{
    this->m_rows = r;
    this->m_cols = c;
    this->m_row_capacity = rowCapacity;

    this->m_values.resize(r * rowCapacity, this->m_zero);
    this->m_col_inds.resize(r * rowCapacity, 0);
    this->m_row_len.resize(r, 0);
}


SparseMatrix::~SparseMatrix()
{}


void SparseMatrix::resize(std::size_t r, std::size_t c, double defVal)
{
    this->m_rows = r;
    this->m_cols = c;

    this->m_values.resize(r * this->m_row_capacity, this->m_zero);
    this->m_col_inds.resize(r * this->m_row_capacity, 0);
    this->m_row_len.resize(r, 0);
}

bool SparseMatrix::has_entry(std::size_t r, std::size_t c) const
{
    size_t base = r * m_row_capacity;
    size_t len = m_row_len[r];
    for(size_t i = 0; i < len; ++i) {
        if(m_col_inds[base + i] == c) {
            return true;
        }
    }
    return false;
}

double SparseMatrix::operator()(std::size_t r, std::size_t c) const
{
    size_t base = r * m_row_capacity;
    size_t len = m_row_len[r];
    for(size_t i = 0; i < len; ++i) {
        if(m_col_inds[base + i] == c) {
            return m_values[base + i];
        }
    }
    return 0.0;
}

double& SparseMatrix::operator()(std::size_t r, std::size_t c)
{
    if (r >= m_rows || c >= m_cols) {
        throw std::out_of_range("SparseMatrix::operator(): Index out of range.");
    }
    size_t base = r * m_row_capacity;
    size_t len = m_row_len[r];
    for(size_t i = 0; i < len; ++i) {
        if(m_col_inds[base + i] == c) {
            return m_values[base + i];
        }
    }
    // not found -> create an entry at the end if capacity allows
    if(len < m_row_capacity){
        size_t i = len;
        m_col_inds[base + i] = c;
        m_values[base + i] = this->m_zero;
        m_row_len[r] = len + 1;
        return m_values[base + i];
    }
    throw std::runtime_error("No capacity to insert new entry in row");
}

SparseMatrix SparseMatrix::operator*(double s) const {
    SparseMatrix result(this->m_rows, this->m_cols, this->m_row_capacity);
    #pragma omp parallel for schedule(static) if(this->m_rows > OMP_MIN_ROWS)
    for(size_t r = 0; r < this->m_rows; ++r) {
        size_t base = r * m_row_capacity;
        //size_t len = m_row_len[r];
        for(size_t i = 0; i < m_row_capacity; ++i) {
            result(r, m_col_inds[base + i]) = m_values[base + i] * s;
        }
    }
    return result;
}

// Optimized for SIMD: processes all rowCapacity elements, using 0 for invalid entries
Vector SparseMatrix::operator*(const Vector& v) const
{
    if(v.size() != this->m_cols){
        throw std::runtime_error("SparseMatrix::operator*: Incompatible sizes.");
    }
    Vector result(this->m_rows, 0.0);
    
    #pragma omp parallel for schedule(static) if(this->m_rows > OMP_MIN_ROWS)
    for(size_t r = 0; r < this->m_rows; ++r) {
        size_t base = r * m_row_capacity;
        double sum = 0.0;
        // Branchless: unused slots have val=0 and col=0
        for(size_t i = 0; i < m_row_capacity; ++i) {
            sum += m_values[base + i] * v[m_col_inds[base + i]];
        }
        result[r] = sum;
    }
    return result;
}

SparseMatrix SparseMatrix::operator*(const SparseMatrix& m) const
{
    if(this->m_cols != m.num_rows()){
        throw std::runtime_error("SparseMatrix::operator*: Incompatible sizes.");
    }
    std::size_t max_row_capacity = this->m_row_capacity * m.row_capacity();
    SparseMatrix result(this->m_rows, m.num_cols(), max_row_capacity);
    
    const bool use_parallel = (this->m_rows * m.num_cols() > OMP_MIN_SIZE);
    #pragma omp parallel for schedule(static) if(use_parallel)
    for(size_t r = 0; r < this->m_rows; ++r) {
        size_t base1 = r * m_row_capacity;
        //size_t len1 = m_row_len[r];
        for(size_t i = 0; i < m_row_capacity; ++i) {
            size_t c1 = m_col_inds[base1 + i];
            double v1 = m_values[base1 + i];
            size_t base2 = c1 * m.m_row_capacity;
            //size_t len2 = m.m_row_len[c1];
            for(size_t j = 0; j < m.m_row_capacity; ++j) {
                size_t c2 = m.m_col_inds[base2 + j];
                double v2 = m.m_values[base2 + j];
                result(r, c2) += v1 * v2;
                // Since we iterate over rows at a time, we accumulate results
                // over multiple iterations to the same entry. It's equivalent
                // but less understandable than a direct multiplication.
            }
        }
    }
    return result;
}

SparseMatrix SparseMatrix::transpose() const
{
    SparseMatrix result(this->m_cols, this->m_rows, this->m_row_capacity);
    for(size_t r = 0; r < this->m_rows; ++r) {
        size_t base = r * m_row_capacity;
        size_t len = m_row_len[r];
        for(size_t i = 0; i < len; ++i) {
            size_t col = m_col_inds[base + i];
            result(col, r) = m_values[base + i];
        }
    }
    return result;
}

Vector SparseMatrix::transpose_multiply(const Vector& v) const
{
    if(v.size() != this->m_rows){
        throw std::runtime_error("SparseMatrix::transpose_multiply: Incompatible sizes.");
    }
    Vector result(this->m_cols, 0.0);
    for(size_t r = 0; r < this->m_rows; ++r) {
        size_t base = r * m_row_capacity;
        size_t len = m_row_len[r];
        for(size_t i = 0; i < len; ++i) {
            size_t col = m_col_inds[base + i];
            result[col] += m_values[base + i] * v[r];
        }
    }
    return result;
}

SparseMatrix SparseMatrix::transpose_multiply(const SparseMatrix& m) const
{
    if (this->m_cols != m.num_cols()) {
        throw std::runtime_error("SparseMatrix::transpose_multiply: Incompatible sizes.");
    }
    SparseMatrix result(this->m_rows, m.num_rows(), this->m_row_capacity);
    
    for (size_t r1 = 0; r1 < this->m_rows; ++r1) {
        size_t base1 = r1 * m_row_capacity;
        size_t len1 = m_row_len[r1];
        for (size_t i = 0; i < len1; ++i) {
            size_t c1 = m_col_inds[base1 + i];
            
            double v1 = m_values[base1 + i];
            size_t base2 = c1 * m.m_row_capacity;
            
            size_t len2 = m.m_row_len[c1];
            for (size_t j = 0; j < len2; ++j) {
                size_t c2 = m.m_col_inds[base2 + j];
                
                double v2 = m.m_values[base2 + j];
                result(r1, c1) += v1 * v2;
            }
        }
    }
    return result;
}

void SparseMatrix::clear(){
    std::fill(this->m_values.begin(), this->m_values.end(), this->m_zero);
    std::fill(this->m_col_inds.begin(), this->m_col_inds.end(), 0);
    std::fill(this->m_row_len.begin(), this->m_row_len.end(), 0);

    // alternatively, could just do:
    //*this = SparseMatrix(this->m_rows, this->m_cols, this->m_row_capacity);
    
}

void SparseMatrix::clear_row(std::size_t r) {
    if (r >= m_rows) {
        throw std::out_of_range("SparseMatrix::clear_row: Row index out of range.");
    }
    size_t base = r * m_row_capacity;
    // Clear all slots in this row (important for SIMD multiplication)
    for(size_t i = 0; i < m_row_capacity; ++i) {
        m_values[base + i] = 0.0;
        m_col_inds[base + i] = 0;
    }
    m_row_len[r] = 0;
}

std::ostream& operator<<(std::ostream& stream, const SparseMatrix& m)
{
    for(std::size_t r = 0; r < m.num_rows(); ++r){
        stream << "| ";
        for(std::size_t c = 0; c < m.num_cols(); ++c){
            stream << std::setw(8) << std::setprecision(2) << m(r,c) << " ";
        }
        stream << "|\n";
    }
    return stream;
}

///calculates scaled transpose of matrix A with scale factor s
void CreateScaledTranspose(SparseMatrix& mT, const double s, const SparseMatrix& m)
{
    // Transpose: mT(j,i) = s * m(i,j)
    // Count non-zeros per column of m to determine row capacities for mT
    std::vector<std::size_t> nnz_per_col(m.num_cols(), 0);
    
    for(std::size_t i = 0; i < m.num_rows(); ++i) {
        size_t base = i * m.row_capacity();
        size_t len = m.get_row_len()[i];
        for(size_t j = 0; j < len; ++j) {
            size_t col = m.get_col_inds()[base + j];
            ++nnz_per_col[col];
        }
    }
    
    // Find max non-zeros per column (will be row capacity for transposed)
    std::size_t max_nnz = 0;
    for(std::size_t i = 0; i < nnz_per_col.size(); ++i) {
        if(nnz_per_col[i] > max_nnz) {
            max_nnz = nnz_per_col[i];
        }
    }
    
    // Resize transposed matrix with appropriate row capacity
    mT = SparseMatrix(m.num_cols(), m.num_rows(), max_nnz);
    
    // Fill transposed matrix
    for(std::size_t i = 0; i < m.num_rows(); ++i) {
        size_t base = i * m.row_capacity();
        size_t len = m.get_row_len()[i];
        for(size_t j = 0; j < len; ++j) {
            size_t col = m.get_col_inds()[base + j];
            mT(col, i) = s * m.get_values()[base + j];
        }
    }
}

