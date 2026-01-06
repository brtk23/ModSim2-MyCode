/*
 * sparse_matrix.cpp
 *
 *  Created on: 2019-04-28
 *      Author: 
 */

#include "headers/sparse_matrix.h"
#include <iostream>
#include <iomanip>

//TODO GET RID OF ITERATORS FOR PARALLELIZATION PURPOSES (SIMD)

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
    this->m_col_inds.resize(r * rowCapacity, (size_t) -1);
}


SparseMatrix::~SparseMatrix()
{}


void SparseMatrix::resize(std::size_t r, std::size_t c, double defVal)
{
    this->m_rows = r;
    this->m_cols = c;

    this->m_values.resize(r * this->m_row_capacity, this->m_zero); 
    this->m_col_inds.resize(r * this->m_row_capacity, (size_t) -1);
}



template<bool is_const>
SparseMatrix::RowIteratorBase<is_const>::RowIteratorBase
(
	typename iterator_traits<is_const>::matrix_type& mat,
	std::size_t rowIndex
)
{
    pMat = &mat;
    if(pMat->num_rows() <= rowIndex){
        throw std::runtime_error("\"ugh.. RowIteratorBase was instantiated with rowIndex > mat.num_rows()-1...");
    }
    pCurInd = &mat.m_col_inds[rowIndex * mat.m_row_capacity];
    pCurVal = &mat.m_values[rowIndex * mat.m_row_capacity];
}


template<bool is_const>
SparseMatrix::RowIteratorBase<is_const>::RowIteratorBase
(
	typename SparseMatrix::iterator_traits<is_const>::matrix_type& mat,
	std::size_t rowIndex,
	std::size_t startFromCol
)
{
    pCurInd = &mat.m_col_inds[rowIndex * mat.m_row_capacity];
    pCurVal = &mat.m_values[rowIndex * mat.m_row_capacity];
	for(size_t i = 0; i < mat.m_row_capacity; ++i){
        if(*pCurInd == startFromCol){
            break;
        }
        ++pCurInd;
        ++pCurVal;
    }
}


template<bool is_const>
bool SparseMatrix::RowIteratorBase<is_const>::operator!=(RowIteratorBase& other) const
{
    return pCurInd != other.pCurInd;
}


template<bool is_const>
bool SparseMatrix::RowIteratorBase<is_const>::operator==(RowIteratorBase& other) const
{
    return pCurInd == other.pCurInd;
}


template<bool is_const>
SparseMatrix::RowIteratorBase<is_const>& SparseMatrix::RowIteratorBase<is_const>::operator++()
{
    ++pCurInd;
    ++pCurVal;
    return *this;
}

template<bool is_const>
typename SparseMatrix::iterator_traits<is_const>::entry_type SparseMatrix::RowIteratorBase<is_const>::value() const
{
    return *pCurVal;
}

template<bool is_const>
typename SparseMatrix::iterator_traits<is_const>::entry_type& SparseMatrix::RowIteratorBase<is_const>::value()
{
    return *pCurVal;
}

template<bool is_const>
std::size_t SparseMatrix::RowIteratorBase<is_const>::col_index() const
{
    return *pCurInd;
}



SparseMatrix::RowIterator SparseMatrix::begin(std::size_t r)
{
    return RowIterator{*this, r};
}


SparseMatrix::RowIterator SparseMatrix::end(std::size_t r)
{
    return RowIterator{*this, r, (size_t) -1};
}


SparseMatrix::ConstRowIterator SparseMatrix::begin(std::size_t r) const
{
    return ConstRowIterator{*this, r};
}


SparseMatrix::ConstRowIterator SparseMatrix::end(std::size_t r) const
{
    return ConstRowIterator{*this, r, (size_t) -1};
}


bool SparseMatrix::has_entry(std::size_t r, std::size_t c) const
{
    auto b = begin(r);
    auto e = end(r);
    for(auto it = b; it != e; ++it) {
        if(it.col_index() == c){
            return true;
        }
    }
    return false;
}


double SparseMatrix::operator()(std::size_t r, std::size_t c) const
{
    auto b = begin(r);
    auto e = end(r);
    for(auto it = b; it != e; ++it) {
        if(it.col_index() == c){
            return it.value();
        }
    }
    return 0.0; // Return a default value
}


double& SparseMatrix::operator()(std::size_t r, std::size_t c)
{
    auto b = begin(r);
    auto e = end(r);
    for(auto it = b; it != e; ++it) {
        if(it.col_index() == c) return it.value();
    }
    // not found -> create an entry (simple strategy: find first free slot)
    size_t base = r * m_row_capacity;
    for(size_t i = 0; i < m_row_capacity; ++i){
        if(m_col_inds[base + i] == (size_t) -1){
            m_col_inds[base + i] = c;
            m_values[base + i] = this->m_zero; // set appropriate default
            return m_values[base + i];
        }
    }
    throw std::runtime_error("No capacity to insert new entry in row");
}

SparseMatrix SparseMatrix::operator*(double s) const {
    SparseMatrix result(this->m_rows, this->m_cols, this->m_row_capacity);
    for(size_t r = 0; r < this->m_rows; ++r) {
        auto b = begin(r);
        auto e = end(r);
        for(auto it = b; it != e; ++it) {
            result(r, it.col_index()) = it.value() * s;
        }
    }
    return result;
}

// TODO : Optimize this function
Vector SparseMatrix::operator*(const Vector& v) const
{
    if(v.size() != this->m_cols){
        throw std::runtime_error("SparseMatrix::operator*: Incompatible sizes.");
    }
    Vector result(this->m_rows, 0.0);
    for(size_t r = 0; r < this->m_rows; ++r) {
        auto b = begin(r);
        auto e = end(r);
        for(auto it = b; it != e; ++it) {
            result[r] += it.value() * v[it.col_index()];
        }
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
    for(size_t r = 0; r < this->m_rows; ++r) { // row of first matrix
        auto b1 = begin(r);
        auto e1 = end(r);
        for(auto it1 = b1; it1 != e1; ++it1) { // 
            size_t c1 = it1.col_index(); // column of first matrix
            double v1 = it1.value(); 
            auto b2 = m.begin(c1);
            auto e2 = m.end(c1);
            for(auto it2 = b2; it2 != e2; ++it2) { 
                size_t c2 = it2.col_index(); // column of second matrix
                double v2 = it2.value();
                result(r, c2) += v1 * v2;
                // Instead of going cell by cell (all muls and adds), we go multiple times
                // through the same cell of the result matrix, accumulating contributions
                // because we can only iterate in one row of the sparse matrix at a time.
            }
        }
    }
    return result;
}


SparseMatrix SparseMatrix::transpose() const
{
    SparseMatrix result(this->m_cols, this->m_rows, this->m_row_capacity);
    for(size_t r = 0; r < this->m_rows; ++r) {
        auto b = begin(r);
        auto e = end(r);
        for(auto it = b; it != e; ++it) {
            result(it.col_index(), r) = it.value();
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
        auto b = begin(r);
        auto e = end(r);
        for(auto it = b; it != e; ++it) {
            result[it.col_index()] += it.value() * v[r];
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
        auto b1 = this->begin(r1);
        auto e1 = this->end(r1);
        for (auto it1 = b1; it1 != e1; ++it1) {
            size_t c1 = it1.col_index();
            double v1 = it1.value();
            auto b2 = m.begin(c1);
            auto e2 = m.end(c1);
            for (auto it2 = b2; it2 != e2; ++it2) {
                size_t c2 = it2.col_index();
                double v2 = it2.value();
                result(r1, c1) += v1 * v2; // IDK if this is correct (TODO)
            }
        }
    }
    return result;
}

void SparseMatrix::clear(){
    std::fill(this->m_values.begin(), this->m_values.end(), this->m_zero);
    std::fill(this->m_col_inds.begin(), this->m_col_inds.end(), (size_t) -1);

    // alternatively, could just do:
    //*this = SparseMatrix(this->m_rows, this->m_cols, this->m_row_capacity);
    
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
        auto b = m.begin(i);
        auto e = m.end(i);
        for(auto it = b; it != e; ++it) {
            ++nnz_per_col[it.col_index()];
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
        auto b = m.begin(i);
        auto e = m.end(i);
        for(auto it = b; it != e; ++it) {
            std::size_t j = it.col_index();
            mT(j, i) = s * it.value();
        }
    }
}


// explicit template instantiations
template class SparseMatrix::RowIteratorBase<true>;
template class SparseMatrix::RowIteratorBase<false>;

