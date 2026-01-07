/*
 * matrix.cpp
 *
 *  Created on: 2019-04-28
 *      Author: 
 */

#include "headers/matrix.h"
#include <cassert>
#include <iostream>
#include <iomanip>

Matrix::Matrix()
{
	m_rows = 0;
	m_cols = 0;
}


Matrix::Matrix(std::size_t r, std::size_t c, double val)
{
    m_rows = r;
    m_cols = c;
    m_mat.resize(r * c, val);
}

Matrix::Matrix(const Matrix& M){
    m_rows = M.m_rows;
    m_cols = M.m_cols;
    m_mat = M.m_mat;
}

Matrix::~Matrix()
{}


std::size_t Matrix::num_rows() const
{
	return m_rows;
}


std::size_t Matrix::num_cols() const
{
	return m_cols;
}



void Matrix::resize(std::size_t r, std::size_t c, double val)
{
    m_rows = r;
    m_cols = c;
    m_mat.resize(r * c, val);
}





double Matrix::operator()(std::size_t r, std::size_t c) const
{
	// the const operator should NOT return a reference to
    //       the matrix entry. -> Create a new double, set
    //       it to the matrix entry and return it!
    double val = m_mat[r * m_cols + c];
    return val;
}


double& Matrix::operator()(std::size_t r, std::size_t c)
{
    //  return reference to matrix entry. This means, we can
    //       then modify this entry from outside of the class i.e.
    //       M(i,j) = value
    return m_mat[r * m_cols + c];
}

///calculates product M \cdot v = b
Vector Matrix::operator*(Vector v) const
{
    Vector result(m_rows, 0.0);
    for (std::size_t i = 0; i < m_rows; ++i) {
        for (std::size_t j = 0; j < m_cols; ++j) {
            result[i] += (*this)(i, j) * v[j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& m) const
{
    if (this->m_cols != m.m_rows) {
        throw std::runtime_error("Matrix::operator*: Incompatible sizes.");
    }
    Matrix result(this->m_rows, m.m_cols, 0.0);
    for (std::size_t i = 0; i < this->m_rows; ++i) {
        for (std::size_t j = 0; j < m.m_cols; ++j) {
            for (std::size_t k = 0; k < this->m_cols; ++k) {
                result(i, j) += (*this)(i, k) * m(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(this->m_cols, this->m_rows, 0.0);
    for (std::size_t i = 0; i < this->m_rows; ++i) {
        for (std::size_t j = 0; j < this->m_cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::clear(){

    for (std::size_t i = 0; i < m_mat.size(); ++i) {
        m_mat[i] = 0.0;
    }
}

std::ostream& operator<<(std::ostream& stream, const Matrix& m)
{

    if(m.num_rows() == 0 || m.num_cols() == 0) {return stream << "()";}

    for(std::size_t r = 0; r < m.num_rows(); ++r){
        stream << "| ";
        for(std::size_t c = 0; c < m.num_cols(); ++c){
            stream << std::setw(8) << std::setprecision(2) << m(r,c) << " ";
        }
        stream << "|\n";
    }
    return stream;

}
void CreateScaledTranspose(Matrix& mT, const double s, const Matrix& m)
{
    mT.resize(m.num_cols(), m.num_rows());
    for (std::size_t i = 0; i < m.num_rows(); ++i) {
        for (std::size_t j = 0; j < m.num_cols(); ++j) {
            mT(j, i) = s * m(i, j);
        }
    }
}



// explicit template instantiations
// Iterators removed for SIMD-friendly design; direct indexing is used.

