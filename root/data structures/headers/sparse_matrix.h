/*
 * sparse_matrix.h
 *
 *  Created on: 2019-04-28
 *      Author: mbreit
 */

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H


#include <cstddef>
#include <vector>
#include <cassert>
#include <set>
#include <map>
#include "vector.h"


class SparseMatrixIteratorRegistry;


class SparseMatrix
{
	public:
		/**
		 * @brief Constructor without arguments
		 * The maximal number of non-zero entries per row is set to a default value of 5.
		 */
		SparseMatrix();

		/** @brief Constructor with size, maximal number of non-zero entries per row and optional default value
		 *	The matrix is to be initialized with the given sizes and all entries are to be initialized with 0.0.
		**/
		SparseMatrix(std::size_t r, std::size_t c, std::size_t rowCapacity = 5);

		/// Destructor
		virtual ~SparseMatrix();


		/// Return number of rows
		std::size_t num_rows() const { return m_rows; }

		/// Return number of columns
		std::size_t num_cols() const { return m_cols; }

		// Return row capacity (for debugging)
		std::size_t row_capacity() const { return m_row_capacity; }

		// Return m_values vector (for debugging)
		const std::vector<double>& get_values() const { return m_values; }

		// Return m_col_inds vector (for debugging)
		const std::vector<size_t>&  get_col_inds() const { return m_col_inds; }

		// Return row lengths (for debugging)
		const std::vector<size_t>& get_row_len() const { return m_row_len; }

		/// Resize the matrix
		/**
		 * The matrix is resized to the given sizes.
		 * New elements are always initialized with 0.0, regardless of the
		 * default value parameter, which is only present to enhance compatibility.
		 */
		void resize(std::size_t r, std::size_t c, double defVal = 0.0);

	public:

		/// Return whether entry exists
		bool has_entry(std::size_t r, std::size_t c) const;

		/**
		 * @brief Element access (read-only)
		 * Returns the value at the given position.
		 */
		double operator()(std::size_t r, std::size_t c) const;

		/**
		 * @brief Element access
		 * Returns a reference to the value at the given position.
		 * If the entry is not in the matrix, it is created.
		 * In that process, it may happen that the row is already full.
		 * This has to produce an error (or - advanced - the row capacity has to be increased).
		 */
		double& operator()(std::size_t r, std::size_t c);

		/**
		 * @brief Matrix-scalar multiplication
		 * Each entry of the matrix is multiplied by the given scalar.
		 * The result is returned in a new instance.
		 */
		SparseMatrix operator*(double s) const;

		/**
		 * @brief Matrix-vector multiplication
		 * The matrix is applied on the given vector.
		 * The result is returned in a new instance.
		 */
		Vector operator*(const Vector& v) const;

		/**
		 * @brief Matrix-matrix multiplication
		 * The result is returned in a new instance.
		 */
		SparseMatrix operator*(const SparseMatrix& m) const;

		/**
		 * @brief Transpose of the matrix
		 * The transpose is returned in a new instance.
		 */
		SparseMatrix transpose() const;

		/**
		 * @brief Efficient Vector-transposed Matrix multiplication
		 * The result is returned in a new instance.
		 */
		Vector transpose_multiply(const Vector& v) const;

		/**
		 * @brief Efficient Matrix-transposed Matrix multiplication
		 * The result is returned in a new instance.
		 */
		SparseMatrix transpose_multiply(const SparseMatrix& m) const;

        /**
         * @brief Sets all entries of the matrix to 0.
         */
        void clear();

        /**
         * @brief Clears a specific row (sets all entries to 0 and resets row length)
         */
        void clear_row(std::size_t r);
		
    protected:
        std::size_t m_rows;
        std::size_t m_cols;
        std::size_t m_row_capacity;
        std::vector<double> m_values;
        std::vector<size_t> m_col_inds;
		std::vector<size_t> m_row_len; // number of valid entries per row
        double m_zero = 0.0;

};



/// write matrix to output stream
std::ostream& operator<<(std::ostream& stream, const SparseMatrix& m);

void CreateScaledTranspose(SparseMatrix& mT, const double s, const SparseMatrix& m);

#endif // SPARSE_MATRIX_H
