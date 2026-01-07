/*
 * matrix.h
 *
 *  Created on: 2019-04-28
 *      Author: mbreit
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"
#include <ostream>


class Matrix
{
	public:
		/** @brief Constructor without arguments
		 *	The matrix is to be initialized with a size of 0x0.
		**/
		Matrix();

		/** @brief Constructor with size and optional default value
		 *	The matrix is to be initialized with the given sizes and all elements
		 *	are to be initialized with the given default value (0.0 if not given).
		**/
		Matrix(std::size_t r, std::size_t c, double val = 0.0);

        Matrix(const Matrix& M);
		/// Destructor
		virtual ~Matrix();

		/// Return number of rows
		std::size_t num_rows() const;

		/// Return number of columns
		std::size_t num_cols() const;

		/// Resize the matrix
		/**
		 * The matrix is resized to the given sizes. If elements are created
		 * in that process, they are to be initialized with the optional default
		 * value (0.0 if not given).
		 */
		void resize(std::size_t r, std::size_t c, double val = 0.0);

	public:
		/**
		 * @brief Element access (read-only)
		 * Returns the value at the given position.
		 */
		double operator()(std::size_t r, std::size_t c) const;

		/**
		 * @brief Element access
		 * Returns a reference to the value at the given position.
		 */
		double& operator()(std::size_t r, std::size_t c);


		/**
		 * @brief Matrix-vector multiplication
		 * The matrix is applied on the given vector.
		 * The result is returned in a new instance.
		 */
		Vector operator*(Vector v) const;

		/**
		 * @brief Matrix-matrix multiplication
		 * The matrix is multiplied with the given matrix.
		 * The result is returned in a new instance.
		 */
		Matrix operator*(const Matrix& m) const;

		/**
		 * @brief Transpose of the matrix
		 * The transpose of the matrix is returned in a new instance.
		 */
		Matrix transpose() const;

        /**
         * @brief Sets all entries of the matrix to 0.
         */
        void clear();
    private:
        Vector m_mat;
        size_t m_rows = 0;
        size_t m_cols = 0;
};

/// write matrix to output stream
std::ostream& operator<<(std::ostream& stream, const Matrix& m);

void CreateScaledTranspose(Matrix& mT, const double s, const Matrix& m);

#endif // MATRIX_H
