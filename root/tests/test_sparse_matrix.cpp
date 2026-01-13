#include <iostream>
#include <cassert>
#include <cmath>
#include <sstream>
#include "../data structures/headers/sparse_matrix.h"
#include "../data structures/headers/matrix.h"
#include "../data structures/headers/vector.h"

// Color codes for output
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"

class TestRunner {
private:
    int passed = 0;
    int failed = 0;
    std::string current_test;

public:
    void test(const std::string& name, bool condition, const std::string& message = "") {
        current_test = name;
        if (condition) {
            passed++;
            std::cout << GREEN << "PASS: " << name << RESET << std::endl;
        } else {
            failed++;
            std::cout << RED << "FAIL: " << name << RESET;
            if (!message.empty()) {
                std::cout << " (" << message << ")";
            }
            std::cout << std::endl;
        }
    }

    void assert_equal(double a, double b, double tolerance = 1e-10) {
        test(current_test, std::fabs(a - b) < tolerance, 
             "expected " + std::to_string(a) + ", got " + std::to_string(b));
    }

    void assert_equal(int a, int b) {
        test(current_test, a == b,
             "expected " + std::to_string(a) + ", got " + std::to_string(b));
    }

    void print_summary() {
        std::cout << "\n" << YELLOW << "======================================" << RESET << std::endl;
        std::cout << "Tests passed: " << GREEN << passed << RESET << std::endl;
        std::cout << "Tests failed: " << (failed > 0 ? RED : GREEN) << failed << RESET << std::endl;
        std::cout << YELLOW << "======================================" << RESET << std::endl;
    }

    bool all_passed() const {
        return failed == 0;
    }
};

void test_constructor(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Constructor..." << RESET << std::endl;

    SparseMatrix m(5, 5, 10);
    test.test("Constructor - rows", m.num_rows() == 5);
    test.test("Constructor - cols", m.num_cols() == 5);
    test.test("Constructor - row_capacity", m.row_capacity() == 10);
}

void test_operator_read(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Read Operator ()..." << RESET << std::endl;
    

    SparseMatrix m(3, 3, 5);
    
    // Reading from empty matrix should return 0
    test.test("Read empty entry", m(0, 0) == 0.0);
    test.test("Read empty entry (different position)", m(2, 1) == 0.0);
}

void test_operator_write(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Write Operator ()..." << RESET << std::endl;
    

    SparseMatrix m(3, 3, 5);
    
    // Write and read back
    m(0, 0) = 5.0;
    test.test("Write and read back (0,0)", m(0, 0) == 5.0);
    
    m(1, 2) = 3.5;
    test.test("Write and read back (1,2)", m(1, 2) == 3.5);
    
    m(2, 2) = -2.0;
    test.test("Write and read back (2,2)", m(2, 2) == -2.0);
    
    // Other entries should still be 0
    test.test("Unwritten entry still 0", m(1, 1) == 0.0);
}

void test_matrix_vector_multiplication(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Matrix-Vector Multiplication..." << RESET << std::endl;
    

    // Create a simple 3x3 matrix
    SparseMatrix m(3, 3, 5);
    m(0, 0) = 2.0;
    m(0, 1) = 1.0;
    m(1, 1) = 3.0;
    m(2, 2) = 4.0;

    // Create vector [1, 2, 3]
    Vector v(3, 0.0);
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;

    // Multiply: A * v
    Vector result = m * v;

    // Expected: 
    // row 0: 2*1 + 1*2 = 4
    // row 1: 3*2 = 6
    // row 2: 4*3 = 12
    test.test("Matrix-Vector multiplication - element 0", result[0] == 4.0);
    test.test("Matrix-Vector multiplication - element 1", result[1] == 6.0);
    test.test("Matrix-Vector multiplication - element 2", result[2] == 12.0);
}

void test_matrix_vector_multiplication_dense(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Matrix-Vector Multiplication (Dense Sparse Matrix)..." << RESET << std::endl;
    

    // Create a denser matrix (tridiagonal-like)
    SparseMatrix m(4, 4, 6);
    m(0, 0) = 4.0;
    m(0, 1) = -1.0;
    m(1, 0) = -1.0;
    m(1, 1) = 4.0;
    m(1, 2) = -1.0;
    m(2, 1) = -1.0;
    m(2, 2) = 4.0;
    m(2, 3) = -1.0;
    m(3, 2) = -1.0;
    m(3, 3) = 4.0;

    Vector v(4, 1.0);  // [1, 1, 1, 1]
    Vector result = m * v;

    // Expected:
    // row 0: 4*1 - 1*1 = 3
    // row 1: -1*1 + 4*1 - 1*1 = 2
    // row 2: -1*1 + 4*1 - 1*1 = 2
    // row 3: -1*1 + 4*1 = 3
    test.test("Dense matrix multiplication - element 0", result[0] == 3.0);
    test.test("Dense matrix multiplication - element 1", result[1] == 2.0);
    test.test("Dense matrix multiplication - element 2", result[2] == 2.0);
    test.test("Dense matrix multiplication - element 3", result[3] == 3.0);
}

void test_matrix_matrix_multiplication(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Matrix-Matrix Multiplication..." << RESET << std::endl;
    

    // Create matrix A: 2x3
    SparseMatrix A(2, 3, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 1) = 3.0;
    A(1, 2) = 4.0;

    // Create matrix B: 3x2
    SparseMatrix B(3, 2, 2);
    B(0, 0) = 2.0;
    B(1, 0) = 1.0;
    B(1, 1) = 3.0;
    B(2, 1) = 2.0;

    // Multiply: A * B (2x3) * (3x2) = (2x2)
    SparseMatrix result = A * B;

    // Expected:
    // (0,0): 1*2 + 2*1 = 4
    // (0,1): 1*0 + 2*3 = 6
    // (1,0): 0*2 + 3*1 + 4*0 = 3
    // (1,1): 0*0 + 3*3 + 4*2 = 17
    test.test("Matrix-Matrix multiplication - (0,0)", result(0, 0) == 4.0);
    test.test("Matrix-Matrix multiplication - (0,1)", result(0, 1) == 6.0);
    test.test("Matrix-Matrix multiplication - (1,0)", result(1, 0) == 3.0);
    test.test("Matrix-Matrix multiplication - (1,1)", result(1, 1) == 17.0);
}

void test_matrix_matrix_multiplication_identity(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Matrix-Matrix Multiplication with Identity..." << RESET << std::endl;
    

    // Create a simple matrix
    SparseMatrix A(3, 3, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 1) = 3.0;
    A(2, 2) = 4.0;

    // Create identity matrix
    SparseMatrix I(3, 3, 1);
    I(0, 0) = 1.0;
    I(1, 1) = 1.0;
    I(2, 2) = 1.0;

    // A * I should equal A
    SparseMatrix result = A * I;

    test.test("Identity multiplication - (0,0)", result(0, 0) == 1.0);
    test.test("Identity multiplication - (0,1)", result(0, 1) == 2.0);
    test.test("Identity multiplication - (1,1)", result(1, 1) == 3.0);
    test.test("Identity multiplication - (2,2)", result(2, 2) == 4.0);
}

void test_scalar_multiplication(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Scalar Multiplication..." << RESET << std::endl;
    

    SparseMatrix m(2, 2, 3);
    m(0, 0) = 2.0;
    m(0, 1) = 4.0;
    m(1, 1) = 6.0;

    SparseMatrix result = m * 0.5;

    std::cout << m << std::endl << result << std::endl;

    test.test("Scalar multiplication - (0,0)", result(0, 0) == 1.0);
    test.test("Scalar multiplication - (0,1)", result(0, 1) == 2.0);
    test.test("Scalar multiplication - (1,1)", result(1, 1) == 3.0);
}

void test_has_entry(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing has_entry()..." << RESET << std::endl;
    

    SparseMatrix m(3, 3, 5);
    m(0, 0) = 1.0;
    m(1, 2) = 2.0;

    test.test("has_entry - existing entry (0,0)", m.has_entry(0, 0));
    test.test("has_entry - existing entry (1,2)", m.has_entry(1, 2));
    test.test("has_entry - non-existing entry (0,1)", !m.has_entry(0, 1));
    test.test("has_entry - non-existing entry (2,2)", !m.has_entry(2, 2));
}

void test_resize(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing resize()..." << RESET << std::endl;
    

    SparseMatrix m(2, 2, 3);
    m(0, 0) = 5.0;
    m(1, 1) = 3.0;

    m.resize(4, 4, 0.0);

    test.test("Resize - new rows", m.num_rows() == 4);
    test.test("Resize - new cols", m.num_cols() == 4);
    // After resize, old data remains
    test.test("Resize - old data consistency", m(0, 0) == 5.0);
}

void test_clear(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing clear()..." << RESET << std::endl;
    

    SparseMatrix m(3, 3, 5);
    m(0, 0) = 1.0;
    m(1, 1) = 2.0;
    m(2, 2) = 3.0;

    m.clear();

    test.test("Clear - (0,0) is 0", m(0, 0) == 0.0);
    test.test("Clear - (1,1) is 0", m(1, 1) == 0.0);
    test.test("Clear - (2,2) is 0", m(2, 2) == 0.0);
}

void test_transpose(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing transpose()..." << RESET << std::endl;
    

    SparseMatrix m(2, 3, 5);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(0, 2) = 3.0;
    m(1, 1) = 4.0;
    m(1, 2) = 5.0;

    SparseMatrix mT = m.transpose();

    test.test("Transpose - dimensions", mT.num_rows() == 3 && mT.num_cols() == 2);
    test.test("Transpose - (0,0)", mT(0, 0) == 1.0);
    test.test("Transpose - (1,0)", mT(1, 0) == 2.0);
    test.test("Transpose - (1,1)", mT(1, 1) == 4.0);
    test.test("Transpose - (2,1)", mT(2, 1) == 5.0);
}

void test_transpose_multiply(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing transpose_multiply(Vector)..." << RESET << std::endl;
    

    SparseMatrix m(2, 3, 5);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 1) = 3.0;
    m(1, 2) = 4.0;

    Vector v(2, 0.0);
    v[0] = 2.0;
    v[1] = 3.0;

    // m^T * v where m^T is 3x2
    // m is [1 2 0; 0 3 4]
    // m^T is [1 0; 2 3; 0 4]
    // m^T * [2, 3] = [1*2 + 0*3; 2*2 + 3*3; 0*2 + 4*3] = [2; 13; 12]
    Vector result = m.transpose_multiply(v);

    test.test("Transpose multiply - size", result.size() == 3);
    test.test("Transpose multiply - element 0", result[0] == 2.0);
    test.test("Transpose multiply - element 1", result[1] == 13.0);
    test.test("Transpose multiply - element 2", result[2] == 12.0);
}

void test_row_capacity_limits(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Row Capacity Limits..." << RESET << std::endl;
    

    SparseMatrix m(2, 5, 3);  // Only 3 entries per row allowed

    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(0, 2) = 3.0;

    // This should not throw - we should handle it gracefully
    bool threw = false;
    try {
        m(0, 3) = 4.0;  // Try to add a 4th entry
    } catch (const std::runtime_error& e) {
        threw = true;
    }

    test.test("Row capacity exceeded throws exception", threw);
    test.test("Existing entries still accessible", m(0, 0) == 1.0);
}

void test_large_sparse_matrix(TestRunner& test) {
    std::cout << "\n" << YELLOW << "Testing Large Sparse Matrix..." << RESET;
    size_t n = 15;
    std::cout << "\n" << YELLOW << "(" << n << "x" << n << ")" << RESET << std::endl;
    SparseMatrix sparse(n, n, 5);  // Only 5 entries per row
    Matrix dense(n, n);

    // Fill diagonal
    for (size_t i = 0; i < n; ++i) {
        sparse(i, i) = 2.0;
        dense(i, i) = 2.0;
    }

    // Fill super/sub diagonals
    for (size_t i = 0; i < n - 1; ++i) {
        sparse(i, i + 1) = -1.0;
        sparse(i + 1, i) = -1.0;
        dense(i, i + 1) = -1.0;
        dense(i + 1, i) = -1.0;
    }

    test.test("Large matrix - diagonal element", sparse(n-1, n-1) == 2.0);
    test.test("Large matrix - superdiagonal element", sparse(n-2, n-1) == -1.0);
    test.test("Large matrix - subdiagonal element", sparse(n-1, n-2) == -1.0);

    Vector v(n, 1.0);
    Vector result = sparse * v;

    // First element: 2*1 - 1*1 = 1
    // Middle elements: -1*1 + 2*1 - 1*1 = 0
    // Last element: -1*1 + 2*1 = 1
    test.test("Large matrix vector multiplication - first element", std::fabs(result[0] - 1.0) < 1e-10);
    test.test("Large matrix vector multiplication - middle element", std::fabs(result[int(n/2)] - 0.0) < 1e-10);
    test.test("Large matrix vector multiplication - last element", std::fabs(result[n-1] - 1.0) < 1e-10);

    // Test matrix-matrix multiplication with larger matrices
    SparseMatrix sparse2(n, n, 5);
    Matrix dense2(n, n);
    // Fill sparse2 as identity plus some off-diagonal
    for (size_t i = 0; i < n; ++i) {
        sparse2(i, i) = 1.0;
        dense2(i, i) = 1.0;
    }
    for (size_t i = 0; i < n - 1; ++i) {
        sparse2(i, i + 1) = 0.5;
        dense2(i, i + 1) = 0.5;
    }

    SparseMatrix result_sparse = sparse * sparse2;
    //std::cout << "Result Sparse\n" << result_sparse << std::endl;
    Matrix result_dense = dense * dense2;
    //std::cout << "Result Dense\n" << result_dense << std::endl;

    bool all_equal = true;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (std::fabs(result_sparse(i, j) - result_dense(i, j)) > 1e-10) {
                all_equal = false;
                break;
            }
        }
        if (!all_equal) break;
    }
    // Expected results
    test.test("Large (100x100) sparse matrix matrix multiplication - equal to dense matrix result",
              all_equal);
}

int main() {
    std::cout << "\n" << YELLOW << "====================================" << RESET << std::endl;
    std::cout << YELLOW << "  SPARSE MATRIX UNIT TESTS" << RESET << std::endl;
    std::cout << YELLOW << "====================================" << RESET << std::endl;
    TestRunner test;

    test_constructor(test);
    test_operator_read(test);
    test_operator_write(test);
    test_matrix_vector_multiplication(test);
    test_matrix_vector_multiplication_dense(test);
    test_matrix_matrix_multiplication(test);
    test_matrix_matrix_multiplication_identity(test);
    test_scalar_multiplication(test);
    test_has_entry(test);
    test_resize(test);
    test_clear(test);
    test_transpose(test);
    test_transpose_multiply(test);
    test_row_capacity_limits(test);
    test_large_sparse_matrix(test);

    test.print_summary();

    

    return 0;
}
