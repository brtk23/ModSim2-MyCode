#include <iostream>
#include <cassert>
#include <cmath>
#include "../data structures/headers/vector.h"

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
            std::cout << GREEN << "✓ PASS: " << name << RESET << std::endl;
        } else {
            failed++;
            std::cout << RED << "✗ FAIL: " << name << RESET;
            if (!message.empty()) {
                std::cout << " (" << message << ")";
            }
            std::cout << std::endl;
        }
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

void test_constructor() {
    std::cout << "\n" << YELLOW << "Testing Vector Constructor..." << RESET << std::endl;
    TestRunner test;

    Vector v1(5, 0.0);
    test.test("Constructor - size", v1.size() == 5);

    Vector v2(3, 2.5);
    test.test("Constructor - initialization", v2[0] == 2.5 && v2[1] == 2.5 && v2[2] == 2.5);
}

void test_element_access() {
    std::cout << "\n" << YELLOW << "Testing Vector Element Access..." << RESET << std::endl;
    TestRunner test;

    Vector v(4, 0.0);
    
    v[0] = 1.0;
    v[1] = 2.5;
    v[2] = -3.0;
    v[3] = 4.5;

    test.test("Element access - v[0]", v[0] == 1.0);
    test.test("Element access - v[1]", v[1] == 2.5);
    test.test("Element access - v[2]", v[2] == -3.0);
    test.test("Element access - v[3]", v[3] == 4.5);
}

void test_vector_addition() {
    std::cout << "\n" << YELLOW << "Testing Vector Addition..." << RESET << std::endl;
    TestRunner test;

    Vector v1(3, 0.0);
    v1[0] = 1.0;
    v1[1] = 2.0;
    v1[2] = 3.0;

    Vector v2(3, 0.0);
    v2[0] = 4.0;
    v2[1] = 5.0;
    v2[2] = 6.0;

    Vector result = v1 + v2;

    test.test("Vector addition - element 0", result[0] == 5.0);
    test.test("Vector addition - element 1", result[1] == 7.0);
    test.test("Vector addition - element 2", result[2] == 9.0);
}

void test_vector_subtraction() {
    std::cout << "\n" << YELLOW << "Testing Vector Subtraction..." << RESET << std::endl;
    TestRunner test;

    Vector v1(3, 0.0);
    v1[0] = 5.0;
    v1[1] = 7.0;
    v1[2] = 9.0;

    Vector v2(3, 0.0);
    v2[0] = 1.0;
    v2[1] = 2.0;
    v2[2] = 3.0;

    Vector result = v1 - v2;

    test.test("Vector subtraction - element 0", result[0] == 4.0);
    test.test("Vector subtraction - element 1", result[1] == 5.0);
    test.test("Vector subtraction - element 2", result[2] == 6.0);
}

void test_scalar_multiplication() {
    std::cout << "\n" << YELLOW << "Testing Scalar Multiplication..." << RESET << std::endl;
    TestRunner test;

    Vector v(3, 0.0);
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;

    Vector result = v * 2.0;

    test.test("Scalar multiplication - element 0", result[0] == 2.0);
    test.test("Scalar multiplication - element 1", result[1] == 4.0);
    test.test("Scalar multiplication - element 2", result[2] == 6.0);
}

void test_dot_product() {
    std::cout << "\n" << YELLOW << "Testing Dot Product..." << RESET << std::endl;
    TestRunner test;

    Vector v1(3, 0.0);
    v1[0] = 1.0;
    v1[1] = 2.0;
    v1[2] = 3.0;

    Vector v2(3, 0.0);
    v2[0] = 4.0;
    v2[1] = 5.0;
    v2[2] = 6.0;

    // dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    double result = v1 * v2;

    test.test("Dot product", std::fabs(result - 32.0) < 1e-10);
}

void test_norm() {
    std::cout << "\n" << YELLOW << "Testing Vector Norm..." << RESET << std::endl;
    TestRunner test;

    Vector v(3, 0.0);
    v[0] = 3.0;
    v[1] = 4.0;
    v[2] = 0.0;

    // norm: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    double norm = v.norm();

    test.test("Vector norm", std::fabs(norm - 5.0) < 1e-10);
}

void test_large_vector() {
    std::cout << "\n" << YELLOW << "Testing Large Vector..." << RESET << std::endl;
    TestRunner test;

    size_t n = 10000;
    Vector v1(n, 1.0);
    Vector v2(n, 2.0);

    // dot product: sum of 1*2 for all n elements = 2*n
    double dot = v1 * v2;
    test.test("Large vector dot product", std::fabs(dot - 2.0 * n) < 1e-8);

    Vector sum = v1 + v2;
    test.test("Large vector addition - element 0", sum[0] == 3.0);
    test.test("Large vector addition - element 5000", sum[5000] == 3.0);
}

int main() {
    std::cout << "\n" << YELLOW << "====================================" << RESET << std::endl;
    std::cout << YELLOW << "  VECTOR UNIT TESTS" << RESET << std::endl;
    std::cout << YELLOW << "====================================" << RESET << std::endl;

    test_constructor();
    test_element_access();
    test_vector_addition();
    test_vector_subtraction();
    test_scalar_multiplication();
    test_dot_product();
    test_norm();
    test_large_vector();

    return 0;
}
