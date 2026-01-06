#include <iostream>
#include <cmath>
#include <algorithm>

//double* matrix = new double[r*c];

const int n = 3;
const double eps = 1e-10;

double A[n][n] = {
    {eps, 1.0, 1.0},
    {2.0, 3.0, 3.0},
    {1.0, 7.0, eps}
};

double A_copy[n][n] = {
    {eps, 1.0, 1.0},
    {2.0, 3.0, 3.0},
    {1.0, 7.0, eps}
};

double b[n] = {0.0, 1.0, 7.0};
double b_copy[n] = {0.0, 1.0, 7.0};

double x[n];

void LRZerlegung(bool pivotisierung) {
    int P[n];
    for (int i = 0; i < n; ++i) P[i] = i + 1; // Pivotvektor

    // Zerlegung in L und R 
    for (int k = 0; k < n; ++k) { // Schleife über Spalten
        if (pivotisierung) { // Pivotisierung nach jeder Eliminationsstufe.
            int pivotRow = k;
            double maxVal = std::fabs(A[k][k]);
            for (int r = k + 1; r < n; ++r) {
                double v = std::fabs(A[r][k]);
                if (v > maxVal) { maxVal = v; pivotRow = r; }
            }
            if (pivotRow != k) {
                for (int j = 0; j < n; ++j) std::swap(A[k][j], A[pivotRow][j]);
                std::swap(b[k], b[pivotRow]);
                std::swap(P[k], P[pivotRow]);
            }
        }
        if (std::fabs(A[k][k]) < eps) {
            std::cerr << "Null- oder Fast-Nullpivot bei k=" << k << std::endl;
            return;
        }
        for (int i = k + 1; i < n; ++i) { // Schleife über Zeilen
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; ++j) { // Aktualisierung der Restmatrix
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }

    // Vorwärtseinsetzen (Ly = b), b enthält bereits permutierte Werte
    for (int k = 1; k < n; ++k) {
        for (int i = 0; i < k; ++i) { // Schleife über vorherige Zeilen
            b[k] -= A[k][i] * b[i];
        }
    }
    // Rückwärtseinsetzen (Rx = y)
    for (int k = n - 1; k >= 0; --k) {
        for (int i = k + 1; i < n; ++i) { // Schleife über nachfolgende Zeilen
            b[k] -= A[k][i] * x[i];
        }
        x[k] = b[k] / A[k][k];
    }

    // Ausgabe der Lösung
    for (int i = n - 1; i >= 0; --i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
    // Pivotvektor ausgeben
    if (pivotisierung) {
        std::cout << "Pivotvektor: ";
        for (int i = 0; i < n; ++i) std::cout << P[i] << (i+1<n ? " " : "\n");
    }

    // Überprüfung der Lösung Ax = b
    double residual[n] = {0.0};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            residual[i] += A_copy[i][j] * x[j];
        }
        residual[i] -= b_copy[i];
    }
    std::cout << "Residual: ";
    for (int i = 0; i < n; ++i) {
        std::cout << residual[i] << " ";
    }
}

int main() {
    LRZerlegung(true);
    return 0;
}
