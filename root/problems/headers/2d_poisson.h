#include "data structures/headers/vector.h"
#include "data structures/headers/matrix.h"
#include "data structures/headers/sparse_matrix.h"

/**
 * @brief Create a 2D Poisson system matrix and right-hand side vector
 * 
 * @tparam TMatrix The type of the system matrix
 * @param m The system matrix to be created
 * @param rhs The right-hand side vector to be created
 * @param numElemsPerDim The number of interior elements per dimension
 * @param eps_x The diffusion coefficient in x direction (default 1.0)
 * @param eps_y The diffusion coefficient in y direction (default 1.0)
 * The diffusion coefficients allow to create anisotropic Poisson problems.
 */
template <typename TMatrix>
void create2dPoissonSystem
        (
                TMatrix& m,
                Vector& rhs,
                std::size_t numElemsPerDim,
                double eps_x = 1.0,
                double eps_y = 1.0
        );