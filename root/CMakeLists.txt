# Specify the minimum CMake version required to build this project
cmake_minimum_required(VERSION 3.10)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_C_COMPILER "gcc" )

# Define the project name as "solver"
project(ModSim2_Praktikum)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Require the specified C++ standard (fail if compiler doesn't support it)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -mtune=native -ffast-math -funroll-loops)
endif()

# Option to disable OpenMP (default is ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)

# Find and configure OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
else()
    # Add -fno-openmp flag to ignore OpenMP pragmas
    add_compile_options(-fno-openmp)
endif()

# Create an executable named "solver" from the listed source files
add_executable(ModSim2_Praktikum
    main.cpp
    "data structures/matrix.cpp"
    "data structures/sparse_matrix.cpp"
    "data structures/vector.cpp"
    "solvers/jacobi.cpp"
    "solvers/gauss_seidel.cpp"
    "solvers/iterative_solver.cpp"
    "solvers/lu_solver.cpp"
    "solvers/multigrid_solver.cpp"
    "solvers/ilu.cpp"
    "problems/2d_poisson.cpp"
    "problems/2d_poisson_settings.cpp"
)

# Add the current source directory to the executable's include search path (for #include "...")
target_include_directories(ModSim2_Praktikum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link OpenMP to main executable
if(USE_OPENMP)
    target_link_libraries(ModSim2_Praktikum PRIVATE OpenMP::OpenMP_CXX)
endif()

# Build unit tests
add_executable(test_sparse_matrix
    tests/test_sparse_matrix.cpp
    "data structures/sparse_matrix.cpp"
    "data structures/vector.cpp"
    "data structures/matrix.cpp"
)

add_executable(test_vector
    tests/test_vector.cpp
    "data structures/vector.cpp"
)

# Add the current source directory to the test executables' include search path
target_include_directories(test_sparse_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_vector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link OpenMP to test executables
if(USE_OPENMP)
    target_link_libraries(test_sparse_matrix PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_vector PRIVATE OpenMP::OpenMP_CXX)
endif()


# reminder for me:
# SETUP USING 
#cd 'c:\Users\mia-l\Documents\code\Uni\ModSim 2\My code\root';  
#cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -S . -B ../build
# FOR DEBUG SETUP USING
#cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DUSE_OPENMP=ON -S . -B ../build  

# BUILD USING
#cmake --build ../build