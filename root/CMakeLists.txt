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

# Create an executable named "solver" from the listed source files
add_executable(ModSim2_Praktikum
    main03.cpp
    "data structures/matrix.cpp"
    "data structures/sparse_matrix.cpp"
    "data structures/vector.cpp"
    "solvers/jacobi.cpp"
    "solvers/gauss_seidel.cpp"
    "solvers/iterative_solver.cpp"
    "solvers/lu_solver.cpp"
    "solvers/multigrid_solver.cpp"
)

# Add the current source directory to the executable's include search path (for #include "...")
target_include_directories(ModSim2_Praktikum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})



# SETUP USING 
#cd 'c:\Users\mia-l\Documents\code\Uni\ModSim 2\My code\root';  
#cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -S . -B build
# FOR DEBUG SETUP USING
#cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -S . -B build  

# BUILD USING
#cmake --build build