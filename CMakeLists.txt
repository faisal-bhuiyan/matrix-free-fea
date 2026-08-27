cmake_minimum_required(VERSION 3.20)

project(MatrixFreeFEA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(matrix-free-fea INTERFACE)
target_include_directories(matrix-free-fea INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(matrix-free-fea INTERFACE Eigen3::Eigen)

include(CTest)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
