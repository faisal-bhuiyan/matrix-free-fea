include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(matrix-free-fea_unit_tests
    test_types.cpp
    test_quadrature.cpp
    test_shape_functions.cpp
    test_geometry.cpp
    test_mesh.cpp
    test_element_kernel.cpp
    test_assembly.cpp
)
target_link_libraries(matrix-free-fea_unit_tests PRIVATE GTest::gtest_main matrix-free-fea)
target_include_directories(matrix-free-fea_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(matrix-free-fea_unit_tests)

add_executable(matrix-free-fea_validation_tests
    test_validation.cpp
)
target_link_libraries(matrix-free-fea_validation_tests PRIVATE GTest::gtest_main matrix-free-fea)
target_include_directories(matrix-free-fea_validation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(matrix-free-fea_validation_tests)
