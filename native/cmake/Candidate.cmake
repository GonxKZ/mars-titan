option(MARS_TITAN_BUILD_CANDIDATE "Compilar el cálculo puro del candidato con LibTorch" OFF)
if(NOT MARS_TITAN_BUILD_CANDIDATE)
    return()
endif()

if(MARS_TITAN_SANITIZER STREQUAL "thread" OR MARS_TITAN_SANITIZER STREQUAL "memory")
    message(FATAL_ERROR "El candidato necesita LibTorch instrumentado para TSan o MSan")
endif()
include(CTest)
include(cmake/TorchDependencies.cmake)
mars_titan_find_torch()
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(mars_titan_candidate STATIC src/candidate.cpp src/candidate_archive.cpp src/candidate_identity.cpp)
target_include_directories(mars_titan_candidate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_candidate PUBLIC mars_titan::torch PRIVATE OpenSSL::Crypto)
mars_titan_configure_target(mars_titan_candidate)
add_executable(mars-titan-candidate src/candidate_main.cpp)
target_link_libraries(mars-titan-candidate PRIVATE mars_titan_candidate)
mars_titan_configure_target(mars-titan-candidate)
if(BUILD_TESTING)
    add_test(NAME candidate_cli COMMAND mars-titan-candidate cpu)
    add_test(NAME candidate_cli_invalid COMMAND mars-titan-candidate cuda:1)
    set_tests_properties(candidate_cli_invalid PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(candidate_cli candidate_cli_invalid PROPERTIES TIMEOUT 120 LABELS "integration;candidate")
    mars_titan_sanitizer_test_environment(candidate_cli candidate_cli_invalid)
    set_property(TEST candidate_cli candidate_cli_invalid APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_executable(candidate_contract_tests tests/candidate_contract_tests.cpp)
    target_link_libraries(candidate_contract_tests PRIVATE mars_titan_candidate)
    mars_titan_configure_target(candidate_contract_tests)
    add_test(NAME candidate_contract COMMAND candidate_contract_tests)
    set_tests_properties(candidate_contract PROPERTIES TIMEOUT 120 LABELS "unit;integration;candidate")
    mars_titan_sanitizer_test_environment(candidate_contract)
    set_property(TEST candidate_contract APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_executable(candidate_tests tests/candidate_tests.cpp)
    target_link_libraries(candidate_tests PRIVATE mars_titan_candidate)
    mars_titan_configure_target(candidate_tests)
    add_test(NAME candidate COMMAND candidate_tests)
    set_tests_properties(candidate PROPERTIES TIMEOUT 120 LABELS "unit;integration;candidate")
    mars_titan_sanitizer_test_environment(candidate)
    set_property(TEST candidate APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
endif()

if(MARS_TITAN_BUILD_FUZZER)
    add_executable(fuzz_candidate tests/fuzz_candidate.cpp)
    target_link_libraries(fuzz_candidate PRIVATE mars_titan_candidate)
    mars_titan_configure_target(fuzz_candidate)
    target_compile_options(mars_titan_candidate PRIVATE -fsanitize=fuzzer-no-link)
    target_compile_options(fuzz_candidate PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(fuzz_candidate PRIVATE -fsanitize=fuzzer)
    if(BUILD_TESTING)
        add_test(NAME fuzz_candidate_smoke COMMAND fuzz_candidate
            -seed=42 -runs=1000 -max_len=32 -timeout=5 -rss_limit_mb=2048)
        set_tests_properties(fuzz_candidate_smoke PROPERTIES TIMEOUT 120 LABELS "fuzz;candidate")
        mars_titan_sanitizer_test_environment(fuzz_candidate_smoke)
        set_property(TEST fuzz_candidate_smoke APPEND PROPERTY ENVIRONMENT
            "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    endif()
endif()
if(MARS_TITAN_ENABLE_STATIC_ANALYZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_custom_target(candidate-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_archive.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_identity.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_main.cpp"
        VERBATIM)
endif()
