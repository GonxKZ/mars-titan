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
include(cmake/CandidateDependencies.cmake)
mars_titan_find_candidate_zip()
add_library(mars_titan_candidate STATIC src/candidate.cpp src/candidate_archive.cpp src/candidate_identity.cpp src/candidate_codec.cpp)
target_include_directories(mars_titan_candidate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Resolver primero el lector privado, sin enlazar sus símbolos contra la copia de LibTorch.
target_link_libraries(mars_titan_candidate PRIVATE miniz PUBLIC mars_titan::torch PRIVATE OpenSSL::Crypto)
mars_titan_configure_target(mars_titan_candidate)
if(TARGET _episodic_native)
    # El caster comparte tensores y autograd con el mismo SDK, sin JSON ni copias de NumPy.
    get_target_property(candidate_torch_root mars_titan_torch MARS_TITAN_TORCH_ROOT)
    set(candidate_python_library "${candidate_torch_root}/lib/libtorch_python.so")
    if(NOT EXISTS "${candidate_python_library}")
        message(FATAL_ERROR "Falta libtorch_python en el SDK del enlace tensorial candidato")
    endif()
    target_sources(_episodic_native PRIVATE src/candidate_python.cpp)
    target_compile_definitions(_episodic_native PRIVATE MARS_TITAN_CANDIDATE_PYTHON=1)
    target_link_libraries(_episodic_native PRIVATE mars_titan_candidate "${candidate_python_library}")
endif()
add_executable(mars-titan-candidate src/candidate_main.cpp)
target_link_libraries(mars-titan-candidate PRIVATE mars_titan_candidate)
mars_titan_configure_target(mars-titan-candidate)
if(BUILD_TESTING)
    add_executable(candidate_codec_tests tests/candidate_codec_tests.cpp)
    target_link_libraries(candidate_codec_tests PRIVATE mars_titan_candidate)
    mars_titan_configure_target(candidate_codec_tests)
    add_test(NAME candidate_codec COMMAND candidate_codec_tests)
    set_tests_properties(candidate_codec PROPERTIES TIMEOUT 120 LABELS "unit;candidate;codec")
    mars_titan_sanitizer_test_environment(candidate_codec)
    set_property(TEST candidate_codec APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=2" "MKL_NUM_THREADS=2")
    add_executable(candidate_historical_tests tests/candidate_historical_tests.cpp)
    target_link_libraries(candidate_historical_tests PRIVATE mars_titan_candidate)
    mars_titan_configure_target(candidate_historical_tests)
    add_test(NAME candidate_historical COMMAND candidate_historical_tests)
    set_tests_properties(candidate_historical PROPERTIES TIMEOUT 120 LABELS "unit;integration;candidate")
    mars_titan_sanitizer_test_environment(candidate_historical)
    set_property(TEST candidate_historical APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_executable(candidate_archive_tests tests/candidate_archive_tests.cpp)
    target_link_libraries(candidate_archive_tests PRIVATE mars_titan_candidate)
    mars_titan_configure_target(candidate_archive_tests)
    get_filename_component(candidate_project "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    add_test(NAME candidate_archives COMMAND "${MARS_TITAN_UV}" run --no-sync
        --project "${candidate_project}" python "${CMAKE_CURRENT_SOURCE_DIR}/tests/candidate_archives.py"
        "$<TARGET_FILE:mars-titan-candidate>" "$<TARGET_FILE:candidate_archive_tests>"
        "${CMAKE_CURRENT_BINARY_DIR}/candidate-archive-fixtures")
    set_tests_properties(candidate_archives PROPERTIES TIMEOUT 120 LABELS "integration;candidate;serialization")
    mars_titan_sanitizer_test_environment(candidate_archives)
    set(candidate_test_environment "${MARS_TITAN_TORCH_ENVIRONMENT}")
    if(NOT candidate_test_environment)
        set(candidate_test_environment "${candidate_project}/.venv")
    endif()
    set_property(TEST candidate_archives APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1"
        "UV_PROJECT_ENVIRONMENT=${candidate_test_environment}")
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
    if(MARS_TITAN_TORCH_HAS_CUDA)
        add_executable(candidate_cuda_tests tests/candidate_cuda_tests.cpp)
        target_link_libraries(candidate_cuda_tests PRIVATE mars_titan_candidate)
        mars_titan_configure_target(candidate_cuda_tests)
        add_test(NAME candidate_cuda COMMAND candidate_cuda_tests)
        set_tests_properties(candidate_cuda PROPERTIES TIMEOUT 180 LABELS "unit;integration;candidate;cuda")
        mars_titan_sanitizer_test_environment(candidate_cuda)
        set_property(TEST candidate_cuda APPEND PROPERTY ENVIRONMENT
            "CUDA_VISIBLE_DEVICES=0" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1"
            "CUBLAS_WORKSPACE_CONFIG=:4096:8" "PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.0625")
    endif()
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
    set(candidate_analysis_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_archive.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_identity.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_codec.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_main.cpp")
    if(TARGET _episodic_native)
        list(APPEND candidate_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/candidate_python.cpp")
    endif()
    add_custom_target(candidate-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            ${candidate_analysis_sources}
        VERBATIM)
endif()
