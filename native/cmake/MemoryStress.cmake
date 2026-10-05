option(MARS_TITAN_BUILD_MEMORY_STRESS "Compilar escenarios controlados de memoria y ruido" OFF)
option(MARS_TITAN_MEMORY_STRESS_FUZZ "Preparar libFuzzer para checkpoints de memoria" OFF)
if(NOT MARS_TITAN_BUILD_MEMORY_STRESS)
    return()
endif()
if(MARS_TITAN_SANITIZER STREQUAL "memory" OR MARS_TITAN_SANITIZER STREQUAL "thread")
    message(FATAL_ERROR "El control de memoria requiere LibTorch instrumentado para MSan o TSan")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/TorchDependencies.cmake")
mars_titan_find_torch()
if(NOT TARGET nlohmann_json::nlohmann_json)
    include("${CMAKE_CURRENT_LIST_DIR}/RunnerDependencies.cmake")
    mars_titan_find_json()
endif()
add_library(mars_titan_memory_stress STATIC src/memory_retention.cpp src/memory_stress.cpp)
target_include_directories(mars_titan_memory_stress PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_memory_stress PUBLIC mars_titan::torch nlohmann_json::nlohmann_json)
mars_titan_configure_target(mars_titan_memory_stress)
set(stress_identity_files
    include/mars_titan/memory_stress.hpp include/mars_titan/episodic_memory.hpp
    src/memory_retention.cpp src/memory_stress.cpp src/memory_stress_internal.hpp
    src/memory_stress_main.cpp cmake/MemoryStress.cmake cmake/Diagnostics.cmake
    cmake/Instrumentation.cmake cmake/TorchDependencies.cmake)
set(stress_identity "")
foreach(source IN LISTS stress_identity_files)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_hash)
    string(APPEND stress_identity "${source}:${source_hash}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
endforeach()
string(SHA256 stress_source_hash "${stress_identity}")
get_property(stress_torch_version TARGET mars_titan_torch PROPERTY MARS_TITAN_TORCH_VERSION)
target_compile_definitions(mars_titan_memory_stress PRIVATE
    "MARS_TITAN_STRESS_SOURCE_SHA256=\"${stress_source_hash}\""
    "MARS_TITAN_STRESS_TORCH_VERSION=\"${stress_torch_version}\"")
add_executable(mars-titan-memory-stress src/memory_stress_main.cpp)
target_link_libraries(mars-titan-memory-stress PRIVATE mars_titan_memory_stress)
mars_titan_configure_target(mars-titan-memory-stress)
include(CTest)
if(BUILD_TESTING)
    add_executable(memory_stress_tests tests/memory_stress_tests.cpp)
    target_link_libraries(memory_stress_tests PRIVATE mars_titan_memory_stress)
    mars_titan_configure_target(memory_stress_tests)
    add_test(NAME memory_stress COMMAND memory_stress_tests)
    set_tests_properties(memory_stress PROPERTIES TIMEOUT 120 LABELS "unit;integration;memory")
    mars_titan_sanitizer_test_environment(memory_stress)
    set_property(TEST memory_stress APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_test(NAME memory_stress_cli COMMAND "${CMAKE_COMMAND}"
        "-DPROGRAM=$<TARGET_FILE:mars-titan-memory-stress>"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/memory-stress-cli"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/memory_stress_cli.cmake")
    set_tests_properties(memory_stress_cli PROPERTIES TIMEOUT 120 LABELS "integration;memory;cli")
    mars_titan_sanitizer_test_environment(memory_stress_cli)
    set_property(TEST memory_stress_cli APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
endif()
if(MARS_TITAN_ENABLE_COVERAGE AND TARGET memory_stress_tests)
    add_custom_target(memory-stress-coverage-report
        COMMAND "${CMAKE_COMMAND}"
            "-DPROFILE_DIR=${MARS_TITAN_COVERAGE_DIR}"
            "-DPROFDATA=${MARS_TITAN_LLVM_PROFDATA}" "-DCOV=${MARS_TITAN_LLVM_COV}"
            "-DOBJECTS=$<TARGET_FILE:memory_stress_tests>;$<TARGET_FILE:mars-titan-memory-stress>"
            "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
            -P "${CMAKE_CURRENT_LIST_DIR}/ReportCoverage.cmake"
        DEPENDS memory_stress_tests mars-titan-memory-stress VERBATIM)
endif()
if(MARS_TITAN_ENABLE_STATIC_ANALYZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_custom_target(memory-stress-static-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_retention.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_stress.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_stress_main.cpp"
        VERBATIM)
endif()

if(MARS_TITAN_MEMORY_STRESS_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       NOT MARS_TITAN_SANITIZER STREQUAL "address-undefined")
        message(FATAL_ERROR "El fuzzing de memoria requiere Clang y ASan/UBSan")
    endif()
    target_compile_options(mars_titan_memory_stress PRIVATE -fsanitize=fuzzer-no-link)
    add_executable(memory_stress_fuzz tests/fuzz_memory_stress.cpp)
    target_link_libraries(memory_stress_fuzz PRIVATE mars_titan_memory_stress)
    mars_titan_configure_target(memory_stress_fuzz)
    target_compile_options(memory_stress_fuzz PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(memory_stress_fuzz PRIVATE -fsanitize=fuzzer)
endif()
