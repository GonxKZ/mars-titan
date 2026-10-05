if(NOT MARS_TITAN_BUILD_COHORT_EXECUTION)
    return()
endif()

if(NOT TARGET mars_titan_simulation_files)
    message(FATAL_ERROR "El ejecutor de cohortes necesita las utilidades nativas de archivos")
endif()

add_library(mars_titan_cohort_execution STATIC src/cohort_execution.cpp)
target_include_directories(mars_titan_cohort_execution PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_cohort_execution PUBLIC mars_titan_simulation_files)
mars_titan_configure_target(mars_titan_cohort_execution)
add_executable(mars-titan-cohorts src/cohort_main.cpp)
target_link_libraries(mars-titan-cohorts PRIVATE mars_titan_cohort_execution)
mars_titan_configure_target(mars-titan-cohorts)
list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/cohort_execution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cohort_main.cpp")
if(BUILD_TESTING)
    add_executable(cohort_execution_tests tests/cohort_execution_tests.cpp)
    target_link_libraries(cohort_execution_tests PRIVATE mars_titan_cohort_execution)
    mars_titan_configure_target(cohort_execution_tests)
    add_test(NAME cohort_execution COMMAND cohort_execution_tests)
    set_tests_properties(cohort_execution PROPERTIES TIMEOUT 120 LABELS "unit;integration;cohorts")
    mars_titan_sanitizer_test_environment(cohort_execution)
    add_test(NAME cohort_cli COMMAND "${CMAKE_COMMAND}"
        "-DCOHORT_EXECUTABLE=$<TARGET_FILE:mars-titan-cohorts>"
        "-DCOHORT_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cohort_cli.cmake")
    set_tests_properties(cohort_cli PROPERTIES TIMEOUT 120 LABELS "integration;cohorts")
    mars_titan_sanitizer_test_environment(cohort_cli)
endif()

if(MARS_TITAN_BUILD_FUZZER)
    add_executable(fuzz_cohort_execution tests/fuzz_cohort_execution.cpp)
    target_link_libraries(fuzz_cohort_execution PRIVATE mars_titan_cohort_execution)
    mars_titan_configure_target(fuzz_cohort_execution)
    target_compile_options(mars_titan_cohort_execution PRIVATE -fsanitize=fuzzer-no-link)
    target_compile_options(fuzz_cohort_execution PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(fuzz_cohort_execution PRIVATE -fsanitize=fuzzer)
    configure_file(tests/fixtures/cohort.json "${CMAKE_CURRENT_BINARY_DIR}/cohort-fuzz-corpus/cohort.json" COPYONLY)
    if(BUILD_TESTING)
        add_test(NAME cohort_fuzz_smoke COMMAND fuzz_cohort_execution
            "${CMAKE_CURRENT_BINARY_DIR}/cohort-fuzz-corpus"
            -seed=42 -runs=1000 -max_len=4096 -timeout=5 -rss_limit_mb=1024)
        set_tests_properties(cohort_fuzz_smoke PROPERTIES TIMEOUT 120 LABELS "fuzz;cohorts")
        mars_titan_sanitizer_test_environment(cohort_fuzz_smoke)
    endif()
endif()

if(MARS_TITAN_ENABLE_COVERAGE AND TARGET cohort_execution_tests)
    add_custom_target(cohort-coverage-report
        COMMAND "${CMAKE_COMMAND}" "-DPROFILE_DIR=${MARS_TITAN_COVERAGE_DIR}"
            "-DPROFDATA=${MARS_TITAN_LLVM_PROFDATA}" "-DCOV=${MARS_TITAN_LLVM_COV}"
            "-DOBJECTS=$<TARGET_FILE:cohort_execution_tests>;$<TARGET_FILE:mars-titan-cohorts>"
            "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReportCoverage.cmake"
        DEPENDS cohort_execution_tests mars-titan-cohorts
        COMMENT "Combinar la cobertura del ejecutor y su CLI" VERBATIM)
endif()
