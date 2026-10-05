if(NOT MARS_TITAN_BUILD_FINANCIAL_CONTROLS)
    return()
endif()
if(NOT TARGET mars_titan_simulation_files)
    message(FATAL_ERROR "Las referencias financieras necesitan Arrow, OpenSSL y la infraestructura de archivos")
endif()

add_library(mars_titan_financial_controls STATIC src/financial_controls.cpp)
target_include_directories(mars_titan_financial_controls PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_financial_controls PUBLIC mars_titan_simulation_files)
mars_titan_configure_target(mars_titan_financial_controls)
add_executable(mars-titan-financial-controls src/financial_controls_main.cpp)
target_link_libraries(mars-titan-financial-controls PRIVATE mars_titan_financial_controls)
mars_titan_configure_target(mars-titan-financial-controls)
list(APPEND mars_analysis_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_controls.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_controls_main.cpp")

if(BUILD_TESTING)
    add_executable(financial_controls_tests tests/financial_controls_tests.cpp)
    target_link_libraries(financial_controls_tests PRIVATE mars_titan_financial_controls)
    mars_titan_configure_target(financial_controls_tests)
    add_test(NAME financial_controls COMMAND financial_controls_tests)
    add_test(NAME financial_controls_help COMMAND mars-titan-financial-controls --help)
    add_test(NAME financial_controls_invalid COMMAND mars-titan-financial-controls)
    set_tests_properties(financial_controls_invalid PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(financial_controls financial_controls_help financial_controls_invalid
        PROPERTIES TIMEOUT 60 LABELS "unit;integration;financial;controls")
    mars_titan_sanitizer_test_environment(
        financial_controls financial_controls_help financial_controls_invalid)
endif()

if(MARS_TITAN_ENABLE_STATIC_ANALYZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_custom_target(financial-controls-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_controls.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_controls_main.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/financial_controls_tests.cpp"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        VERBATIM)
endif()
