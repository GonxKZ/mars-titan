cmake_minimum_required(VERSION 3.24)

project(mars_titan_native VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MARS_TITAN_ENABLE_CUDA "Configurar las herramientas de CUDA" OFF)
option(MARS_TITAN_WARNINGS_AS_ERRORS "Tratar avisos propios como errores" OFF)
option(MARS_TITAN_ENABLE_CLANG_TIDY "Analizar los objetivos propios con clang-tidy" OFF)
option(MARS_TITAN_ENABLE_STATIC_ANALYZER "Preparar el análisis estático de rutas" OFF)
option(MARS_TITAN_ENABLE_LIFETIME "Activar el análisis de lifetime compatible" ON)
option(MARS_TITAN_PROFILE "Conservar información para perfiles de rendimiento" OFF)
option(MARS_TITAN_ENABLE_IPO "Activar optimización entre unidades en Release" OFF)
option(MARS_TITAN_ENABLE_COVERAGE "Instrumentar cobertura de líneas y ramas con Clang" OFF)
option(MARS_TITAN_BUILD_SIMULATION "Compilar la biblioteca de simulación" ON)
option(MARS_TITAN_BUILD_FINANCIAL "Compilar la sesión financiera pura y sus pruebas" OFF)
option(MARS_TITAN_BUILD_RUNNER "Compilar el ejecutable financiero autónomo" OFF)
option(MARS_TITAN_BUILD_FUZZER "Compilar el ejecutable de libFuzzer" OFF)
set(MARS_TITAN_SANITIZER "none" CACHE STRING "Instrumentación independiente de memoria o hilos")
set_property(CACHE MARS_TITAN_SANITIZER PROPERTY STRINGS none address address-undefined thread memory)
set(MARS_TITAN_MSAN_STDLIB_ROOT "" CACHE PATH "Raíz de libc++ y libc++abi instrumentadas con MSan")
set(MARS_TITAN_PGO "off" CACHE STRING "Perfil dirigido: off, generate o use")
set_property(CACHE MARS_TITAN_PGO PROPERTY STRINGS off generate use)
set(MARS_TITAN_PGO_DATA "${CMAKE_BINARY_DIR}/profiles" CACHE PATH "Directorio GCC o perfil Clang de PGO")

add_library(mars_titan_native_options INTERFACE)
add_library(mars_titan::native_options ALIAS mars_titan_native_options)
target_compile_features(mars_titan_native_options INTERFACE cxx_std_20)

include(cmake/Instrumentation.cmake)
include(cmake/Diagnostics.cmake)
include(cmake/Coverage.cmake)

if(MARS_TITAN_BUILD_RUNNER)
    set(MARS_TITAN_BUILD_FINANCIAL ON)
endif()
if(MARS_TITAN_BUILD_FINANCIAL AND NOT MARS_TITAN_BUILD_SIMULATION)
    message(FATAL_ERROR "La sesión financiera necesita el núcleo de simulación")
endif()

function(mars_titan_configure_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "No existe el objetivo nativo solicitado")
    endif()
    target_link_libraries(${target} PRIVATE mars_titan::native_options)
    set_target_properties(${target} PROPERTIES
        C_STANDARD 17 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        EXPORT_COMPILE_COMMANDS ON
    )
    mars_titan_instrument_target(${target})
    mars_titan_cover_target(${target})
    if(MARS_TITAN_ENABLE_CLANG_TIDY)
        set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY
            "${MARS_TITAN_CLANG_TIDY};--config-file=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.clang-tidy")
    endif()
endfunction()

if(MARS_TITAN_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_features(mars_titan_native_options INTERFACE cuda_std_20)
    message(STATUS "CUDA configurada con las herramientas ${CUDAToolkit_VERSION}")
endif()

if(MARS_TITAN_BUILD_SIMULATION)
    add_library(mars_titan_simulation SHARED src/simulation.cpp)
    target_include_directories(mars_titan_simulation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set_target_properties(mars_titan_simulation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    mars_titan_configure_target(mars_titan_simulation)
    set(mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cpp")

    include(CTest)
    if(BUILD_TESTING)
        find_package(Threads REQUIRED)
        add_executable(simulation_tests tests/simulation_tests.cpp)
        target_link_libraries(simulation_tests PRIVATE mars_titan_simulation)
        mars_titan_configure_target(simulation_tests)
        add_executable(threaded_simulation tests/threaded_simulation.cpp)
        target_link_libraries(threaded_simulation PRIVATE mars_titan_simulation Threads::Threads)
        mars_titan_configure_target(threaded_simulation)
        add_executable(c_abi_test tests/c_abi_test.c)
        target_link_libraries(c_abi_test PRIVATE mars_titan_simulation)
        mars_titan_configure_target(c_abi_test)
        set_target_properties(c_abi_test PROPERTIES LINKER_LANGUAGE C)
        add_test(NAME simulation COMMAND simulation_tests)
        add_test(NAME concurrency COMMAND threaded_simulation)
        add_test(NAME c_abi COMMAND c_abi_test)
        set_tests_properties(simulation concurrency c_abi PROPERTIES TIMEOUT 60)
        set_tests_properties(simulation PROPERTIES LABELS "unit;integration")
        set_tests_properties(concurrency PROPERTIES LABELS "concurrency")
        set_tests_properties(c_abi PROPERTIES LABELS "integration;c17")
        mars_titan_sanitizer_test_environment(simulation concurrency c_abi)
    endif()

    if(MARS_TITAN_BUILD_FINANCIAL)
        add_library(mars_titan_financial STATIC src/financial_session.cpp)
        target_include_directories(mars_titan_financial PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_financial PUBLIC mars_titan_simulation)
        mars_titan_configure_target(mars_titan_financial)
        list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_session.cpp")
        if(BUILD_TESTING)
            add_executable(financial_session_tests tests/financial_session_tests.cpp)
            target_link_libraries(financial_session_tests PRIVATE mars_titan_financial)
            mars_titan_configure_target(financial_session_tests)
            add_test(NAME financial_session COMMAND financial_session_tests)
            set_tests_properties(financial_session PROPERTIES TIMEOUT 60 LABELS "unit;integration;financial")
            mars_titan_sanitizer_test_environment(financial_session)
        endif()
    endif()

    if(MARS_TITAN_BUILD_RUNNER)
        if(MARS_TITAN_SANITIZER STREQUAL "memory")
            message(STATUS "MSan verifica núcleo y sesión pura. El ejecutable con Arrow/OpenSSL precompilados queda excluido")
        elseif(NOT UNIX)
            message(FATAL_ERROR "El ejecutable financiero requiere UNIX. Las bibliotecas se pueden compilar por separado")
        else()
            include(cmake/RunnerDependencies.cmake)
            mars_titan_find_json()
            mars_titan_find_arrow()
            find_package(OpenSSL REQUIRED COMPONENTS Crypto)
            find_package(Threads REQUIRED)
            add_executable(mars-titan-sim src/simulation_main.cpp src/financial_parquet.cpp src/simulation_files.cpp)
            target_link_libraries(mars-titan-sim PRIVATE mars_titan_financial mars_titan::parquet
                OpenSSL::Crypto Threads::Threads nlohmann_json::nlohmann_json)
            mars_titan_configure_target(mars-titan-sim)
            list(APPEND mars_analysis_sources
                "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation_main.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_parquet.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation_files.cpp")
        endif()
    endif()

    if(MARS_TITAN_BUILD_FINANCIAL)
        set(identity_files ${mars_analysis_sources}
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/simulation.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/financial_session.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/accurate_sum.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Diagnostics.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Instrumentation.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Coverage.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildIdentity.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunnerDependencies.cmake")
        if(TARGET mars-titan-sim)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/simulation_files.hpp")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${identity_files})
        set(identity_content "")
        foreach(source IN LISTS identity_files)
            file(SHA256 "${source}" source_hash)
            file(RELATIVE_PATH relative_source "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
            string(APPEND identity_content "${relative_source}:${source_hash}\n")
        endforeach()
        string(SHA256 native_source_hash "${identity_content}")
        target_compile_definitions(mars_titan_financial PUBLIC
            "MARS_TITAN_NATIVE_VERSION=\"${PROJECT_VERSION}\""
            "MARS_TITAN_NATIVE_SOURCE_SHA256=\"${native_source_hash}\"")
    endif()

    if(MARS_TITAN_BUILD_FUZZER)
        add_executable(fuzz_simulation tests/fuzz_simulation.cpp)
        target_link_libraries(fuzz_simulation PRIVATE mars_titan_simulation)
        mars_titan_configure_target(fuzz_simulation)
        target_compile_options(mars_titan_simulation PRIVATE -fsanitize=fuzzer-no-link)
        target_compile_options(fuzz_simulation PRIVATE -fsanitize=fuzzer-no-link)
        target_link_options(fuzz_simulation PRIVATE -fsanitize=fuzzer)
        if(BUILD_TESTING)
            add_test(NAME fuzz_smoke COMMAND fuzz_simulation
                -seed=42 -runs=1000 -max_len=4096 -timeout=5 -rss_limit_mb=2048)
            set_tests_properties(fuzz_smoke PROPERTIES TIMEOUT 60 LABELS "fuzz")
            mars_titan_sanitizer_test_environment(fuzz_smoke)
        endif()
        if(TARGET mars_titan_financial)
            add_executable(fuzz_session tests/fuzz_session.cpp)
            target_link_libraries(fuzz_session PRIVATE mars_titan_financial)
            mars_titan_configure_target(fuzz_session)
            target_compile_options(mars_titan_financial PRIVATE -fsanitize=fuzzer-no-link)
            target_compile_options(fuzz_session PRIVATE -fsanitize=fuzzer-no-link)
            target_link_options(fuzz_session PRIVATE -fsanitize=fuzzer)
            if(BUILD_TESTING)
                add_test(NAME fuzz_session_smoke COMMAND fuzz_session
                    -seed=42 -runs=1000 -max_len=4096 -timeout=5 -rss_limit_mb=2048)
                set_tests_properties(fuzz_session_smoke PROPERTIES TIMEOUT 60 LABELS "fuzz;financial")
                mars_titan_sanitizer_test_environment(fuzz_session_smoke)
            endif()
        endif()
    endif()

    if(MARS_TITAN_BUILD_FINANCIAL)
        include(cmake/BuildIdentity.cmake)
        mars_titan_build_identity("${native_source_hash}")
    endif()

    if(MARS_TITAN_ENABLE_STATIC_ANALYZER)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_custom_target(static-analysis
                COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze
                    "-p=${CMAKE_BINARY_DIR}" --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
                    ${mars_analysis_sources}
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                COMMENT "Analizar las rutas del núcleo de simulación"
                VERBATIM)
        else()
            add_custom_target(static-analysis DEPENDS mars_titan_simulation)
        endif()
    endif()
    mars_titan_coverage_report()
elseif(MARS_TITAN_BUILD_FUZZER)
    message(FATAL_ERROR "El fuzzing necesita la biblioteca de simulación")
endif()
