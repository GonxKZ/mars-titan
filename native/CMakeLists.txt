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
option(MARS_TITAN_BUILD_PPO "Compilar PPO nativo con LibTorch" OFF)
option(MARS_TITAN_BUILD_COHORT_EXECUTION "Compilar el ejecutor de cohortes recuperable" OFF)
option(MARS_TITAN_BUILD_EPISODIC_PYTHON "Enlazar el banco y el ejecutor con Python" OFF)
option(MARS_TITAN_BUILD_FINANCIAL_CONTROLS "Compilar referencias para auditorías financieras congeladas" OFF)
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

if(MARS_TITAN_BUILD_EPISODIC_PYTHON)
    set(MARS_TITAN_BUILD_COHORT_EXECUTION ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(MARS_TITAN_BUILD_RUNNER OR MARS_TITAN_BUILD_COHORT_EXECUTION OR MARS_TITAN_BUILD_FINANCIAL_CONTROLS)
    set(MARS_TITAN_BUILD_FINANCIAL ON)
endif()
if(MARS_TITAN_BUILD_PPO)
    set(MARS_TITAN_BUILD_FINANCIAL ON)
    if(MARS_TITAN_SANITIZER STREQUAL "memory" OR MARS_TITAN_SANITIZER STREQUAL "thread")
        message(FATAL_ERROR "PPO necesita un runtime LibTorch instrumentado para MSan o TSan. El núcleo conserva sus perfiles independientes")
    endif()
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

include(cmake/RLObjectives.cmake)

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
        find_package(Threads REQUIRED)
        add_library(mars_titan_financial STATIC src/financial_session.cpp src/financial_batch.cpp
            src/markov_filter.cpp)
        target_include_directories(mars_titan_financial PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_financial PUBLIC mars_titan_simulation Threads::Threads)
        mars_titan_configure_target(mars_titan_financial)
        add_executable(mars-titan-batch-benchmark benchmarks/batch_rollout.cpp)
        target_link_libraries(mars-titan-batch-benchmark PRIVATE mars_titan_financial)
        mars_titan_configure_target(mars-titan-batch-benchmark)
        list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_session.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_batch.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/markov_filter.cpp")
        if(BUILD_TESTING)
            add_executable(financial_session_tests tests/financial_session_tests.cpp)
            target_link_libraries(financial_session_tests PRIVATE mars_titan_financial)
            mars_titan_configure_target(financial_session_tests)
            add_test(NAME financial_session COMMAND financial_session_tests)
            set_tests_properties(financial_session PROPERTIES TIMEOUT 60 LABELS "unit;integration;financial")
            mars_titan_sanitizer_test_environment(financial_session)
            add_executable(financial_batch_tests tests/financial_batch_tests.cpp)
            target_link_libraries(financial_batch_tests PRIVATE mars_titan_financial)
            mars_titan_configure_target(financial_batch_tests)
            add_test(NAME financial_batch COMMAND financial_batch_tests)
            set_tests_properties(financial_batch PROPERTIES TIMEOUT 60 LABELS "unit;integration;financial;concurrency")
            mars_titan_sanitizer_test_environment(financial_batch)
            add_executable(markov_filter_tests tests/markov_filter_tests.cpp)
            target_link_libraries(markov_filter_tests PRIVATE mars_titan_financial)
            mars_titan_configure_target(markov_filter_tests)
            add_test(NAME markov_filter COMMAND markov_filter_tests)
            set_tests_properties(markov_filter PROPERTIES TIMEOUT 60 LABELS "unit;integration;markov")
            mars_titan_sanitizer_test_environment(markov_filter)
        endif()
    endif()

    if(MARS_TITAN_BUILD_RUNNER OR MARS_TITAN_BUILD_PPO OR MARS_TITAN_BUILD_COHORT_EXECUTION OR MARS_TITAN_BUILD_FINANCIAL_CONTROLS)
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
            add_library(mars_titan_simulation_files STATIC src/financial_parquet.cpp src/simulation_files.cpp)
            target_include_directories(mars_titan_simulation_files PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
            target_link_libraries(mars_titan_simulation_files PUBLIC mars_titan_financial mars_titan::parquet
                OpenSSL::Crypto Threads::Threads nlohmann_json::nlohmann_json)
            mars_titan_configure_target(mars_titan_simulation_files)
            if(MARS_TITAN_BUILD_RUNNER)
                add_executable(mars-titan-sim src/simulation_main.cpp)
                target_link_libraries(mars-titan-sim PRIVATE mars_titan_simulation_files)
                mars_titan_configure_target(mars-titan-sim)
                list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation_main.cpp")
            endif()
            list(APPEND mars_analysis_sources
                "${CMAKE_CURRENT_SOURCE_DIR}/src/financial_parquet.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation_files.cpp")
        endif()
    endif()

    include(cmake/CohortExecution.cmake)
    include(cmake/FinancialControls.cmake)

    if(MARS_TITAN_BUILD_PPO)
        include(cmake/TorchDependencies.cmake)
        mars_titan_find_torch()
        add_library(mars_titan_ppo STATIC src/ppo_policy.cpp src/ppo_training.cpp
            src/episodic_memory.cpp src/policy_context.cpp src/learning_replay.cpp)
        target_include_directories(mars_titan_ppo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_ppo PUBLIC mars_titan_financial mars_titan::torch
            mars_titan_rl_objectives mars_titan_simulation_files)
        mars_titan_configure_target(mars_titan_ppo)
        add_executable(mars-titan-memory-benchmark benchmarks/episodic_query.cpp)
        target_link_libraries(mars-titan-memory-benchmark PRIVATE mars_titan_ppo)
        mars_titan_configure_target(mars-titan-memory-benchmark)
        add_library(mars_titan_ppo_files STATIC src/ppo_checkpoints.cpp src/ppo_inputs.cpp src/decision_trace.cpp)
        target_include_directories(mars_titan_ppo_files PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_ppo_files PUBLIC mars_titan_simulation_files)
        mars_titan_configure_target(mars_titan_ppo_files)
        add_library(mars_titan_klpo_collection STATIC src/klpo_collection.cpp)
        target_include_directories(mars_titan_klpo_collection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_klpo_collection PUBLIC mars_titan_ppo mars_titan_ppo_files)
        mars_titan_configure_target(mars_titan_klpo_collection)
        add_library(mars_titan_klpo_learning STATIC src/klpo_learning.cpp)
        target_include_directories(mars_titan_klpo_learning PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(mars_titan_klpo_learning PUBLIC mars_titan_klpo_collection)
        mars_titan_configure_target(mars_titan_klpo_learning)
        add_executable(mars-titan-ppo src/ppo_main.cpp src/ppo_experiment.cpp)
        target_link_libraries(mars-titan-ppo PRIVATE mars_titan_ppo mars_titan_ppo_files)
        mars_titan_configure_target(mars-titan-ppo)
        list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_policy.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_collection.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_learning.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_training.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/episodic_memory.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/policy_context.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/learning_replay.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_checkpoints.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_inputs.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/decision_trace.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_main.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_experiment.cpp")
        if(BUILD_TESTING)
            add_executable(klpo_reference_tests tests/klpo_reference_tests.cpp)
            target_link_libraries(klpo_reference_tests PRIVATE mars_titan_klpo_collection)
            mars_titan_configure_target(klpo_reference_tests)
            add_test(NAME klpo_reference COMMAND klpo_reference_tests)
            set_tests_properties(klpo_reference PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(klpo_reference)
            set_property(TEST klpo_reference APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(klpo_learning_tests tests/klpo_learning_tests.cpp)
            target_link_libraries(klpo_learning_tests PRIVATE mars_titan_klpo_learning)
            mars_titan_configure_target(klpo_learning_tests)
            add_test(NAME klpo_learning COMMAND klpo_learning_tests)
            set_tests_properties(klpo_learning PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(klpo_learning)
            set_property(TEST klpo_learning APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(klpo_actor_tests tests/klpo_actor_tests.cpp)
            target_link_libraries(klpo_actor_tests PRIVATE mars_titan_klpo_collection)
            mars_titan_configure_target(klpo_actor_tests)
            add_test(NAME klpo_actor COMMAND klpo_actor_tests)
            set_tests_properties(klpo_actor PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(klpo_actor)
            set_property(TEST klpo_actor APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(klpo_collection_tests tests/klpo_collection_tests.cpp)
            target_link_libraries(klpo_collection_tests PRIVATE mars_titan_klpo_collection)
            mars_titan_configure_target(klpo_collection_tests)
            add_test(NAME klpo_collection COMMAND klpo_collection_tests)
            set_tests_properties(klpo_collection PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(klpo_collection)
            set_property(TEST klpo_collection APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(klpo_policy_tests tests/klpo_policy_tests.cpp)
            target_link_libraries(klpo_policy_tests PRIVATE mars_titan_ppo)
            mars_titan_configure_target(klpo_policy_tests)
            add_test(NAME klpo_policy COMMAND klpo_policy_tests)
            set_tests_properties(klpo_policy PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(klpo_policy)
            set_property(TEST klpo_policy APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(ppo_policy_tests tests/ppo_policy_tests.cpp)
            target_link_libraries(ppo_policy_tests PRIVATE mars_titan_ppo)
            mars_titan_configure_target(ppo_policy_tests)
            add_executable(ppo_variant_state_tests tests/ppo_variant_state_tests.cpp)
            target_link_libraries(ppo_variant_state_tests PRIVATE mars_titan_ppo)
            mars_titan_configure_target(ppo_variant_state_tests)
            add_test(NAME ppo_variant_state COMMAND ppo_variant_state_tests)
            set_tests_properties(ppo_variant_state PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(ppo_variant_state)
            set_property(TEST ppo_variant_state APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_executable(ppo_variant_objective_tests tests/ppo_variant_objective_tests.cpp)
            target_link_libraries(ppo_variant_objective_tests PRIVATE mars_titan_ppo)
            mars_titan_configure_target(ppo_variant_objective_tests)
            add_test(NAME ppo_variant_objective COMMAND ppo_variant_objective_tests)
            set_tests_properties(ppo_variant_objective PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
            mars_titan_sanitizer_test_environment(ppo_variant_objective)
            set_property(TEST ppo_variant_objective APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            add_test(NAME ppo_policy COMMAND ppo_policy_tests)
            set_tests_properties(ppo_policy PROPERTIES TIMEOUT 120 LABELS "unit;integration;ppo")
            mars_titan_sanitizer_test_environment(ppo_policy)
            set_property(TEST ppo_policy APPEND PROPERTY ENVIRONMENT
                "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            foreach(component training checkpoint inputs gru_packing)
                add_executable(ppo_${component}_tests tests/ppo_${component}_tests.cpp)
                if(component STREQUAL "training" OR component STREQUAL "gru_packing")
                    target_link_libraries(ppo_${component}_tests PRIVATE mars_titan_ppo)
                else()
                    target_link_libraries(ppo_${component}_tests PRIVATE mars_titan_ppo_files)
                endif()
                mars_titan_configure_target(ppo_${component}_tests)
                add_test(NAME ppo_${component} COMMAND ppo_${component}_tests)
                set_tests_properties(ppo_${component} PROPERTIES TIMEOUT 120 LABELS "unit;integration;ppo")
                mars_titan_sanitizer_test_environment(ppo_${component})
                set_property(TEST ppo_${component} APPEND PROPERTY ENVIRONMENT
                    "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            endforeach()
            foreach(component episodic_memory policy_context decision_trace learning_replay)
                add_executable(${component}_tests tests/${component}_tests.cpp)
                if(component STREQUAL "decision_trace")
                    target_link_libraries(${component}_tests PRIVATE mars_titan_ppo_files)
                else()
                    target_link_libraries(${component}_tests PRIVATE mars_titan_ppo)
                endif()
                mars_titan_configure_target(${component}_tests)
                add_test(NAME ${component} COMMAND ${component}_tests)
                set_tests_properties(${component} PROPERTIES TIMEOUT 120 LABELS "unit;integration;ppo")
                mars_titan_sanitizer_test_environment(${component})
                set_property(TEST ${component} APPEND PROPERTY ENVIRONMENT
                    "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
            endforeach()
        endif()
    endif()

    include(cmake/EpisodicPython.cmake)

    if(MARS_TITAN_BUILD_FINANCIAL)
        set(identity_files ${mars_analysis_sources}
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/simulation.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/financial_session.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/financial_batch.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/markov_filter.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/accurate_sum.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Diagnostics.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Instrumentation.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Coverage.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildIdentity.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunnerDependencies.cmake")
        if(TARGET mars_titan_simulation_files)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/simulation_files.hpp")
        endif()
        if(MARS_TITAN_BUILD_COHORT_EXECUTION)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/cohort_execution.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CohortExecution.cmake")
        endif()
        if(MARS_TITAN_BUILD_EPISODIC_PYTHON)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/episodic_memory.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EpisodicPython.cmake"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TorchDependencies.cmake")
        endif()
        if(MARS_TITAN_BUILD_FINANCIAL_CONTROLS)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/financial_controls.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FinancialControls.cmake")
        endif()
        if(MARS_TITAN_BUILD_PPO)
            list(APPEND identity_files "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_policy.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/klpo_collection.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/klpo_learning.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/klpo_episodes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_episodes.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/klpo_terminal.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_terminal.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_controller.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_controller.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_objectives.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_objectives.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RLObjectives.cmake"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_training.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_checkpoints.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_inputs.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/ppo_experiment.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/episodic_memory.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/policy_context.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/learning_replay.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/mars_titan/decision_trace.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TorchDependencies.cmake")
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
        if(TARGET mars-titan-sim)
            add_executable(fuzz_serialization tests/fuzz_serialization.cpp src/simulation_files.cpp)
            target_link_libraries(fuzz_serialization PRIVATE mars_titan_financial OpenSSL::Crypto
                Threads::Threads nlohmann_json::nlohmann_json)
            mars_titan_configure_target(fuzz_serialization)
            target_compile_options(fuzz_serialization PRIVATE -fsanitize=fuzzer-no-link)
            target_link_options(fuzz_serialization PRIVATE -fsanitize=fuzzer)
            set(fuzz_json_corpus "${CMAKE_CURRENT_BINARY_DIR}/fuzz-json-corpus")
            configure_file(tests/fixtures/snapshot.json "${fuzz_json_corpus}/snapshot.json" COPYONLY)
            if(BUILD_TESTING)
                add_test(NAME fuzz_serialization_smoke COMMAND fuzz_serialization
                    "${fuzz_json_corpus}" -seed=42 -runs=1000 -max_len=4096 -timeout=5 -rss_limit_mb=2048)
                set_tests_properties(fuzz_serialization_smoke PROPERTIES
                    TIMEOUT 60 LABELS "fuzz;serialization")
                mars_titan_sanitizer_test_environment(fuzz_serialization_smoke)
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
elseif(MARS_TITAN_BUILD_FUZZER AND NOT MARS_TITAN_BUILD_CANDIDATE)
    message(FATAL_ERROR "El fuzzing necesita la biblioteca de simulación")
endif()

include(cmake/MemoryStress.cmake)
include(cmake/LearningControls.cmake)
include(cmake/Candidate.cmake)

if(MARS_TITAN_ENABLE_STATIC_ANALYZER)
    if(NOT TARGET static-analysis)
        add_custom_target(static-analysis)
    endif()
    foreach(target memory-stress-static-analysis learning-controls-analysis adapter-controls-analysis candidate-analysis rl-objectives-analysis)
        if(TARGET ${target})
            add_dependencies(static-analysis ${target})
        endif()
    endforeach()
endif()
mars_titan_coverage_report()
