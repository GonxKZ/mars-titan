option(MARS_TITAN_BUILD_RL_OBJECTIVES "Compilar controles puros de objetivos RL, sin entrenador" OFF)
if(NOT MARS_TITAN_BUILD_RL_OBJECTIVES)
    return()
endif()

if(MARS_TITAN_SANITIZER STREQUAL "thread" OR MARS_TITAN_SANITIZER STREQUAL "memory")
    message(FATAL_ERROR "Los objetivos RL necesitan LibTorch instrumentado para TSan o MSan")
endif()
include(CTest)
include(cmake/TorchDependencies.cmake)
mars_titan_find_torch()
add_library(mars_titan_rl_objectives STATIC src/ppo_objectives.cpp src/klpo_terminal.cpp)
target_include_directories(mars_titan_rl_objectives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_rl_objectives PUBLIC mars_titan::torch)
mars_titan_configure_target(mars_titan_rl_objectives)
if(BUILD_TESTING)
    add_executable(ppo_objective_tests tests/ppo_objective_tests.cpp)
    target_link_libraries(ppo_objective_tests PRIVATE mars_titan_rl_objectives)
    mars_titan_configure_target(ppo_objective_tests)
    add_test(NAME ppo_objectives COMMAND ppo_objective_tests)
    set_tests_properties(ppo_objectives PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
    mars_titan_sanitizer_test_environment(ppo_objectives)
    set_property(TEST ppo_objectives APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_executable(klpo_terminal_tests tests/klpo_terminal_tests.cpp)
    target_link_libraries(klpo_terminal_tests PRIVATE mars_titan_rl_objectives)
    mars_titan_configure_target(klpo_terminal_tests)
    add_test(NAME klpo_terminal COMMAND klpo_terminal_tests)
    set_tests_properties(klpo_terminal PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
    mars_titan_sanitizer_test_environment(klpo_terminal)
    set_property(TEST klpo_terminal APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=2" "MKL_NUM_THREADS=2")
endif()

if(MARS_TITAN_ENABLE_STATIC_ANALYZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_custom_target(rl-objectives-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_objectives.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_terminal.cpp"
        COMMENT "Analizar los objetivos puros de PPO y KLPO"
        VERBATIM)
endif()
