option(MARS_TITAN_BUILD_RL_OBJECTIVES "Compilar controles puros de objetivos RL, sin entrenador" OFF)
if(NOT MARS_TITAN_BUILD_RL_OBJECTIVES AND NOT MARS_TITAN_BUILD_PPO)
    return()
endif()

if(MARS_TITAN_SANITIZER STREQUAL "thread" OR MARS_TITAN_SANITIZER STREQUAL "memory")
    message(FATAL_ERROR "Los objetivos RL necesitan LibTorch instrumentado para TSan o MSan")
endif()
include(CTest)
include(cmake/TorchDependencies.cmake)
mars_titan_find_torch()
add_library(mars_titan_rl_objectives STATIC
    src/ppo_objectives.cpp src/ppo_controller.cpp src/klpo_terminal.cpp src/klpo_episodes.cpp
    src/group_relative.cpp src/quantile_dqn.cpp)
target_include_directories(mars_titan_rl_objectives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_rl_objectives PUBLIC mars_titan::torch)
mars_titan_configure_target(mars_titan_rl_objectives)
if(BUILD_TESTING)
    add_executable(klpo_episodes_tests tests/klpo_episodes_tests.cpp)
    target_link_libraries(klpo_episodes_tests PRIVATE mars_titan_rl_objectives)
    mars_titan_configure_target(klpo_episodes_tests)
    add_test(NAME klpo_episodes COMMAND klpo_episodes_tests)
    set_tests_properties(klpo_episodes PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
    mars_titan_sanitizer_test_environment(klpo_episodes)
    set_property(TEST klpo_episodes APPEND PROPERTY ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1")
    add_executable(ppo_objective_tests tests/ppo_objective_tests.cpp)
    target_link_libraries(ppo_objective_tests PRIVATE mars_titan_rl_objectives)
    mars_titan_configure_target(ppo_objective_tests)
    add_test(NAME ppo_objectives COMMAND ppo_objective_tests)
    set_tests_properties(ppo_objectives PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
    mars_titan_sanitizer_test_environment(ppo_objectives)
    set_property(TEST ppo_objectives APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
    add_executable(ppo_controller_tests tests/ppo_controller_tests.cpp)
    target_link_libraries(ppo_controller_tests PRIVATE mars_titan_rl_objectives)
    mars_titan_configure_target(ppo_controller_tests)
    add_test(NAME ppo_controller COMMAND ppo_controller_tests)
    set_tests_properties(ppo_controller PROPERTIES TIMEOUT 120 LABELS "unit;rl-objectives")
    mars_titan_sanitizer_test_environment(ppo_controller)
    set_property(TEST ppo_controller APPEND PROPERTY ENVIRONMENT
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
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_controller.cpp"
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_terminal.cpp"
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/klpo_episodes.cpp"
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/group_relative.cpp"
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/quantile_dqn.cpp"
        COMMENT "Analizar los objetivos y el controlador de PPO y KLPO"
        VERBATIM)
endif()
