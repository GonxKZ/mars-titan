option(MARS_TITAN_BUILD_LEARNING_CONTROLS "Compilar los controles técnicos de replay y adaptación" OFF)
if(NOT MARS_TITAN_BUILD_LEARNING_CONTROLS)
    return()
endif()

include(CTest)
add_library(mars_titan_replay_schedule STATIC src/replay_schedule.cpp)
target_include_directories(mars_titan_replay_schedule PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
mars_titan_configure_target(mars_titan_replay_schedule)
add_executable(mars-titan-replay-control src/replay_control_main.cpp)
target_link_libraries(mars-titan-replay-control PRIVATE mars_titan_replay_schedule)
mars_titan_configure_target(mars-titan-replay-control)
if(BUILD_TESTING)
    add_executable(replay_schedule_tests tests/replay_schedule_tests.cpp)
    target_link_libraries(replay_schedule_tests PRIVATE mars_titan_replay_schedule)
    mars_titan_configure_target(replay_schedule_tests)
    add_test(NAME replay_schedule COMMAND replay_schedule_tests)
    set_tests_properties(replay_schedule PROPERTIES TIMEOUT 60 LABELS "unit;integration;controls")
    mars_titan_sanitizer_test_environment(replay_schedule)
    add_test(NAME replay_control COMMAND mars-titan-replay-control 17 3 4 71 3)
    add_test(NAME replay_control_invalid COMMAND mars-titan-replay-control 0)
    set_tests_properties(replay_control_invalid PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(replay_control replay_control_invalid PROPERTIES TIMEOUT 60 LABELS "integration;controls")
    mars_titan_sanitizer_test_environment(replay_control replay_control_invalid)
endif()

if(MARS_TITAN_BUILD_FUZZER)
    add_executable(fuzz_replay_schedule tests/fuzz_replay_schedule.cpp)
    target_link_libraries(fuzz_replay_schedule PRIVATE mars_titan_replay_schedule)
    mars_titan_configure_target(fuzz_replay_schedule)
    target_compile_options(mars_titan_replay_schedule PRIVATE -fsanitize=fuzzer-no-link)
    target_compile_options(fuzz_replay_schedule PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(fuzz_replay_schedule PRIVATE -fsanitize=fuzzer)
    if(BUILD_TESTING)
        add_test(NAME fuzz_replay_schedule_smoke COMMAND fuzz_replay_schedule
            -seed=71 -runs=1000 -max_len=4096 -timeout=5 -rss_limit_mb=1024)
        set_tests_properties(fuzz_replay_schedule_smoke PROPERTIES TIMEOUT 60 LABELS "fuzz;controls")
        mars_titan_sanitizer_test_environment(fuzz_replay_schedule_smoke)
    endif()
endif()

if(MARS_TITAN_ENABLE_STATIC_ANALYZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_custom_target(learning-controls-analysis
        COMMAND "${MARS_TITAN_CLANG_CHECK}" --analyze "-p=${CMAKE_BINARY_DIR}"
            --extra-arg=-Xanalyzer --extra-arg=-analyzer-werror
            "${CMAKE_CURRENT_SOURCE_DIR}/src/replay_schedule.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/replay_control_main.cpp"
        VERBATIM)
endif()
