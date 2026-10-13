if(NOT MARS_TITAN_BUILD_EPISODIC_PYTHON)
    return()
endif()
if(MARS_TITAN_SANITIZER STREQUAL "thread" OR MARS_TITAN_SANITIZER STREQUAL "memory")
    message(FATAL_ERROR "El enlace episódico necesita Python y LibTorch instrumentados para TSan o MSan")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/TorchDependencies.cmake")
mars_titan_find_torch()
get_filename_component(episode_project "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_target_property(episode_environment mars_titan_torch MARS_TITAN_TORCH_ENVIRONMENT)
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "UV_PROJECT_ENVIRONMENT=${episode_environment}"
    "${MARS_TITAN_UV}" run --no-sync --offline --project "${episode_project}"
    python -c "import sys,sysconfig; print(sys.executable); print(sysconfig.get_paths()['include']); print(sysconfig.get_config_var('SOABI'))"
    WORKING_DIRECTORY "${episode_project}"
    RESULT_VARIABLE episode_probe OUTPUT_VARIABLE episode_paths ERROR_VARIABLE episode_error
    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT ${MARS_TITAN_PROBE_TIMEOUT})
if(NOT episode_probe EQUAL 0)
    message(FATAL_ERROR "No se pudo localizar Python del entorno uv: ${episode_error}")
endif()
string(REPLACE "\n" ";" episode_paths "${episode_paths}")
list(GET episode_paths 0 episode_python)
list(GET episode_paths 1 episode_include)
list(GET episode_paths 2 episode_soabi)
set(Python3_EXECUTABLE "${episode_python}")
if(CMAKE_CROSSCOMPILING)
    # La raíz de destino reubicaría las cabeceras del intérprete aarch64, que se dan explícitas.
    set(Python3_INCLUDE_DIR "${episode_include}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT EXISTS "${MARS_TITAN_TORCH_ROOT}/include/pybind11/pybind11.h")
    message(FATAL_ERROR "El SDK LibTorch no contiene las cabeceras pybind11 del enlace")
endif()

add_library(mars_titan_episode_storage STATIC src/episodic_memory.cpp)
target_include_directories(mars_titan_episode_storage PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mars_titan_episode_storage PUBLIC mars_titan::torch)
mars_titan_configure_target(mars_titan_episode_storage)
Python3_add_library(_episodic_native MODULE WITH_SOABI src/episodic_python.cpp)
target_link_libraries(_episodic_native PRIVATE mars_titan_episode_storage mars_titan_cohort_execution)
target_compile_definitions(_episodic_native PRIVATE
    "MARS_TITAN_EPISODIC_TORCH_VERSION=\"${MARS_TITAN_TORCH_VERSION}\"")
set_target_properties(_episodic_native PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(CMAKE_CROSSCOMPILING)
    # Al compilar en cruzado FindPython deja vacío el sufijo de extensión, que da el intérprete.
    set_target_properties(_episodic_native PROPERTIES
        SUFFIX ".${episode_soabi}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()
mars_titan_configure_target(_episodic_native)
list(APPEND mars_analysis_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/episodic_memory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/episodic_python.cpp")
if(BUILD_TESTING)
    add_test(NAME episodic_python_environment COMMAND "${CMAKE_COMMAND}"
        "-DSDK_ENVIRONMENT=${MARS_TITAN_TORCH_ENVIRONMENT}"
        "-DPROJECT_ROOT=${episode_project}"
        "-DUV_EXECUTABLE=${MARS_TITAN_UV}"
        "-DSELECTED_PYTHON=${Python3_EXECUTABLE}"
        "-DPROBE_TIMEOUT=${MARS_TITAN_PROBE_TIMEOUT}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/episodic_python_environment.cmake")
    set_tests_properties(episodic_python_environment PROPERTIES TIMEOUT 60 LABELS "configuration;memory")
endif()
if(BUILD_TESTING AND NOT TARGET episodic_memory_tests)
    add_executable(episodic_memory_tests tests/episodic_memory_tests.cpp)
    target_link_libraries(episodic_memory_tests PRIVATE mars_titan_episode_storage)
    mars_titan_configure_target(episodic_memory_tests)
    add_test(NAME episodic_memory COMMAND episodic_memory_tests)
    set_tests_properties(episodic_memory PROPERTIES TIMEOUT 120 LABELS "unit;integration;memory")
    mars_titan_sanitizer_test_environment(episodic_memory)
    set_property(TEST episodic_memory APPEND PROPERTY ENVIRONMENT
        "CUDA_VISIBLE_DEVICES=-1" "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1")
endif()
