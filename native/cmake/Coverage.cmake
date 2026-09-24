if(MARS_TITAN_ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       NOT MARS_TITAN_SANITIZER STREQUAL "none" OR MARS_TITAN_ENABLE_IPO OR
       NOT MARS_TITAN_PGO STREQUAL "off" OR MARS_TITAN_PROFILE OR MARS_TITAN_ENABLE_CUDA)
        message(FATAL_ERROR "La cobertura requiere Clang y un perfil separado de sanitizadores, CUDA, LTO y PGO")
    endif()
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fprofile-instr-generate -fcoverage-mapping -fprofile-update=atomic")
    set(CMAKE_REQUIRED_LINK_OPTIONS -fprofile-instr-generate)
    check_cxx_source_compiles("int main() { return 0; }" MARS_TITAN_HAS_LLVM_COVERAGE)
    cmake_pop_check_state()
    if(NOT MARS_TITAN_HAS_LLVM_COVERAGE)
        message(FATAL_ERROR "El compilador no puede instrumentar y enlazar la cobertura LLVM")
    endif()
    string(REGEX MATCH "^[0-9]+" compiler_major "${CMAKE_CXX_COMPILER_VERSION}")
    find_program(MARS_TITAN_LLVM_PROFDATA NAMES "llvm-profdata-${compiler_major}" llvm-profdata REQUIRED)
    find_program(MARS_TITAN_LLVM_COV NAMES "llvm-cov-${compiler_major}" llvm-cov REQUIRED)
    foreach(tool MARS_TITAN_LLVM_PROFDATA MARS_TITAN_LLVM_COV)
        execute_process(COMMAND "${${tool}}" --version
            RESULT_VARIABLE tool_result OUTPUT_VARIABLE tool_version ERROR_VARIABLE tool_error)
        if(NOT tool_result EQUAL 0 OR NOT tool_version MATCHES "version ${compiler_major}\\.")
            message(FATAL_ERROR "${tool} debe compartir la versión principal de Clang: ${tool_error}")
        endif()
    endforeach()
    set(MARS_TITAN_COVERAGE_DIR "${CMAKE_BINARY_DIR}/coverage")
    file(MAKE_DIRECTORY "${MARS_TITAN_COVERAGE_DIR}")
endif()

function(mars_titan_cover_target target)
    if(MARS_TITAN_ENABLE_COVERAGE)
        get_filename_component(project_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../.." ABSOLUTE)
        target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping
            -fprofile-update=atomic "-fcoverage-prefix-map=${project_root}=.")
        target_link_options(${target} PUBLIC -fprofile-instr-generate)
    endif()
endfunction()

function(mars_titan_coverage_report)
    if(NOT MARS_TITAN_ENABLE_COVERAGE)
        return()
    endif()
    set(objects "")
    set(targets "")
    foreach(target mars_titan_simulation simulation_tests threaded_simulation c_abi_test
                   financial_session_tests mars-titan-sim)
        if(TARGET ${target})
            list(APPEND objects "$<TARGET_FILE:${target}>")
            list(APPEND targets ${target})
        endif()
    endforeach()
    add_custom_target(coverage-report
        COMMAND "${CMAKE_COMMAND}"
            "-DPROFILE_DIR=${MARS_TITAN_COVERAGE_DIR}"
            "-DPROFDATA=${MARS_TITAN_LLVM_PROFDATA}" "-DCOV=${MARS_TITAN_LLVM_COV}"
            "-DOBJECTS=${objects}" "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ReportCoverage.cmake"
        DEPENDS ${targets}
        COMMENT "Combinar cobertura LLVM de líneas y ramas"
        VERBATIM)
endfunction()
