cmake_minimum_required(VERSION 3.24)

project(mars_titan_native VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MARS_TITAN_ENABLE_CUDA "Configurar las herramientas de CUDA" OFF)
option(MARS_TITAN_WARNINGS_AS_ERRORS "Tratar avisos propios como errores" OFF)
option(MARS_TITAN_ENABLE_CLANG_TIDY "Analizar los objetivos propios con clang-tidy" OFF)
option(MARS_TITAN_PROFILE "Conservar información para perfiles de rendimiento" OFF)
set(MARS_TITAN_SANITIZER "none" CACHE STRING "Instrumentación: none, address-undefined o thread")
set_property(CACHE MARS_TITAN_SANITIZER PROPERTY STRINGS none address-undefined thread)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT MARS_TITAN_SANITIZER MATCHES "^(none|address-undefined|thread)$")
    message(FATAL_ERROR "La instrumentación solicitada no está soportada")
endif()
if(NOT MARS_TITAN_SANITIZER STREQUAL "none" AND (MARS_TITAN_PROFILE OR MARS_TITAN_ENABLE_CUDA))
    message(FATAL_ERROR "Separar sanitizadores CPU del perfil de rendimiento y de CUDA")
endif()
if(MARS_TITAN_ENABLE_CLANG_TIDY)
    find_program(MARS_TITAN_CLANG_TIDY NAMES clang-tidy clang-tidy-21 clang-tidy-20 REQUIRED)
    if(NOT EXISTS "${MARS_TITAN_CLANG_TIDY}")
        message(FATAL_ERROR "No existe el ejecutable clang-tidy solicitado")
    endif()
endif()

add_library(mars_titan_native_options INTERFACE)
add_library(mars_titan::native_options ALIAS mars_titan_native_options)

target_compile_features(mars_titan_native_options INTERFACE cxx_std_20)
target_compile_options(
    mars_titan_native_options
    INTERFACE
        "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->"
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wconversion;-Wshadow>"
)

if(MARS_TITAN_WARNINGS_AS_ERRORS)
    target_compile_options(mars_titan_native_options INTERFACE
        "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Werror>"
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>"
    )
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
    if(NOT MARS_TITAN_SANITIZER STREQUAL "none")
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            message(FATAL_ERROR "Estos sanitizadores requieren GCC o Clang")
        endif()
        if(MARS_TITAN_SANITIZER STREQUAL "address-undefined")
            set(sanitizers "address,undefined")
        else()
            set(sanitizers "thread")
        endif()
        target_compile_options(${target} PRIVATE -fsanitize=${sanitizers} -fno-omit-frame-pointer -fno-sanitize-recover=all)
        target_link_options(${target} PUBLIC -fsanitize=${sanitizers})
    endif()
    if(MARS_TITAN_PROFILE)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fno-omit-frame-pointer>")
    endif()
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

message(STATUS "Configuración sin fuentes científicas ni objetivos binarios")
