cmake_minimum_required(VERSION 3.24)

project(mars_titan_native VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MARS_TITAN_ENABLE_CUDA "Configurar las herramientas de CUDA" OFF)

add_library(mars_titan_native_options INTERFACE)
add_library(mars_titan::native_options ALIAS mars_titan_native_options)

target_compile_features(mars_titan_native_options INTERFACE c_std_17 cxx_std_20)
target_compile_options(
    mars_titan_native_options
    INTERFACE
        "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->"
)

if(MARS_TITAN_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_features(mars_titan_native_options INTERFACE cuda_std_20)

    message(STATUS "CUDA configurada con toolkit ${CUDAToolkit_VERSION}")
endif()

message(STATUS "Configuracion sin fuentes cientificas ni objetivos binarios")
