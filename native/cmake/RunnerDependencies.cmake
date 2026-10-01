include(FetchContent)

function(mars_titan_find_json)
    # La huella corresponde al archivo publicado por el proyecto nlohmann/json.
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
        TLS_VERIFY TRUE
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    FetchContent_MakeAvailable(nlohmann_json)
    get_target_property(json_includes nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET nlohmann_json PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${json_includes}")
endfunction()

function(mars_titan_find_arrow)
    find_package(Arrow CONFIG QUIET)
    find_package(Parquet CONFIG QUIET)
    foreach(candidate Arrow::arrow_shared Arrow::arrow_static)
        if(TARGET ${candidate})
            set(arrow_target ${candidate})
            break()
        endif()
    endforeach()
    foreach(candidate Parquet::parquet_shared Parquet::parquet_static)
        if(TARGET ${candidate})
            set(parquet_target ${candidate})
            break()
        endif()
    endforeach()
    if(arrow_target AND parquet_target)
        add_library(mars_titan::arrow ALIAS ${arrow_target})
        add_library(mars_titan::parquet ALIAS ${parquet_target})
        message(STATUS "Arrow y Parquet encontrados mediante sus paquetes CMake")
        return()
    endif()

    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "Esta plataforma necesita un SDK Arrow/Parquet con paquetes CMake")
    endif()
    find_program(MARS_TITAN_UV NAMES uv REQUIRED)
    get_filename_component(project_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../.." ABSOLUTE)
    execute_process(COMMAND "${MARS_TITAN_UV}" run --no-sync --project "${project_root}" python -c
        "import json,pathlib,pyarrow; print(json.dumps(dict(include=pyarrow.get_include(),root=str(pathlib.Path(pyarrow.__file__).parent),so=pyarrow.cpp_build_info.so_version,version=pyarrow.__version__)))"
        RESULT_VARIABLE probe_result OUTPUT_VARIABLE sdk ERROR_VARIABLE probe_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT probe_result EQUAL 0)
        message(FATAL_ERROR "No hay SDK Arrow/Parquet ni bibliotecas C++ de PyArrow disponibles: ${probe_error}")
    endif()
    foreach(key include root so version)
        string(JSON sdk_${key} ERROR_VARIABLE json_error GET "${sdk}" ${key})
        if(json_error)
            message(FATAL_ERROR "No se pudo interpretar el SDK C++ de PyArrow: ${json_error}")
        endif()
    endforeach()
    if(NOT EXISTS "${sdk_include}/arrow/api.h" OR NOT EXISTS "${sdk_include}/parquet/arrow/reader.h")
        message(FATAL_ERROR "PyArrow no incluye las cabeceras C++ requeridas")
    endif()
    find_file(arrow_library NAMES "libarrow.so.${sdk_so}" libarrow.so
        PATHS "${sdk_root}" NO_DEFAULT_PATH REQUIRED NO_CACHE)
    find_file(parquet_library NAMES "libparquet.so.${sdk_so}" libparquet.so
        PATHS "${sdk_root}" NO_DEFAULT_PATH REQUIRED NO_CACHE)
    add_library(mars_titan_arrow SHARED IMPORTED)
    set_target_properties(mars_titan_arrow PROPERTIES
        IMPORTED_LOCATION "${arrow_library}"
        MARS_TITAN_DEPENDENCY_VERSION "${sdk_version}"
        INTERFACE_INCLUDE_DIRECTORIES "${sdk_include}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${sdk_include}")
    add_library(mars_titan_parquet SHARED IMPORTED)
    set_target_properties(mars_titan_parquet PROPERTIES
        IMPORTED_LOCATION "${parquet_library}"
        MARS_TITAN_DEPENDENCY_VERSION "${sdk_version}"
        INTERFACE_INCLUDE_DIRECTORIES "${sdk_include}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${sdk_include}"
        INTERFACE_LINK_LIBRARIES mars_titan_arrow)
    add_library(mars_titan::arrow ALIAS mars_titan_arrow)
    add_library(mars_titan::parquet ALIAS mars_titan_parquet)
    message(STATUS "SDK C++ de PyArrow ${sdk_version}: libarrow y libparquet, sin enlaces a Python")
endfunction()
