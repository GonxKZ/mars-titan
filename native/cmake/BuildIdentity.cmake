function(mars_titan_identity_value key value)
    string(LENGTH "${value}" value_length)
    string(APPEND build_identity "${key}[${value_length}]=${value}\n")
    set(build_identity "${build_identity}" PARENT_SCOPE)
endfunction()

function(mars_titan_build_identity source_hash)
    set(build_definitions "")
    if(CMAKE_CONFIGURATION_TYPES)
        set(configurations ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(configurations "${CMAKE_BUILD_TYPE}")
    else()
        set(configurations unspecified)
    endif()

    foreach(configuration IN LISTS configurations)
        set(build_identity "")
        set(active_configuration "${configuration}")
        if(configuration STREQUAL "unspecified" AND NOT CMAKE_BUILD_TYPE)
            set(active_configuration "")
        endif()
        string(TOUPPER "${active_configuration}" configuration_upper)
        mars_titan_identity_value(source_sha256 "${source_hash}")
        mars_titan_identity_value(build_type "${active_configuration}")
        foreach(variable CMAKE_VERSION CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM
                CMAKE_GENERATOR_TOOLSET CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION
                CMAKE_SYSTEM_PROCESSOR CMAKE_SIZEOF_VOID_P CMAKE_SYSROOT
                CMAKE_SYSROOT_COMPILE CMAKE_SYSROOT_LINK CMAKE_TOOLCHAIN_FILE
                CMAKE_MSVC_RUNTIME_LIBRARY CMAKE_POSITION_INDEPENDENT_CODE
                CMAKE_LINKER CMAKE_LINKER_TYPE CMAKE_AR CMAKE_RANLIB
                CMAKE_INTERPROCEDURAL_OPTIMIZATION
                MARS_TITAN_SANITIZER MARS_TITAN_ENABLE_IPO MARS_TITAN_PGO
                MARS_TITAN_ENABLE_COVERAGE MARS_TITAN_PROFILE MARS_TITAN_BUILD_FUZZER
                MARS_TITAN_ENABLE_CUDA MARS_TITAN_ENABLE_LIFETIME
                MARS_TITAN_ENABLE_CLANG_TIDY MARS_TITAN_ENABLE_STATIC_ANALYZER
                MARS_TITAN_WARNINGS_AS_ERRORS MARS_TITAN_MSAN_STDLIB_ROOT
                MARS_TITAN_USES_LIBSTDCXX MARS_TITAN_HAS_LIBCPP_HARDENING
                OPENSSL_VERSION Arrow_VERSION Parquet_VERSION)
            mars_titan_identity_value("${variable}" "${${variable}}")
        endforeach()
        foreach(language C CXX)
            foreach(suffix COMPILER COMPILER_ID COMPILER_VERSION COMPILER_FRONTEND_VARIANT
                    COMPILER_TARGET COMPILER_EXTERNAL_TOOLCHAIN COMPILER_ARG1 COMPILER_LAUNCHER
                    COMPILER_LINKER_ID COMPILER_LINKER_VERSION
                    SIMULATE_ID SIMULATE_VERSION STANDARD STANDARD_REQUIRED EXTENSIONS
                    FLAGS "FLAGS_${configuration_upper}" COMPILER_AR COMPILER_RANLIB
                    IMPLICIT_INCLUDE_DIRECTORIES IMPLICIT_LINK_DIRECTORIES IMPLICIT_LINK_LIBRARIES)
                set(variable "CMAKE_${language}_${suffix}")
                mars_titan_identity_value("${variable}" "${${variable}}")
            endforeach()
        endforeach()
        foreach(kind EXE SHARED MODULE STATIC)
            foreach(suffix "" "_${configuration_upper}")
                set(variable "CMAKE_${kind}_LINKER_FLAGS${suffix}")
                mars_titan_identity_value("${variable}" "${${variable}}")
            endforeach()
        endforeach()

        # Las expresiones de CMake se conservan junto a la configuración que las selecciona.
        set(targets mars_titan_native_options mars_titan_simulation mars_titan_financial)
        if(TARGET mars-titan-sim)
            list(APPEND targets mars-titan-sim mars_titan::arrow mars_titan::parquet
                OpenSSL::Crypto Threads::Threads nlohmann_json::nlohmann_json)
        endif()
        foreach(target IN LISTS targets)
            foreach(property TYPE COMPILE_FEATURES COMPILE_FLAGS COMPILE_OPTIONS COMPILE_DEFINITIONS
                    "COMPILE_DEFINITIONS_${configuration_upper}"
                    INCLUDE_DIRECTORIES LINK_OPTIONS LINK_LIBRARIES LINK_DIRECTORIES
                    LINK_FLAGS "LINK_FLAGS_${configuration_upper}"
                    INTERFACE_COMPILE_FEATURES INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS
                    INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                    INTERFACE_LINK_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_LINK_DIRECTORIES
                    C_STANDARD CXX_STANDARD C_EXTENSIONS CXX_EXTENSIONS
                    POSITION_INDEPENDENT_CODE INTERPROCEDURAL_OPTIMIZATION
                    "INTERPROCEDURAL_OPTIMIZATION_${configuration_upper}"
                    MSVC_RUNTIME_LIBRARY CXX_VISIBILITY_PRESET VISIBILITY_INLINES_HIDDEN
                    IMPORTED_LOCATION "IMPORTED_LOCATION_${configuration_upper}"
                    IMPORTED_IMPLIB "IMPORTED_IMPLIB_${configuration_upper}"
                    MAP_IMPORTED_CONFIG_${configuration_upper} VERSION
                    MARS_TITAN_DEPENDENCY_VERSION)
                get_target_property(value "${target}" "${property}")
                if(value STREQUAL "value-NOTFOUND")
                    set(value "")
                endif()
                mars_titan_identity_value("${target}.${property}" "${value}")
            endforeach()
            if(target MATCHES "^(mars_titan_simulation|mars_titan_financial|mars-titan-sim)$")
                get_target_property(sources "${target}" SOURCES)
                foreach(source IN LISTS sources)
                    foreach(property COMPILE_FLAGS COMPILE_OPTIONS COMPILE_DEFINITIONS
                            "COMPILE_DEFINITIONS_${configuration_upper}" INCLUDE_DIRECTORIES)
                        get_source_file_property(value "${source}" "${property}")
                        if(value STREQUAL "NOTFOUND")
                            set(value "")
                        endif()
                        mars_titan_identity_value("${source}.${property}" "${value}")
                    endforeach()
                endforeach()
            endif()
        endforeach()

        if(NOT MARS_TITAN_PGO STREQUAL "off")
            mars_titan_identity_value(pgo_path "${MARS_TITAN_PGO_DATA}")
        endif()
        if(MARS_TITAN_PGO STREQUAL "use")
            if(IS_DIRECTORY "${MARS_TITAN_PGO_DATA}")
                file(GLOB_RECURSE profile_files CONFIGURE_DEPENDS "${MARS_TITAN_PGO_DATA}/*.gcda")
            else()
                set(profile_files "${MARS_TITAN_PGO_DATA}")
            endif()
            foreach(profile IN LISTS profile_files)
                file(SHA256 "${profile}" profile_hash)
                mars_titan_identity_value("pgo.${profile}" "${profile_hash}")
            endforeach()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${profile_files})
        endif()

        string(SHA256 build_hash "${build_identity}")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/native-build-identity-${configuration}.txt"
            "sha256=${build_hash}\n${build_identity}")
        list(APPEND build_definitions
            "$<$<CONFIG:${active_configuration}>:MARS_TITAN_NATIVE_BUILD_SHA256=\"${build_hash}\">")
    endforeach()
    target_compile_definitions(mars_titan_financial PUBLIC
        ${build_definitions}
        "MARS_TITAN_NATIVE_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\""
        "MARS_TITAN_NATIVE_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\""
        "MARS_TITAN_NATIVE_BUILD_TYPE=\"$<CONFIG>\"")
endfunction()
