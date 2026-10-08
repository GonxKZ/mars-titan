include(FetchContent)

function(mars_titan_find_candidate_zip)
    # miniz 3.1.0, revisión 174573d60290f447c13a2b1b3405de2b96e27d6c, licencia MIT.
    FetchContent_Declare(candidate_miniz
        URL https://github.com/richgel999/miniz/archive/refs/tags/3.1.0.tar.gz
        URL_HASH SHA256=09569fc19d060ac9f5999ba9356728c2494ebe6a24ac0eb0a6b6ae3d396cfea6
        TLS_VERIFY TRUE DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_TESTS OFF)
    set(INSTALL_PROJECT OFF)
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(candidate_miniz)
    # No interponer estos símbolos en la copia interna de miniz de LibTorch.
    set_target_properties(miniz PROPERTIES C_VISIBILITY_PRESET hidden)
    target_compile_definitions(miniz PUBLIC MINIZ_NO_ZLIB_COMPATIBLE_NAMES
        PRIVATE MINIZ_STATIC_DEFINE _POSIX_C_SOURCE=200809L)
    get_target_property(miniz_includes miniz INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET miniz PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${miniz_includes}")
    mars_titan_instrument_target(miniz)
    configure_file("${candidate_miniz_SOURCE_DIR}/LICENSE"
        "${CMAKE_CURRENT_BINARY_DIR}/candidate-miniz-LICENSE" COPYONLY)
endfunction()
