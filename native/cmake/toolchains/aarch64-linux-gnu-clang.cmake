# Compilación cruzada a Linux aarch64 (DGX GB10) desde un anfitrión x86-64 con Clang y lld.
# Clang usa las cabeceras, libstdc++ y crt del GCC cruzado de la distribución
# (g++-aarch64-linux-gnu), así que la ABI coincide con la de LibTorch. Es el mismo compilador
# que los perfiles PPO nativos. No fija ninguna extensión de ISA, de modo que el binario usa
# la base armv8-a.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)

# Raíz opcional con bibliotecas aarch64 que no trae el compilador cruzado, como OpenSSL y zlib.
# Se declara una vez y pasa a los proyectos de prueba de try_compile.
if(NOT DEFINED MARS_TITAN_AARCH64_SYSROOT)
    set(MARS_TITAN_AARCH64_SYSROOT "$ENV{MARS_TITAN_AARCH64_SYSROOT}" CACHE PATH
        "Raíz adicional con bibliotecas aarch64, vacía si el compilador cruzado basta")
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES MARS_TITAN_AARCH64_SYSROOT)
set(aarch64_runtime_root /usr/aarch64-linux-gnu)
if(NOT IS_DIRECTORY "${aarch64_runtime_root}/lib")
    message(FATAL_ERROR "Falta el runtime aarch64 del compilador cruzado en ${aarch64_runtime_root}")
endif()
set(CMAKE_FIND_ROOT_PATH "${aarch64_runtime_root}")
set(aarch64_emulator_environment "")
if(MARS_TITAN_AARCH64_SYSROOT)
    if(NOT IS_DIRECTORY "${MARS_TITAN_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu")
        message(FATAL_ERROR "La raíz aarch64 declarada no contiene usr/lib/aarch64-linux-gnu")
    endif()
    list(APPEND CMAKE_FIND_ROOT_PATH "${MARS_TITAN_AARCH64_SYSROOT}")
    # Las cabeceras multiarquitectura de Debian (opensslconf.h, por ejemplo) no están en las
    # rutas por defecto del compilador cruzado.
    if(IS_DIRECTORY "${MARS_TITAN_AARCH64_SYSROOT}/usr/include/aarch64-linux-gnu")
        foreach(language C CXX)
            set(CMAKE_${language}_STANDARD_INCLUDE_DIRECTORIES
                "${MARS_TITAN_AARCH64_SYSROOT}/usr/include/aarch64-linux-gnu")
        endforeach()
    endif()
    set(aarch64_emulator_environment -E "LD_LIBRARY_PATH=${MARS_TITAN_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
# Programas del anfitrión y bibliotecas solo del destino, para no enlazar nada x86-64 por error.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CTest y los ejecutables de configuración pasan por qemu-user con el runtime del destino.
find_program(MARS_TITAN_AARCH64_EMULATOR NAMES qemu-aarch64 qemu-aarch64-static REQUIRED)
set(CMAKE_CROSSCOMPILING_EMULATOR "${MARS_TITAN_AARCH64_EMULATOR}" -L "${aarch64_runtime_root}"
    ${aarch64_emulator_environment})

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_LINKER_TYPE LLD)
