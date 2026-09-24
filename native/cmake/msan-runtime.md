# Biblioteca estándar para MemorySanitizer

El perfil MSan necesita una biblioteca estándar instrumentada. La preparación local utiliza [LLVM 21.1.8](https://github.com/llvm/llvm-project/tree/llvmorg-21.1.8), commit `2078da43e25a4623cab2d0d60decddf709aaea28`, con Clang 21. Las bibliotecas se instalan fuera del repositorio y no cambian la biblioteca estándar de Release.

La siguiente preparación requiere Git, CMake, Ninja y Clang 21. No utiliza permisos de administrador. La descarga inicial conserva solo los directorios necesarios del mismo commit:

```bash
MARS_MSAN_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/mars-titan/toolchains/llvm-msan-21.1.8"
git clone --filter=blob:none --depth 1 --sparse --branch llvmorg-21.1.8 \
  https://github.com/llvm/llvm-project.git "$MARS_MSAN_DIR/source"
git -C "$MARS_MSAN_DIR/source" sparse-checkout set \
  runtimes libcxx libcxxabi libc cmake llvm/cmake
test "$(git -C "$MARS_MSAN_DIR/source" rev-parse HEAD)" = \
  2078da43e25a4623cab2d0d60decddf709aaea28
cmake -S "$MARS_MSAN_DIR/source/runtimes" -B "$MARS_MSAN_DIR/build" -G Ninja \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$MARS_MSAN_DIR/install" \
  '-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi' \
  -DLLVM_USE_SANITIZER=MemoryWithOrigins \
  -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_TESTS=OFF \
  -DLIBCXXABI_INCLUDE_TESTS=OFF \
  -DLIBCXX_ENABLE_SHARED=OFF -DLIBCXXABI_ENABLE_SHARED=OFF \
  -DLIBCXX_USE_COMPILER_RT=ON -DLIBCXXABI_USE_COMPILER_RT=ON \
  -DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON \
  -DLIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON \
  -DLIBCXXABI_USE_LLVM_UNWINDER=OFF \
  -DLIBCXXABI_ENABLE_STATIC_UNWINDER=OFF
cmake --build "$MARS_MSAN_DIR/build" --target install --parallel 2
```

Desde `native/`, la instalación se utiliza únicamente en el perfil instrumentado:

```bash
cmake --preset native-msan -DMARS_TITAN_BUILD_FINANCIAL=ON \
  -DMARS_TITAN_MSAN_STDLIB_ROOT="$MARS_MSAN_DIR/install"
cmake --build --preset native-msan
ctest --preset native-msan
```

`LLVM_USE_SANITIZER` instrumenta libc++ y libc++abi. El unwinder de soporte procede de libgcc y queda fuera de esa instrumentación. Compilar libunwind con MSan provoca recursión al intentar describir sus propios avisos. El código del proyecto mantiene la instrumentación completa.

CMake comprueba las cabeceras, las bibliotecas y sus símbolos de instrumentación. Las pruebas del proyecto cubren el núcleo y la sesión financiera que utilizan estas dependencias. Esta preparación no ejecuta la suite completa de LLVM ni permite atribuir cobertura MSan a Arrow u OpenSSL precompilados.
