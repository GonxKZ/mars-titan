# Implementaciones C y C++

Directorio reservado para funciones nativas que respondan a un coste observado mediante perfilado. La configuración actual declara C17 y C++20, pero no contiene fuentes, algoritmos ni extensiones.

Antes de introducir una función se definirá una referencia en Python, su contrato y un criterio de mejora medible. La comparación deberá considerar error numérico, memoria, latencia y coste de mover datos entre componentes. Una optimización no puede cambiar el orden de información ni la definición de las etiquetas.

Los futuros objetivos usarán `mars_titan::native_options` para compartir los estándares y avisos de compilación definidos en CMake. El mecanismo de enlace con Python queda pendiente.
