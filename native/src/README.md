# Implementaciones C y C++

El núcleo C++20 ejecuta las operaciones y valora cuentas sin cortos ni deuda. `accurate_sum.hpp` comparte la suma por parciales y la conciliación del redondeo entre `simulation.cpp` y la sesión financiera.

`financial_session.cpp` conserva posiciones, órdenes, dividendos y acciones aplicadas. Confirma cada paso después de comprobar su resultado y permite recuperar un estado compatible. `financial_parquet.cpp` lee datos con Arrow C++, `simulation_files.cpp` gestiona recibos y checkpoints y `simulation_main.cpp` ejecuta las políticas desde la terminal, incluida la comparación concurrente de escenarios independientes.

Los objetivos comparten los diagnósticos de `mars_titan::native_options`. Los perfiles instrumentados están separados de Release. La referencia Python se conserva para contrastar decisiones, contabilidad y recuperación. La interfaz C permite integrarla también con Python, pero el ejecutable autónomo no utiliza ese enlace.
