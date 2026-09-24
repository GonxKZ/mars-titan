# Contratos públicos nativos

`mars_titan/simulation.h` declara la interfaz C17 del cálculo contable. Define dimensiones, tamaños binarios, propiedad de los buffers y códigos de error. Las entradas se prestan durante cada llamada y los resultados usan memoria independiente.

`mars_titan/financial_session.hpp` define la cinta de mercado, las posiciones persistentes, las acciones corporativas, el estado recuperable y las políticas de referencia en C++20. Cada sesión posee su estado y puede compartir una cinta inmutable con otras sesiones.

`mars_titan/simulation_files.hpp` declara lectura Parquet, comprobación de identidades y persistencia del ejecutable autónomo. No requiere un intérprete Python. La versión inicial ejecuta validación sintética y conserva el test cerrado. La admisión histórica sigue pendiente de acreditar los ajustes OHLC y las acciones corporativas.
