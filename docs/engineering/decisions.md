# Registro de decisiones

Las decisiones documentan el motivo y la condición que justificaría revisarlas. No sustituyen resultados experimentales.

| ID | Decisión | Motivo | Revisar cuando |
| --- | --- | --- | --- |
| D01 | Monorrepositorio por responsabilidades. | Un autor, una comparación y cambios coordinados entre datos, modelos y memoria. | Aparezcan componentes con mantenimiento y ciclos de publicación independientes. |
| D02 | Repositorio privado con MIT para material propio. | Permite preparar el trabajo conservando control de publicación y separando derechos ajenos. | Se acuerde publicación de la memoria y de los recursos derivados. |
| D03 | Python con uv y entorno bloqueado. CUDA explícita. | Reproducibilidad y coherencia con el equipo disponible. | Una dependencia necesaria no sea compatible con la versión elegida. |
| D04 | Estados de memoria fuera de datos y evaluación. | Evitar acoplamientos que permitan leer etiquetas futuras o reutilizar estados contaminados. | Solo mediante una revisión del contrato temporal. |
| D05 | Global antes de jerarquía completa. Modelos compactos. | Permite aislar la aportación de memoria dentro de 8 GB. | El piloto muestre margen y la comparación global esté cerrada. |
| D06 | Simulación apertura-cierre tras señal del día anterior. | Separa claramente información observada y primera ejecución posible. | La auditoría demuestre otro reloj de decisión y precios ejecutables adecuados. |
| D07 | PDF externos locales, catálogos y hashes versionados. | Acceso gratuito no garantiza redistribución y los libros añaden volumen al historial. | Exista permiso claro y una necesidad concreta de distribuirlos. |
| D08 | C++/CUDA optativos y guiados por perfil. | La comparación científica tiene prioridad sobre optimización prematura. | Un cuello de botella medido justifique el coste de implementación y validación. |
| D09 | No iniciar código científico durante la preparación. | El alcance actual es dejar lista investigación, documentación, configuración y planificación. | Se solicite iniciar una tarea de implementación del tablero. |
