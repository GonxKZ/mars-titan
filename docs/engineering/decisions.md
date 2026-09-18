# Registro de decisiones

Las decisiones documentan el motivo y la condición que justificaría revisarlas. No sustituyen resultados experimentales.

| ID | Decisión | Motivo | Revisar cuando |
| --- | --- | --- | --- |
| D01 | Monorrepositorio por responsabilidades. | Un autor, una comparación y cambios coordinados entre datos, modelos y memoria. | Aparezcan componentes con mantenimiento y ciclos de publicación independientes. |
| D02 | Repositorio público con MIT para material propio, por autorización expresa. | Permite consultar investigación y página de seguimiento. Los datos, pesos y PDF ajenos permanecen fuera de Git. El Project conserva su acceso independiente. | Cambien los derechos, la información publicada o las condiciones de acceso. |
| D03 | Python con uv y entorno bloqueado. CUDA explícita. | Reproducibilidad y coherencia con el equipo disponible. | Una dependencia necesaria no sea compatible con la versión elegida. |
| D04 | Estados de memoria fuera de datos y evaluación. | Evitar acoplamientos que permitan leer etiquetas futuras o reutilizar estados contaminados. | Solo mediante una revisión del contrato temporal. |
| D05 | Global antes de jerarquía completa. Modelos compactos. | Permite aislar la aportación de memoria dentro de 8 GB. | El piloto muestre margen y la comparación global esté cerrada. |
| D06 | Simulación apertura-cierre tras señal del día anterior. | Separa claramente información observada y primera ejecución posible. | La auditoría demuestre otro reloj de decisión y precios ejecutables adecuados. |
| D07 | PDF externos locales, catálogos y hashes versionados. | Acceso gratuito no garantiza redistribución y los libros añaden volumen al historial. | Exista permiso claro y una necesidad concreta de distribuirlos. |
| D08 | C++/CUDA optativos y guiados por perfil. | La comparación científica tiene prioridad sobre optimización prematura. | Un cuello de botella medido justifique el coste de implementación y validación. |
| D09 | No iniciar código científico durante la preparación. | El alcance actual es dejar lista investigación, documentación, configuración y planificación. | Se solicite iniciar una tarea de implementación del tablero. |
| D10 | La propuesta y O1–O6 gobiernan las ampliaciones. | La comparación multimodal residual, su memoria y la evaluación tienen prioridad sobre añadir mecanismos. | Una pregunta concreta justifique una extensión y haya datos, presupuesto y una regla de descarte. |
| D11 | Python/PyTorch para ciencia, nativo solo por perfil y Go no obligatorio. | Facilita una referencia verificable y evita mantener varios sistemas sin necesidad. | Una medida justifique una frontera pequeña con beneficio reproducible, sin alterar el protocolo. |
| D12 | La página de seguimiento consume artefactos, no dirige la investigación. | El estado observable y la presentación no deben cambiar muestras, modelos ni el test sellado. | Se acuerden publicación, privacidad y despliegue. Su indisponibilidad no detiene entrenamiento ni comparación. |
