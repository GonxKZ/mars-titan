# Fuentes de la escritura por banco

Consulta realizada el 21 de octubre de 2026. Se registran las versiones y los apartados inspeccionados. No se han reproducido experimentos de los autores ni auditado todo su software. La implementación, la proyección de gradientes y las derivaciones de esta variante pertenecen al proyecto.

| Fuente primaria | Versión utilizada | Inspección y alcance |
| --- | --- | --- |
| Roy y colaboradores, [Brain-wide mapping reveals that engrams for a single memory are distributed across multiple brain regions](https://pmc.ncbi.nlm.nih.gov/articles/PMC8980018/) | Artículo editorial de 2022 | Resumen, figura 6 y su discusión. La reactivación multirregional y su límite motivan contrastes de cooperación. No describen esta regla numérica ni una mejora financiera. |
| Behrouz y colaboradores, [Nested Learning: The Illusion of Deep Learning Architectures](https://proceedings.neurips.cc/paper_files/paper/2025/file/4309616aaed8e848009bc4a7ef73b493-Paper-Conference.pdf) | PDF oficial de NeurIPS 2025, 35 páginas | Apéndice C, ecuación (42), y distinción entre actualizaciones internas y externas. Ya contempla agregar memorias. No se reproduce HOPE ni una jerarquía de frecuencias. |
| Du y colaboradores, [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/html/2502.13685v4) | arXiv 2502.13685v4, revisión del 18 de noviembre de 2025 | Sección 3.2 y reglas de actualización. Su selección top-k y las proyecciones por memoria difieren de la mezcla densa de este proyecto. No se reivindica novedad por enrutar estados. |
| Cai y colaboradores, [Universal Test-Time Training](https://arxiv.org/html/2610.05484v1) | arXiv 2610.05484v1, 4 de octubre de 2026 | Secciones 2–3 y apéndice C. Distingue propiedad del estado, contribuciones por capa y buffers. Esta variante no comparte una reserva entre profundidades ni adopta su pérdida o Muon. |

Se intentó acceder al artículo de Govindarajan de 2011 mediante PMC3032443, PubMed y el repositorio del MIT. Las respuestas fueron incompletas, un desafío de acceso o una redirección fallida. No se utiliza como fundamento de la implementación ni se declara una lectura de su manuscrito.

La evidencia propia separa comprobación algebraica, validación numérica, integración y efecto predictivo. La proyección P y el Jacobiano escalar se derivan en la especificación. Autograd, diferencias finitas y casos adversariales contrastan mecanismos en coma flotante. No hay comprobación Lean, certificado numérico o mejora de MAE medida. Los resultados técnicos no acreditan aprender una organización cerebral.
