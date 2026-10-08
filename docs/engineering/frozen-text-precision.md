# Precisión del codificador de noticias

El 8 de octubre se comparó FP32 con TF32 de matmul sobre ocho textos ya admitidos, de 103 a 5.373 tokens. Se utilizaron los mismos pesos congelados de `paraphrase-multilingual-MiniLM-L12-v2`, la fragmentación existente y lotes de ocho fragmentos. El [recibo](../../reports/resources/tf32-text-check-20261008.json) fija revisión, hashes, formas, tolerancias y repeticiones. No contiene los textos.

La prueba usó una RTX 4070 Laptop de 8.188 MiB y PyTorch 2.14.0+cu130. Otra aplicación ocupaba 6.264 MiB y no se detuvo. Se calentaron ambas opciones y se ejecutaron seis parejas, alternando el orden. El tiempo por lote incluye tokenización, inferencia, agregación y copia a CPU, con sincronización. ResNet no estuvo residente.

La mediana pasó de 149,32 ms en FP32 a 118,45 ms con TF32. El máximo error absoluto fue 2,45154e-4. En las seis parejas, 2.862 de 3.072 componentes excedieron `atol + rtol * abs(referencia)`, con `atol=2e-6` y `rtol=2e-5`. TF32 queda descartado para esta edición. No se relajan las tolerancias ni se sustituyen embeddings ya confirmados.

El pico asignado por Torch fue 499.737.600 bytes y el reservado 532.676.608, dentro del límite de 768 MiB. El proceso completo tardó 9,33 segundos, incluidas importaciones, carga y cierre, y alcanzó 2.065.724 KiB de RSS. El campo de tiempo interno empieza después de importar dependencias. Se conservaron dos intentos de configuración anteriores sin medidas de inferencia válidas.

Esta comparación describe el componente textual y no una aceleración del encoder completo o del corpus. No se midieron energía ni contadores de uso de Tensor Cores. No se ejecutó TensorRT. El código, la configuración y la edición de corpus en producción permanecen iguales. No hubo entrenamiento ni evaluación financiera.
