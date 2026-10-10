"""Codificador de control sin preentrenamiento posterior para noticias y gráficos.

Los pesos fijados de MiniLM se publicaron el 23 de junio de 2021 y ResNet18 se entrenó con
ImageNet-1K, de 2012. Su representación de una noticia o un gráfico de 2005 puede contener
conocimiento del mundo posterior a esa fecha. Este control representa las mismas entradas sin ningún
parámetro aprendido y con reglas fijas, de modo que el vector de una entrada solo depende de
ella misma:

- Texto: hashing con signo de n-gramas de caracteres (Weinberger et al., 2009) con
  `HashingVectorizer` de scikit-learn, MurmurHash3 de 32 bits con semilla 0, 384 posiciones,
  n-gramas de 2 a 4 caracteres dentro de cada palabra, minúsculas tras normalizar con NFKC y
  norma L2. No hay vocabulario ni pesos ajustados con el corpus. El texto se procesa completo,
  igual que MiniLM con sus fragmentos.
- Gráfico: el mismo PNG de 224 × 224 que recibe ResNet18 se reduce a la media de la tinta roja
  y verde (1 − canal / 255) en bloques de 14 × 14 píxeles, 16 × 16 bloques por canal y 512
  valores en total. La tinta roja separa las velas alcistas de las bajistas y la verde marca
  la presencia de cualquier vela. El canal azul se descarta porque repite esa información.

Los anchos coinciden con los del codificador congelado (384 y 512), así que
`corpus_encoding` puede calcular una edición de control con su propia identidad sin cambiar
el resto de la canalización. Se calcula en CPU con NumPy en FP64 y se redondea una sola vez a
FP32.
"""

import hashlib
import unicodedata
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .storage import sha256

FAMILY = "pretraining_free_control_v1"
TEXT_WIDTH = 384
IMAGE_WIDTH = 512
SIDE = 224
BLOCK = 14
MAX_IMAGES = 64
# Parámetros fijos del hashing. Cambiar cualquiera cambia la identidad del codificador.
HASHING = dict(
    n_features=TEXT_WIDTH,
    analyzer="char_wb",
    ngram_range=(2, 4),
    lowercase=True,
    alternate_sign=True,
    norm="l2",
)
TEXT_POLICY = "nfkc_char_wb_2_4_signed_murmurhash3_384_l2_no_truncation"
IMAGE_POLICY = "224_rgb_red_green_ink_14px_box_mean_16x16_channel_major"
# Entradas fijas cuya salida forma parte de la identidad. Si una versión de scikit-learn o de
# Pillow cambiara el resultado, la identidad cambiaría aunque los parámetros fueran los mismos.
_PROBE_TEXTS = (
    "Alcoa reports a quarterly loss and cuts its dividend.",
    "中金：预计今年上市银行净利润增速同比下滑。",
)


def _vectorizer():
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "El codificador de control necesita scikit-learn del extra research"
        ) from error
    return HashingVectorizer(dtype=np.float64, **HASHING)


def text_vector(text, vectorizer):
    """Vector de 384 valores de un texto completo, sin truncar."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No se puede codificar un texto ausente")
    values = vectorizer.transform([unicodedata.normalize("NFKC", text)]).toarray()[0]
    if values.shape != (TEXT_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("El hashing no ha producido un vector válido")
    return values.astype(np.float32)


def chart_vectors(pngs):
    """Tinta roja y verde media por bloque de cada gráfico, en una matriz [gráficos, 512]."""
    if not isinstance(pngs, list | tuple) or not 1 <= len(pngs) <= MAX_IMAGES:
        raise ValueError("El lote de imágenes debe contener entre 1 y 64 gráficos")
    side = SIDE // BLOCK
    vectors = np.empty((len(pngs), IMAGE_WIDTH), dtype=np.float32)
    for index, png in enumerate(pngs):
        with Image.open(BytesIO(png)) as image:
            if image.size != (SIDE, SIDE):
                raise ValueError("Las dimensiones del gráfico no son las esperadas")
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        ink = 1.0 - pixels[:, :, :2] / 255.0
        blocks = ink.reshape(side, BLOCK, side, BLOCK, 2).mean(axis=(1, 3))
        vectors[index] = blocks.transpose(2, 0, 1).reshape(-1)
    return vectors


def _probe_chart():
    """Gráfico fijo con una vela de cada color sobre el fondo de `charts.chart_png`."""
    image = Image.new("RGB", (SIDE, SIDE), "#fafafa")
    image.paste((23, 101, 82), (40, 60, 52, 160))
    image.paste((184, 65, 67), (120, 30, 132, 120))
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _probe_sha256(vectorizer):
    texts = [text_vector(text, vectorizer) for text in _PROBE_TEXTS]
    values = np.concatenate([*texts, chart_vectors([_probe_chart()])[0]])
    return hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()


def pretraining_free_spec(vectorizer=None):
    """Identidad del codificador de control, comparable con la de `embeddings.encoder_spec`."""
    import PIL
    import sklearn

    vectorizer = _vectorizer() if vectorizer is None else vectorizer
    return {
        "family": FAMILY,
        "text_model": "none_signed_feature_hashing",
        "text_policy": TEXT_POLICY,
        "text_hashing": dict(
            library="scikit-learn",
            vectorizer="HashingVectorizer",
            hash="murmurhash3_32_seed_0",
            unicode_normalization="NFKC",
            **{
                key: list(value) if isinstance(value, tuple) else value
                for key, value in HASHING.items()
            },
        ),
        "image_model": "none_box_mean",
        "image_policy": IMAGE_POLICY,
        "learned_parameters": 0,
        "pretraining": "none",
        "probe_sha256": _probe_sha256(vectorizer),
        "code_sha256": sha256(Path(__file__)),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "device": "cpu",
        "precision": "float32",
        # Sin PyTorch no hay TF32. Los campos son los que exige `vector_carry.strict_fp32_spec`.
        "runtime_precision": dict(
            dtype="float32",
            matmul_tf32=False,
            cudnn_tf32=False,
            computation="numpy_float64_rounded_once_to_float32",
        ),
        # El vector de una entrada solo depende de ella y de reglas fijadas sin datos, así que
        # se puede calcular con lo disponible en cada fecha.
        "historical_simulation": True,
    }


class PretrainingFreeEncoders:
    """Misma interfaz que `embeddings.FrozenEncoders`, sin modelos ni CUDA."""

    def __init__(self):
        self.vectorizer = _vectorizer()
        self.spec = pretraining_free_spec(self.vectorizer)

    def text(self, text):
        return text_vector(text, self.vectorizer)

    def images(self, pngs):
        return chart_vectors(pngs)
