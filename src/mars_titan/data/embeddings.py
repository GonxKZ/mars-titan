"""Representaciones congeladas y caché por contenido, modelo y transformación."""

import hashlib
import json
import sqlite3
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .storage import sha256

TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def text_artifact_fingerprints(snapshot: Path) -> dict[str, str]:
    """Identifica la configuración y todos los archivos del tokenizador fijado."""
    return {
        name: sha256(snapshot / name)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }


def token_chunks(tokens: list[int], size: int = 126):
    if size < 1:
        raise ValueError("El tamaño del fragmento debe ser positivo")
    for start in range(0, len(tokens), size):
        yield tokens[start : start + size]


def add_special_tokens(tokens: list[int], cls_id: int, sep_id: int) -> list[int]:
    if type(cls_id) is not int or type(sep_id) is not int:
        raise ValueError("El codificador fijado requiere tokens CLS y SEP explícitos")
    return [cls_id, *tokens, sep_id]


class EmbeddingCache:
    """Una fila confirmada por representación. Los valores corruptos nunca se reutilizan."""

    def __init__(self, path: Path, *, read_only: bool = False, cache_charts: bool = True):
        if type(cache_charts) is not bool:
            raise ValueError("La política de caché de gráficos debe ser booleana")
        self.cache_charts = cache_charts
        if read_only:
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            self.db.execute("PRAGMA query_only=ON")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, identity TEXT, vector BLOB, checksum TEXT)"
        )

    @staticmethod
    def identity(identity: dict) -> tuple[str, str]:
        text = json.dumps(identity, sort_keys=True, allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest(), text

    def get(self, identity: dict, *, max_bytes: int = 64 * 1024**2) -> np.ndarray | None:
        if type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024**2:
            raise ValueError("El presupuesto de la representación no es válido")
        if not self.cache_charts and identity.get("kind") == "chart":
            return None
        key, description = self.identity(identity)
        row = self.db.execute(
            "SELECT identity,CASE WHEN length(vector)<=? THEN vector END,checksum,length(vector) "
            "FROM embeddings WHERE key=?",
            (max_bytes, key),
        ).fetchone()
        if row is None:
            return None
        if type(row[3]) is not int or not 0 < row[3] <= max_bytes:
            raise ValueError("La representación de la caché supera su presupuesto")
        if row[0] != description or len(row[1]) % 4 or hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise ValueError("La caché de representaciones está corrupta")
        result = np.frombuffer(row[1], dtype="<f4").copy()
        if not np.isfinite(result).all() or not result.size:
            raise ValueError("La caché de representaciones contiene valores corruptos")
        return result

    def put(self, identity: dict, vector: np.ndarray) -> None:
        self.put_many([(identity, vector)])

    def put_many(self, items) -> None:
        """Guardar varios vectores en una sola transacción.

        Cada confirmación de SQLite espera a que el disco sincronice el registro, unos 12 ms en
        este equipo con carga. Agrupar las escrituras evita pagar esa espera por cada vector.
        """
        rows = []
        for identity, vector in items:
            vector = np.asarray(vector, dtype="<f4")
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError("La representación debe ser un vector no vacío de valores finitos")
            if not self.cache_charts and identity.get("kind") == "chart":
                continue
            key, description = self.identity(identity)
            payload = vector.tobytes()
            rows.append((key, description, payload, hashlib.sha256(payload).hexdigest()))
        if rows:
            with self.db:
                self.db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)", rows)

    def close(self) -> None:
        self.db.close()


def require_cuda(*, max_bytes=6 * 1024**3, min_free_bytes=0):
    import torch

    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or type(min_free_bytes) is not int
        or min_free_bytes < 0
    ):
        raise ValueError("El presupuesto CUDA no es válido")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. La sustitución por CPU está desactivada.")
    torch.cuda.set_device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    if min_free_bytes and torch.cuda.mem_get_info(0)[0] < min_free_bytes:
        raise RuntimeError("La memoria CUDA libre no alcanza la reserva de codificación")
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_bytes / total), 0)
    return torch.device("cuda:0")


def strict_fp32():
    """Desactivar TF32 en matmul y cuDNN y fijar la precisión matmul más alta de PyTorch."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _runtime_precision():
    import torch

    if torch.get_default_dtype() != torch.float32 or any(
        torch.is_autocast_enabled(device) for device in ("cpu", "cuda")
    ):
        raise ValueError("La precisión del codificador requiere FP32 global y autocast desactivado")
    return dict(
        dtype="float32",
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def _strict_precision():
    """Precisión efectiva, que debe ser FP32 estricto antes de nombrar o cargar el codificador."""
    precision = _runtime_precision()
    if (
        precision["matmul_tf32"]
        or precision["cudnn_tf32"]
        or precision["float32_matmul_precision"] != "highest"
    ):
        raise ValueError(
            "El codificador exige FP32 estricto y hay TF32 activo en matmul o cuDNN. "
            "Llama antes a strict_fp32()"
        )
    return precision


def _check_options(text_batch_size, image_batch_size, word_embedding_placement):
    if type(word_embedding_placement) is not str or word_embedding_placement not in (
        "cuda",
        "cpu",
    ):
        raise ValueError("La ubicación de la tabla de palabras debe ser cuda o cpu")
    if (
        type(text_batch_size) is not int
        or not 1 <= text_batch_size <= 32
        or type(image_batch_size) is not int
        or not 1 <= image_batch_size <= 64
    ):
        raise ValueError("Los presupuestos del codificador no son válidos")


def _tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL, revision=TEXT_REVISION, trust_remote_code=False, token=False
    )
    if tokenizer("", add_special_tokens=True)["input_ids"] != add_special_tokens(
        [], tokenizer.cls_token_id, tokenizer.sep_token_id
    ):
        raise ValueError("La disposición de tokens especiales del tokenizador fijado ha cambiado")
    return tokenizer


def encoder_spec(
    *, text_batch_size=32, image_batch_size=64, word_embedding_placement="cuda", tokenizer=None
):
    """Identidad del codificador, calculable en CPU sin cargar los modelos ni abrir CUDA.

    Las pasadas en CPU de la edición v3.1 nombran con ella los vectores que después calcula la
    GPU. `FrozenEncoders` usa esta misma función, así que las dos identidades coinciden.
    """
    _check_options(text_batch_size, image_batch_size, word_embedding_placement)
    precision = _strict_precision()
    import tokenizers
    import torch
    import torchvision
    import transformers
    from huggingface_hub import hf_hub_download
    from torchvision.models import ResNet18_Weights

    tokenizer = tokenizer if tokenizer is not None else _tokenizer()
    text_weights = Path(
        hf_hub_download(
            TEXT_MODEL,
            "model.safetensors",
            revision=TEXT_REVISION,
            local_files_only=True,
            token=False,
        )
    )
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    spec = {
        "text_model": TEXT_MODEL,
        "text_revision": TEXT_REVISION,
        "text_weights_sha256": sha256(text_weights),
        "text_artifacts_sha256": text_artifact_fingerprints(text_weights.parent),
        "tokenizer_backend_sha256": hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "text_policy": "all_126_token_chunks_weighted_mean_no_truncation",
        "image_model": "resnet18.IMAGENET1K_V1",
        "image_url": weights.url,
        "image_weights_sha256": sha256(weight_path),
        "image_policy": "224_rgb_imagenet_normalization_no_crop",
        "code_sha256": sha256(Path(__file__)),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "device": "cuda:0",
        "precision": "float32",
        "runtime_precision": precision,
        "historical_simulation": False,
        "word_embedding_placement": word_embedding_placement,
    }
    if word_embedding_placement == "cpu":
        from . import embedding_placement

        spec["word_embedding_lookup"] = {
            "policy": "cpu_word_inputs_embeds_fp32_v1",
            "code_sha256": sha256(Path(embedding_placement.__file__)),
        }
    if text_batch_size != 32 or image_batch_size != 64:
        spec["batch_sizes"] = dict(text_chunks=text_batch_size, images=image_batch_size)
    return spec


def _check_frozen_model(model):
    import torch

    if any(module.training for module in model.modules()):
        raise ValueError("Todos los módulos del codificador deben permanecer en evaluación")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Los parámetros del codificador deben permanecer congelados")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Los pesos y buffers del codificador deben conservar FP32")


class FrozenEncoders:
    """MiniLM por fragmentos completos y ResNet18 sin recortar los extremos del gráfico."""

    def __init__(
        self,
        *,
        cuda_memory_bytes=6 * 1024**3,
        min_free_cuda_bytes=0,
        text_batch_size=32,
        image_batch_size=64,
        word_embedding_placement="cuda",
    ):
        _check_options(text_batch_size, image_batch_size, word_embedding_placement)
        if (
            type(cuda_memory_bytes) is not int
            or cuda_memory_bytes <= 0
            or type(min_free_cuda_bytes) is not int
            or min_free_cuda_bytes < 0
        ):
            raise ValueError("Los presupuestos del codificador no son válidos")
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size
        self.word_embedding_placement = word_embedding_placement
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        from transformers import AutoModel

        # Falla al arrancar, antes de abrir CUDA o cargar pesos, si hay TF32 activo.
        _strict_precision()
        self.device = require_cuda(max_bytes=cuda_memory_bytes, min_free_bytes=min_free_cuda_bytes)
        self.tokenizer = _tokenizer()
        self.text_model = (
            AutoModel.from_pretrained(
                TEXT_MODEL,
                revision=TEXT_REVISION,
                trust_remote_code=False,
                use_safetensors=True,
                token=False,
            )
            .eval()
            .requires_grad_(False)
        )
        if word_embedding_placement == "cpu":
            from . import embedding_placement

            self.text_model = embedding_placement.place_cpu_word_embeddings(
                self.text_model, self.device
            )
        else:
            self.text_model = self.text_model.to(self.device)
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.image_model = resnet18(weights=None)
        self.image_model.load_state_dict(
            weights.get_state_dict(progress=False, check_hash=True, weights_only=True)
        )
        self.image_model.fc = torch.nn.Identity()
        self.image_model = self.image_model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32)[
            None, :, None, None
        ]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32)[
            None, :, None, None
        ]
        _check_frozen_model(self.text_model)
        _check_frozen_model(self.image_model)
        self.spec = encoder_spec(
            text_batch_size=text_batch_size,
            image_batch_size=image_batch_size,
            word_embedding_placement=word_embedding_placement,
            tokenizer=self.tokenizer,
        )
        self.execution_budget = dict(
            cuda_memory_bytes=cuda_memory_bytes,
            min_free_cuda_bytes=min_free_cuda_bytes,
            text_chunks=text_batch_size,
            images=image_batch_size,
        )
        self._check_precision()

    def _check_precision(self):
        if self.spec["runtime_precision"] != _runtime_precision():
            raise ValueError("La precisión efectiva cambió respecto a la identidad del codificador")

    def text(self, text: str) -> np.ndarray:
        import torch

        self._check_precision()
        _check_frozen_model(self.text_model)
        if not text.strip():
            raise ValueError("No se puede codificar un texto ausente")
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        if not tokens:
            raise ValueError("El tokenizador no ha producido contenido")
        chunks = list(token_chunks(tokens))
        total, count = np.zeros(384, dtype=np.float64), 0
        with torch.inference_mode():
            for offset in range(0, len(chunks), self.text_batch_size):
                group = chunks[offset : offset + self.text_batch_size]
                examples = [
                    {
                        "input_ids": add_special_tokens(
                            c, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
                        )
                    }
                    for c in group
                ]
                batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
                if self.word_embedding_placement == "cpu":
                    from .embedding_placement import cpu_word_values

                    batch["inputs_embeds"] = cpu_word_values(
                        self.text_model.get_input_embeddings(), batch.pop("input_ids"), self.device
                    )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                mask = batch["attention_mask"].unsqueeze(-1)
                output = self.text_model(**batch).last_hidden_state
                pooled = ((output * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
                lengths = np.array([len(c) for c in group])
                total += (pooled * lengths[:, None]).sum(0)
                count += int(lengths.sum())
        return (total / count).astype(np.float32)

    def images(self, pngs: list[bytes]) -> np.ndarray:
        import torch

        self._check_precision()
        _check_frozen_model(self.image_model)
        if not pngs or len(pngs) > 64:
            raise ValueError("El lote de imágenes debe contener entre 1 y 64 gráficos")
        outputs = []
        with torch.inference_mode():
            for offset in range(0, len(pngs), self.image_batch_size):
                pixels = []
                for png in pngs[offset : offset + self.image_batch_size]:
                    with Image.open(BytesIO(png)) as image:
                        if image.size != (224, 224):
                            raise ValueError("Las dimensiones del gráfico no son las esperadas")
                        pixels.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
                batch = torch.from_numpy(np.stack(pixels).transpose(0, 3, 1, 2)).to(self.device)
                outputs.append(
                    self.image_model((batch - self.mean) / self.std).float().cpu().numpy()
                )
        return np.concatenate(outputs)
