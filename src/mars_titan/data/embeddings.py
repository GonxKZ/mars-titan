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


def token_chunks(tokens: list[int], size: int = 126):
    if size < 1:
        raise ValueError("Chunk size must be positive")
    for start in range(0, len(tokens), size):
        yield tokens[start : start + size]


def add_special_tokens(tokens: list[int], cls_id: int, sep_id: int) -> list[int]:
    if type(cls_id) is not int or type(sep_id) is not int:
        raise ValueError("The pinned encoder requires explicit CLS and SEP tokens")
    return [cls_id, *tokens, sep_id]


class EmbeddingCache:
    """Una fila confirmada por representación. Los valores corruptos nunca se reutilizan."""

    def __init__(self, path: Path):
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

    def get(self, identity: dict) -> np.ndarray | None:
        key, description = self.identity(identity)
        row = self.db.execute(
            "SELECT identity,vector,checksum FROM embeddings WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != description or hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise ValueError("Embedding cache is corrupt")
        result = np.frombuffer(row[1], dtype="<f4").copy()
        if not np.isfinite(result).all() or not result.size:
            raise ValueError("Embedding cache contains corrupt values")
        return result

    def put(self, identity: dict, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype="<f4")
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a nonempty finite vector")
        key, description = self.identity(identity)
        payload = vector.tobytes()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                (key, description, payload, hashlib.sha256(payload).hexdigest()),
            )

    def close(self) -> None:
        self.db.close()


def require_cuda():
    import torch

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU fallback is disabled.")
    torch.cuda.set_device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6 * 1024**3 / total), 0)
    return torch.device("cuda:0")


class FrozenEncoders:
    """MiniLM por fragmentos completos y ResNet18 sin recortar los extremos del gráfico."""

    def __init__(self):
        import torch
        import torchvision
        import transformers
        from huggingface_hub import hf_hub_download
        from torchvision.models import ResNet18_Weights, resnet18
        from transformers import AutoModel, AutoTokenizer

        self.device = require_cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL, revision=TEXT_REVISION, trust_remote_code=False, token=False
        )
        if self.tokenizer("", add_special_tokens=True)["input_ids"] != add_special_tokens(
            [], self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        ):
            raise ValueError("Pinned tokenizer special-token layout changed")
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
            .to(self.device)
        )
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
        self.image_model = resnet18(weights=None)
        self.image_model.load_state_dict(
            weights.get_state_dict(progress=False, check_hash=True, weights_only=True)
        )
        self.image_model.fc = torch.nn.Identity()
        self.image_model = self.image_model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        self.spec = {
            "text_model": TEXT_MODEL,
            "text_revision": TEXT_REVISION,
            "text_weights_sha256": sha256(text_weights),
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
            "historical_simulation": False,
        }

    def text(self, text: str) -> np.ndarray:
        import torch

        if not text.strip():
            raise ValueError("Cannot encode missing text")
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        if not tokens:
            raise ValueError("Tokenizer produced no content")
        chunks = list(token_chunks(tokens))
        total, count = np.zeros(384, dtype=np.float64), 0
        with torch.inference_mode():
            for offset in range(0, len(chunks), 32):
                group = chunks[offset : offset + 32]
                examples = [
                    {
                        "input_ids": add_special_tokens(
                            c, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
                        )
                    }
                    for c in group
                ]
                batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
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

        if not pngs or len(pngs) > 64:
            raise ValueError("Image batch must contain between 1 and 64 charts")
        pixels = []
        for png in pngs:
            with Image.open(BytesIO(png)) as image:
                if image.size != (224, 224):
                    raise ValueError("Unexpected chart dimensions")
                pixels.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
        batch = torch.from_numpy(np.stack(pixels).transpose(0, 3, 1, 2)).to(self.device)
        with torch.inference_mode():
            return self.image_model((batch - self.mean) / self.std).float().cpu().numpy()
