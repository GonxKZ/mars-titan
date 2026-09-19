import importlib
import sqlite3

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.embeddings")
    except ModuleNotFoundError:
        pytest.fail("La caché verificable de representaciones todavía no existe")


def test_chunking_preserves_all_tokens_without_truncation():
    assert list(module().token_chunks([11, 12, 13, 14, 15], 2)) == [[11, 12], [13, 14], [15]]
    with pytest.raises(ValueError):
        list(module().token_chunks([11], 0))


def test_encoder_chunks_add_the_pinned_tokenizers_special_ids():
    assert module().add_special_tokens([11, 12], 0, 2) == [0, 11, 12, 2]
    with pytest.raises(ValueError):
        module().add_special_tokens([11], None, 2)


def test_cache_identity_includes_encoder_and_content(tmp_path):
    cache = module().EmbeddingCache(tmp_path / "embeddings.sqlite")
    a = {"encoder": "first", "content": "abc"}
    b = {"encoder": "second", "content": "abc"}
    assert cache.get(a) is None
    cache.put(a, np.array([1, 2], dtype=np.float32))
    assert cache.get(a).tolist() == [1, 2]
    assert cache.get(b) is None
    cache.close()


def test_cache_detects_corruption_and_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "embeddings.sqlite"
    cache = module().EmbeddingCache(path)
    with pytest.raises(ValueError):
        cache.put({"encoder": "x"}, np.array([np.nan]))
    identity = {"encoder": "x"}
    cache.put(identity, np.array([1], dtype=np.float32))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE embeddings SET vector=?", (b"broken",))
    with pytest.raises(ValueError, match="corrupt"):
        cache.get(identity)
    cache.close()
