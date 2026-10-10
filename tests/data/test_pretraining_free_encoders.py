"""Codificador de control sin preentrenamiento posterior.

Las referencias se calculan aquí de forma independiente: el hashing con `murmurhash3_32` y la
definición de n-gramas de palabra de scikit-learn, y la tinta por bloques con imágenes cuyo
valor esperado se conoce de antemano. No se carga ningún modelo ni se usa la GPU.
"""

import json
import unicodedata
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from mars_titan.data import corpus_encoding
from mars_titan.data import pretraining_free_encoders as pf
from mars_titan.data.charts import chart_png
from mars_titan.data.vector_carry import strict_fp32_spec
from tests.data.test_corpus_encoding import encode, prepared_edition

murmurhash3_32 = pytest.importorskip("sklearn.utils").murmurhash3_32
UP, DOWN, BACKGROUND = (23, 101, 82), (184, 65, 67), (250, 250, 250)


@pytest.fixture(scope="module")
def encoders():
    return pf.PretrainingFreeEncoders()


def reference_text(text):
    """Hashing con signo de n-gramas de 2 a 4 caracteres dentro de cada palabra, norma L2."""
    words = unicodedata.normalize("NFKC", text).lower().split()
    counts = np.zeros(pf.TEXT_WIDTH)
    for word in words:
        padded, grams = f" {word} ", []
        for n in range(2, 5):
            count = len(padded) - n + 1
            # Una palabra que no supera n caracteres cuenta una sola vez y no sigue con n + 1.
            if count <= 1:
                grams.append(padded[:n])
                break
            grams.extend(padded[start : start + n] for start in range(count))
        for gram in grams:
            value = murmurhash3_32(gram, seed=0, positive=False)
            counts[abs(value) % pf.TEXT_WIDTH] += 1.0 if value >= 0 else -1.0
    return (counts / np.linalg.norm(counts)).astype(np.float32)


def png(blocks=(), size=pf.SIDE, background=BACKGROUND):
    """PNG con bloques de 14 × 14 píxeles de un color en las posiciones (fila, columna)."""
    image = Image.new("RGB", (size, size), background)
    for (row, column), color in blocks:
        left, top = column * pf.BLOCK, row * pf.BLOCK
        image.paste(color, (left, top, left + pf.BLOCK, top + pf.BLOCK))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def ink(color, channel):
    return 1.0 - color[channel] / 255.0


@pytest.mark.parametrize(
    "text",
    [
        "Alcoa reports a quarterly loss and cuts its dividend.",
        "中金：受疫情影响行业景气度下行，预计今年上市银行净利润增速同比下滑至3.7%。",
        "a I x",
        "  Varias   líneas\ncon\tespacios  ",
    ],
)
def test_text_is_the_signed_hashing_of_word_ngrams(encoders, text):
    vector = encoders.text(text)
    assert vector.dtype == np.float32 and vector.shape == (pf.TEXT_WIDTH,)
    np.testing.assert_allclose(vector, reference_text(text), rtol=0, atol=1e-7)
    assert np.linalg.norm(vector.astype(np.float64)) == pytest.approx(1.0, abs=1e-6)


def test_text_normalises_width_and_case_and_reads_the_whole_document(encoders):
    assert np.array_equal(encoders.text("ＡＬＣＯＡ Results"), encoders.text("alcoa results"))
    long = "precio " * 3_000
    assert not np.array_equal(encoders.text(long), encoders.text(long + "quiebra"))


@pytest.mark.parametrize("text", ["", "   \n", None, b"bytes"])
def test_absent_text_is_rejected(encoders, text):
    with pytest.raises(ValueError, match="texto ausente"):
        encoders.text(text)


def test_vectors_do_not_depend_on_previous_inputs(encoders):
    fresh = pf.PretrainingFreeEncoders()
    for text in ("primero", "segundo texto", "tercero"):
        encoders.text(text)
    assert np.array_equal(encoders.text("segundo texto"), fresh.text("segundo texto"))
    chart = png([((3, 4), UP)])
    encoders.images([png([((0, 0), DOWN)])])
    assert np.array_equal(encoders.images([chart]), fresh.images([chart]))


def test_charts_average_red_and_green_ink_by_block_in_channel_major_order(encoders):
    chart = png([((2, 5), UP), ((15, 0), DOWN)])
    vector = encoders.images([chart])[0]
    expected = np.empty((2, 16, 16))
    for channel in range(2):
        expected[channel] = ink(BACKGROUND, channel)
        expected[channel, 2, 5] = ink(UP, channel)
        expected[channel, 15, 0] = ink(DOWN, channel)
    np.testing.assert_allclose(vector, expected.reshape(-1), rtol=0, atol=1e-7)
    # La tinta roja separa una vela alcista de una bajista.
    assert vector[2 * 16 + 5] > 0.9 and vector[15 * 16] < 0.3


def test_a_partial_block_is_the_exact_mean_of_its_pixels(encoders):
    image = Image.new("RGB", (pf.SIDE, pf.SIDE), BACKGROUND)
    image.paste(UP, (0, 0, 7, 14))
    output = BytesIO()
    image.save(output, format="PNG")
    vector = encoders.images([output.getvalue()])[0]
    for channel in range(2):
        mean = (ink(UP, channel) + ink(BACKGROUND, channel)) / 2
        assert vector[channel * 256] == pytest.approx(mean, abs=1e-7)


def test_a_rendered_chart_keeps_each_candle_in_its_column(encoders):
    rising = np.array([[10 + i, 12 + i, 9 + i, 11 + i] for i in range(64)], dtype=np.float64)
    vector = encoders.images([chart_png(rising, end_index=63)])[0]
    red = vector[:256].reshape(16, 16)
    assert np.isfinite(vector).all() and 0 <= vector.min() and vector.max() <= 1
    # Precios crecientes: la tinta sube de la parte baja a la alta de izquierda a derecha.
    assert np.argmax(red[:, 1]) > np.argmax(red[:, 14])


def test_batches_match_single_images_and_bad_batches_are_rejected(encoders):
    charts = [png([((row, row), UP)]) for row in range(5)]
    batch = encoders.images(charts)
    assert batch.shape == (5, pf.IMAGE_WIDTH) and batch.dtype == np.float32
    for index, chart in enumerate(charts):
        assert np.array_equal(batch[index], encoders.images([chart])[0])
    with pytest.raises(ValueError, match="entre 1 y 64"):
        encoders.images([])
    with pytest.raises(ValueError, match="entre 1 y 64"):
        encoders.images([charts[0]] * 65)
    with pytest.raises(ValueError, match="dimensiones"):
        encoders.images([png(size=223)])


def test_identity_records_the_rules_and_changes_with_them(encoders, monkeypatch):
    spec = encoders.spec
    assert spec == pf.pretraining_free_spec() and json.loads(json.dumps(spec)) == spec
    assert spec["family"] == pf.FAMILY and spec["learned_parameters"] == 0
    assert spec["historical_simulation"] is True and spec["device"] == "cpu"
    assert spec["text_hashing"]["ngram_range"] == [2, 4]
    assert strict_fp32_spec(spec)
    monkeypatch.setitem(pf.HASHING, "ngram_range", (1, 3))
    changed = pf.pretraining_free_spec()
    assert changed["probe_sha256"] != spec["probe_sha256"]
    assert changed["text_hashing"] != spec["text_hashing"]


def test_the_probe_also_follows_the_image_rule(encoders, monkeypatch):
    # Bloques de 28 píxeles darían 8 × 8 por canal: la huella de la sonda debe cambiar.
    monkeypatch.setattr(pf, "BLOCK", 28)
    monkeypatch.setattr(pf, "IMAGE_WIDTH", 128)
    assert pf.pretraining_free_spec()["probe_sha256"] != encoders.spec["probe_sha256"]


def test_the_corpus_encoding_writes_the_control_vectors_and_identity(tmp_path):
    class Recording(pf.PretrainingFreeEncoders):
        def __init__(self):
            super().__init__()
            self.texts, self.charts = [], []

        def text(self, text):
            self.texts.append(super().text(text))
            return self.texts[-1]

        def images(self, pngs):
            self.charts.extend(super().images(pngs))
            return np.stack(self.charts[-len(pngs) :])

    manifest, clock, macro = prepared_edition(tmp_path)
    recording = Recording()
    output = tmp_path / "encoded"
    result = encode(
        manifest, output, macros={"US": macro}, encoders=recording, clocks={"US": clock}, context=2
    )
    assert result["cohort_complete"] is True and result["samples"] == 5
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["encoders"] == recording.spec
    table = pq.read_table(output / "samples/US/A/samples.parquet").to_pylist()
    texts = {np.asarray(vector, dtype=np.float32).tobytes() for vector in recording.texts}
    charts = {np.asarray(vector, dtype=np.float32).tobytes() for vector in recording.charts}
    for row in table:
        chart = np.asarray(row["charts"], dtype=np.float32)
        assert chart.tobytes() in charts
        if row["news_count"] == 1:
            assert np.asarray(row["news"], dtype=np.float32).tobytes() in texts
    assert any(row["news_count"] == 1 for row in table)


def summary():
    return dict(
        cohort_id="fixture",
        candidate_count=1,
        samples=1,
        failed_assets=0,
        cohort_complete=True,
        reused_assets=0,
    )


def test_the_command_line_selects_the_control_without_cuda_options(monkeypatch):
    import sys

    import torch

    captured = []
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(
        corpus_encoding, "encode_corpus", lambda *a, **k: captured.append(k) or summary()
    )
    arguments = ["encode", "--prepared", "p", "--output", "o", "--encoders", "pretraining_free"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert corpus_encoding.main() == 0
    assert isinstance(captured[0]["encoders"], pf.PretrainingFreeEncoders)
    assert captured[0]["encoder_options"] is None
    monkeypatch.setattr(sys, "argv", arguments[:5])
    corpus_encoding.main()
    assert captured[1]["encoders"] is None and captured[1]["encoder_options"]


@pytest.mark.parametrize("flag", [["--reuse-only"], ["--collect"], ["--text-carry", "t"]])
def test_the_control_rejects_the_gpu_passes(monkeypatch, flag):
    import sys

    arguments = ["encode", "--prepared", "p", "--output", "o", "--encoders", "pretraining_free"]
    monkeypatch.setattr(sys, "argv", [*arguments, *flag])
    with pytest.raises(SystemExit):
        corpus_encoding.main()
