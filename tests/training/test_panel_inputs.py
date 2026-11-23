"""Extensión de entradas de panel en cohortes, campaña, corpus, índices y copias empaquetadas."""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data import panel_signals, technical_features
from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory import event_archive, financial_observations, training_event_cache
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.learning_scale import LearningScale
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training import comparison_cohorts as cohorts
from mars_titan.training import window_indices
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_comparison_campaign import cn_source
from tests.training.test_comparison_cohorts import joint_source, write_source
from tests.training.test_historical_corpus_inputs import dump, read_all, supervised
from tests.training.test_screening_cohorts import source as us_source

ROOT = Path(__file__).resolve().parents[2]
PANEL = panel_signals.NAME_PRICE_VOLUME_16_V1
K = len(panel_signals.PANELS[PANEL])


def _asset_days_and_prices(meta, market="US", symbol="AAA"):
    path = Path(meta["roots"]["prepared"]) / market / symbol / "prices.parquet"
    prices = pq.read_table(path)
    values = np.column_stack(
        [prices[c].to_numpy() for c in ("open", "high", "low", "close", "volume")]
    )
    stamps = prices["available_at"].cast(pa.timestamp("us")).cast(pa.int64()).to_numpy()
    days = stamps.astype("datetime64[us]").astype("datetime64[D]")
    return values, days


def _make_day_coded_panel(root, days, symbols, *, market="US", scale=0.01, offset=0.0):
    """Construir un artefacto donde la sesión t y la señal k codifican inequívocamente el día t."""
    sessions = [str(day) for day in days if str(day) < "2024-01-01"]
    symbols = sorted(symbols)
    arr = np.zeros((len(sessions), len(symbols), K), dtype=np.float32)
    for t in range(len(sessions)):
        for n in range(len(symbols)):
            for k in range(K):
                arr[t, n, k] = np.float32(
                    np.clip(offset + scale * (t + 1) + 0.001 * k + 0.05 * n, -8.0, 8.0)
                )
    return (
        panel_signals.write_panel(
            root,
            name=PANEL,
            market=market,
            sessions=sessions,
            symbols=symbols,
            values=arr,
            source={"fixture": "day_coded"},
        ),
        sessions,
        arr,
    )


def _prepared_view(tmp_path):
    """Vista supervisada histórica con convención del factor de mercado declarada."""
    path = supervised(tmp_path)
    meta = json.loads(path.read_text())
    declared = dict(meta["market_factors"]["US"], return_convention="close_over_open_minus_one")
    meta = dict(meta, market_factors=dict(US=declared))
    dump(path, meta)
    return path, meta


# ---------------------------------------------------------------------------
# 1. Cohortes (`comparison_cohorts`)
# ---------------------------------------------------------------------------


def test_checked_extensions_accepts_and_validates_panel_and_markets():
    sha_us, sha_cn = "a" * 64, "b" * 64
    ext_us = dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha_us)))
    ext_req = dict(ext_us, market_model="required")
    ext_joint = dict(
        technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha_us, CN=sha_cn))
    )
    assert cohorts.checked_extensions(ext_us) == ext_us
    assert cohorts.checked_extensions(ext_us, markets=["US"]) == ext_us
    assert cohorts.checked_extensions(ext_req, markets=["US"]) == ext_req
    assert cohorts.checked_extensions(ext_joint, markets=["CN", "US"]) == ext_joint

    # Sin `technical`, el panel no se admite.
    with pytest.raises(ValueError, match="extensión de entradas|technical"):
        cohorts.checked_extensions(dict(panel=dict(name=PANEL, sha256=dict(US=sha_us))))
    # Panel desconocido, forma inválida o mercados que no coinciden se rechazan.
    for wrong in (
        dict(technical="long_horizon_v5", panel=None),
        dict(technical="long_horizon_v5", panel=PANEL),
        dict(technical="long_horizon_v5", panel=dict(name="otro", sha256=dict(US=sha_us))),
        dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US="corto"))),
        dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256={})),
    ):
        with pytest.raises(ValueError):
            cohorts.checked_extensions(wrong)
    with pytest.raises(ValueError, match="mercado"):
        cohorts.checked_extensions(ext_us, markets=["CN"])
    with pytest.raises(ValueError, match="mercado"):
        cohorts.checked_extensions(ext_us, markets=["US", "CN"])
    with pytest.raises(ValueError, match="mercado"):
        cohorts.checked_extensions(ext_joint, markets=["US"])


def test_cohorts_and_full_views_write_panel_in_manifest_and_receipt_and_change_identity(tmp_path):
    origin = tmp_path / "views/us/manifest.json"
    atomic_json(origin, us_source())
    sha_a, sha_b = "a" * 64, "b" * 64
    tech = dict(technical="long_horizon_v5")
    panel_a = dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha_a)))
    panel_b = dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha_b)))

    plain_m, plain_r = cohorts.write_cohort(origin, tmp_path / "tech", size=16, extensions=tech)
    m_a, r_a = cohorts.write_cohort(origin, tmp_path / "panel_a", size=16, extensions=panel_a)
    m_b, r_b = cohorts.write_cohort(origin, tmp_path / "panel_b", size=16, extensions=panel_b)

    assert "panel" not in json.loads(plain_m.read_text())["input_extensions"]
    assert "panel" not in plain_r["input_extensions"]
    assert json.loads(m_a.read_text())["input_extensions"] == panel_a
    assert r_a["input_extensions"] == panel_a
    assert sha256(m_a) != sha256(plain_m)
    assert sha256(m_a) != sha256(m_b)
    # Reutilizar la misma cohorte con el mismo panel pasa; con otro sha256 se rechaza.
    assert cohorts.write_cohort(origin, tmp_path / "panel_a", size=16, extensions=panel_a) == (
        m_a,
        r_a,
    )
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "panel_a", size=16, extensions=panel_b)
    # Una cohorte US rechaza un panel declarado para CN o US+CN.
    with pytest.raises(ValueError, match="mercado"):
        cohorts.write_cohort(
            origin,
            tmp_path / "bad_mkt",
            size=16,
            extensions=dict(
                technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(CN=sha_a))
            ),
        )

    # Vista completa conjunta US+CN: exige ambos mercados exactos.
    joint_path = write_source(tmp_path / "views/joint/manifest.json", joint_source(2019))
    ext_joint = dict(
        technical="long_horizon_v5",
        panel=dict(name=PANEL, sha256=dict(US=sha_a, CN=sha_b)),
    )
    jm, jr = cohorts.write_full_view(
        joint_path, tmp_path / "full_joint", size=341, extensions=ext_joint
    )
    assert json.loads(jm.read_text())["input_extensions"] == ext_joint
    assert jr["input_extensions"] == ext_joint
    with pytest.raises(ValueError, match="mercado"):
        cohorts.write_full_view(joint_path, tmp_path / "bad_joint", size=341, extensions=panel_a)

    # Vista completa de China sola: exige solo CN.
    cn_path = write_source(tmp_path / "views/cn/manifest.json", cn_source(2019))
    ext_cn = dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(CN=sha_b)))
    cm, cr = cohorts.write_full_view(cn_path, tmp_path / "full_cn", size=301, extensions=ext_cn)
    assert json.loads(cm.read_text())["input_extensions"] == ext_cn
    assert cr["input_extensions"] == ext_cn
    with pytest.raises(ValueError, match="mercado"):
        cohorts.write_full_view(cn_path, tmp_path / "bad_cn", size=301, extensions=panel_a)


# ---------------------------------------------------------------------------
# 2. Declaración de la campaña (`comparison_campaign.checked_declaration`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config_name", "market"),
    [
        ("us-campaign-v12-features-v5.json", "US"),
        ("cn-campaign-v12-features-v5.json", "CN"),
    ],
)
def test_checked_declaration_requires_k_plus_c_columns_with_panel_and_exact_market(
    config_name, market
):
    base = json.loads((ROOT / "configs/experiments" / config_name).read_text(encoding="utf-8"))
    # Sin panel la declaración actual pasa intacta con 46 columnas.
    campaign.checked_declaration(base, root=ROOT)

    sha = "c" * 64
    with_panel = copy.deepcopy(base)
    with_panel["inputs"] = dict(
        base["inputs"],
        panel=dict(name=PANEL, sha256={market: sha}),
    )
    # Con panel declarado y rama técnica aún en 46 columnas, debe rechazarse con mensaje claro.
    with pytest.raises(ValueError, match="rama técnica.*62|62"):
        campaign.checked_declaration(with_panel, root=ROOT)

    # Al cambiar solo `technical_branch.columns` a 46 + 16 = 62, la declaración pasa.
    recipe = json.loads((ROOT / base["recipes"]["core"]).read_text(encoding="utf-8"))
    branch = dict(recipe["predictor"]["technical_branch"], columns=46 + K)
    candidate = copy.deepcopy(with_panel)
    candidate["models"]["titans_mac"]["predictor_changes"] = dict(technical_branch=branch)
    campaign.checked_declaration(candidate, root=ROOT)

    # Cualquier otro ancho (61, 63, 16) se rechaza.
    for wrong_cols in (16, 46, 61, 63):
        bad = copy.deepcopy(candidate)
        bad["models"]["titans_mac"]["predictor_changes"] = dict(
            technical_branch=dict(branch, columns=wrong_cols)
        )
        with pytest.raises(ValueError, match="rama técnica"):
            campaign.checked_declaration(bad, root=ROOT)

    # Mercado equivocado o extra en panel.sha256 se rechaza.
    other = "CN" if market == "US" else "US"
    for bad_sha in ({other: sha}, {"US": sha, "CN": sha}):
        bad = copy.deepcopy(candidate)
        bad["inputs"]["panel"]["sha256"] = bad_sha
        with pytest.raises(ValueError, match="mercado"):
            campaign.checked_declaration(bad, root=ROOT)


def test_checked_declaration_preserves_named_columns_when_panel_precedes_technical():
    screen1 = json.loads(
        (ROOT / "configs/experiments/us-campaign-v11-screen1.json").read_text(encoding="utf-8")
    )
    sha = "d" * 64
    recipe_icop = json.loads((ROOT / screen1["recipes"]["core_icop"]).read_text(encoding="utf-8"))
    assert recipe_icop["recipe"]["session_ic_term"]["offset_column"] == -46
    branch_62 = dict(recipe_icop["predictor"]["technical_branch"], columns=46 + K)

    # Con panel (16) + long_horizon_v5 (46), alpha_percent sigue estando en -46 desde el final.
    ok_icop = dict(
        screen1,
        inputs=dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha))),
        tiers=["y1_operable"],
        recipes=dict(core_icop=screen1["recipes"]["core_icop"]),
        models=dict(
            titans_mac_icop_panel=dict(
                screen1["models"]["titans_mac_icop"],
                predictor_changes=dict(technical_branch=branch_62),
            )
        ),
    )
    campaign.checked_declaration(ok_icop, root=ROOT)

    # Con long_horizon_v4 (62) + panel (16) = 78, offset_column=-46 sigue rechazándose.
    branch_78 = dict(branch_62, columns=62 + K)
    bad_icop = dict(
        ok_icop,
        inputs=dict(technical="long_horizon_v4", panel=dict(name=PANEL, sha256=dict(US=sha))),
        models=dict(
            titans_mac_icop_panel=dict(
                screen1["models"]["titans_mac_icop"],
                predictor_changes=dict(technical_branch=branch_78),
            )
        ),
    )
    with pytest.raises(ValueError, match="desplazamiento conocido"):
        campaign.checked_declaration(bad_icop, root=ROOT)

    # También la columna de liquidez (-29 en v2/v5) sigue valiendo con panel + v5.
    v10 = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v10.json").read_text(encoding="utf-8")
    )
    liq_recipe = json.loads((ROOT / v10["recipes"]["core_liq1"]).read_text(encoding="utf-8"))
    assert liq_recipe["recipe"]["session_ic_term"]["liquidity_column"] == -29
    liq_model = dict(
        v10["models"]["titans_mac_liq1"],
        predictor_changes=dict(technical_branch=branch_62),
    )
    ok_liq = {
        key: value
        for key, value in v10.items()
        if key not in ("analysis", "models", "tiers", "recipes", "inputs")
    }
    ok_liq.update(
        tiers=["r4_liquid"],
        recipes=dict(core_liq1=v10["recipes"]["core_liq1"]),
        models=dict(titans_mac_liq1=liq_model),
        inputs=dict(technical="long_horizon_v5", panel=dict(name=PANEL, sha256=dict(US=sha))),
    )
    campaign.checked_declaration(ok_liq, root=ROOT)


# ---------------------------------------------------------------------------
# 3. Lectura en el corpus (`corpus_inputs.py`) y causalidad
# ---------------------------------------------------------------------------


def test_corpus_places_panel_immediately_before_technical_and_caches_per_asset(
    tmp_path, monkeypatch
):
    path, meta = _prepared_view(tmp_path)
    values, days = _asset_days_and_prices(meta)
    table = pq.read_table(tmp_path / "samples/US/AAA/samples.parquet")
    ends = dict(
        zip(
            table["prediction_at"].cast(pa.timestamp("us")).cast(pa.int64()).to_pylist(),
            table["price_end_index"].to_pylist(),
            strict=True,
        )
    )

    # Sin panel, lectura base y técnica-sola funcionan sin MARS_TITAN_PANEL_SIGNALS.
    monkeypatch.delenv(panel_signals.ENV, raising=False)
    plain = read_all(path)
    dump(path, dict(meta, input_extensions=dict(technical=technical_features.NAME_V5)))
    tech_only = read_all(path)

    # Con panel declarado: [gráfico | panel (16) | técnicas (46)].
    panel_dir = tmp_path / "panels"
    (_, digest), sessions, arr = _make_day_coded_panel(panel_dir, days, ["AAA"])
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest)),
            ),
        ),
    )

    calls = []
    real_rows = panel_signals.PanelSignals.rows

    def counted_rows(self, symbol, query_days):
        calls.append(symbol)
        return real_rows(self, symbol, query_days)

    monkeypatch.setattr(panel_signals.PanelSignals, "rows", counted_rows)
    ds = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    with_panel = list(ds.batches(partition="train", batch_size=1, epoch=0, seed=42))
    # También leer observaciones sobre el mismo dataset reutiliza la caché por activo.
    obs = list(
        ds.observation_batches(start=946_684_800_000_000, end=1_704_067_200_000_000, batch_size=2)
    )
    assert calls == ["AAA"]
    assert len(with_panel) == len(plain) > 0 and len(obs) > 0

    base_width = plain[0]["inputs"]["charts"].shape[1]
    tech_width = len(technical_features.VERSIONS[technical_features.NAME_V5])
    session_index = {day: idx for idx, day in enumerate(sessions)}

    for before, only_t, after in zip(plain, tech_only, with_panel, strict=True):
        charts = after["inputs"]["charts"]
        assert charts.dtype == np.float32
        assert charts.shape == (1, base_width + K + tech_width)
        # Prefijo visual intacto.
        np.testing.assert_array_equal(charts[:, :base_width], before["inputs"]["charts"])
        # Cola técnica intacta en las últimas C columnas (mismas posiciones negativas).
        np.testing.assert_array_equal(
            charts[:, -tech_width:], only_t["inputs"]["charts"][:, -tech_width:]
        )
        # Panel justo antes de las técnicas: fila del día de `price_end_index`.
        instant = int(after["prediction_at"][0].astype("datetime64[us]").astype(np.int64))
        end_idx = ends[instant]
        day_str = str(days[end_idx])
        expected_panel = arr[session_index[day_str], 0]
        np.testing.assert_array_equal(charts[0, base_width : base_width + K], expected_panel)
        # Otras modalidades, presencia y etiqueta intactas.
        for name in ("prices", "news", "fundamentals", "macro"):
            np.testing.assert_array_equal(after["inputs"][name], before["inputs"][name])
        np.testing.assert_array_equal(after["presence"], before["presence"])
        np.testing.assert_array_equal(after["target"], before["target"])


def test_causality_row_receives_day_d_not_d_plus_1_and_future_changes_do_not_leak(
    tmp_path, monkeypatch
):
    path, meta = _prepared_view(tmp_path)
    _, days = _asset_days_and_prices(meta)
    table = pq.read_table(tmp_path / "samples/US/AAA/samples.parquet")
    ends = dict(
        zip(
            table["prediction_at"].cast(pa.timestamp("us")).cast(pa.int64()).to_pylist(),
            table["price_end_index"].to_pylist(),
            strict=True,
        )
    )
    sessions = [str(day) for day in days if str(day) < "2024-01-01"]
    session_index = {day: idx for idx, day in enumerate(sessions)}
    # Codificar exactamente el índice del día d en todas las columnas (d * 0.01).
    arr = np.zeros((len(sessions), 1, K), dtype=np.float32)
    for t in range(len(sessions)):
        arr[t, 0, :] = np.float32(t * 0.01)

    panel_dir_a = tmp_path / "panels_a"
    _, digest_a = panel_signals.write_panel(
        panel_dir_a, name=PANEL, market="US", sessions=sessions, symbols=["AAA"], values=arr
    )
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir_a))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest_a)),
            ),
        ),
    )
    ds_a = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    obs_a = list(
        ds_a.observation_batches(start=946_684_800_000_000, end=1_704_067_200_000_000, batch_size=1)
    )
    assert len(obs_a) >= 2
    tech_width = len(technical_features.VERSIONS[technical_features.NAME_V5])
    base_width = obs_a[0]["inputs"]["charts"].shape[1] - K - tech_width

    # (a) Cada fila con price_end_index = d recibe el día d, nunca d+1 ni d-1.
    first_instant = int(obs_a[0]["prediction_at"][0].astype("datetime64[us]").astype(np.int64))
    d_idx = session_index[str(days[ends[first_instant]])]
    assert d_idx + 1 < len(sessions)
    got_d = obs_a[0]["inputs"]["charts"][0, base_width : base_width + K]
    np.testing.assert_array_equal(got_d, np.full(K, np.float32(d_idx * 0.01), dtype=np.float32))
    assert not np.array_equal(got_d, np.full(K, np.float32((d_idx + 1) * 0.01), dtype=np.float32))

    # (b) Perturbar todas las sesiones posteriores a d no altera en absoluto la fila d.
    perturbed = arr.copy()
    perturbed[d_idx + 1 :, :, :] = np.float32(-7.5)
    panel_dir_b = tmp_path / "panels_b"
    _, digest_b = panel_signals.write_panel(
        panel_dir_b,
        name=PANEL,
        market="US",
        sessions=sessions,
        symbols=["AAA"],
        values=perturbed,
    )
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir_b))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest_b)),
            ),
        ),
    )
    ds_b = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    obs_b = list(
        ds_b.observation_batches(start=946_684_800_000_000, end=1_704_067_200_000_000, batch_size=1)
    )
    np.testing.assert_array_equal(obs_b[0]["inputs"]["charts"], obs_a[0]["inputs"]["charts"])
    # Pero una fila posterior a d sí refleja el valor perturbado.
    assert not np.array_equal(obs_b[-1]["inputs"]["charts"], obs_a[-1]["inputs"]["charts"])


def test_corpus_fails_closed_on_missing_env_or_mismatched_or_2024_panel(tmp_path, monkeypatch):
    path, meta = _prepared_view(tmp_path)
    _, days = _asset_days_and_prices(meta)
    panel_dir = tmp_path / "panels"
    (_, digest), sessions, arr = _make_day_coded_panel(panel_dir, days, ["AAA"])
    valid_ext = dict(
        technical=technical_features.NAME_V5,
        panel=dict(name=PANEL, sha256=dict(US=digest)),
    )
    dump(path, dict(meta, input_extensions=valid_ext))

    # Variable de entorno ausente -> error claro, nunca ceros.
    monkeypatch.delenv(panel_signals.ENV, raising=False)
    with pytest.raises(ValueError, match=panel_signals.ENV):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    # Carpeta válida pero sin artefacto -> error claro.
    empty_dir = tmp_path / "empty_panels"
    empty_dir.mkdir()
    monkeypatch.setenv(panel_signals.ENV, str(empty_dir))
    with pytest.raises(ValueError, match="manifest.json|artefacto"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    # Huella de manifiesto distinta -> error claro.
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir))
    wrong_sha = ("0" if digest[0] != "0" else "1") + digest[1:]
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=wrong_sha)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="manifest.json|sha256|huella"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    # Panel sin technical, o con ablación de modalidades, o sin máscaras -> error.
    dump(path, dict(meta, input_extensions=dict(panel=dict(name=PANEL, sha256=dict(US=digest)))))
    with pytest.raises(ValueError, match="extensión de entradas"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    dump(path, dict(meta, input_extensions=valid_ext))
    with pytest.raises(ValueError, match="extensión de entradas"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED, modality_ablation="mask_news")
    with pytest.raises(ValueError):
        CorpusDataset(path, input_policy=STRICT_INPUTS)

    # Mercados del panel que no coinciden con la vista.
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest, CN=digest)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="mercado"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    # Artefacto con sesión de 2024 en el manifiesto -> rechazado al abrir CorpusDataset.
    leak_dir = tmp_path / "leak_panels"
    leak_folder = leak_dir / PANEL / "US"
    leak_folder.mkdir(parents=True)
    leak_sessions = [*sessions[:2], "2024-01-02"]
    leak_arr = np.zeros((3, 1, K), dtype=np.float32)
    np.save(leak_folder / "panel.npy", leak_arr, allow_pickle=False)
    leak_manifest = {
        "schema_version": 1,
        "kind": "panel_signals_v1",
        "name": PANEL,
        "market": "US",
        "names": list(panel_signals.PANELS[PANEL]),
        "sessions": leak_sessions,
        "symbols": ["AAA"],
        "shape": [3, 1, K],
        "dtype": "float32",
        "panel_sha256": sha256(leak_folder / "panel.npy"),
        "final_test_opened": False,
        "source": {},
    }
    atomic_json(leak_folder / "manifest.json", leak_manifest)
    leak_digest = sha256(leak_folder / "manifest.json")
    monkeypatch.setenv(panel_signals.ENV, str(leak_dir))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=leak_digest)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="2024|sellado"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)

    # Símbolo ausente en el panel recibe ceros en las columnas de panel sin perder las técnicas.
    other_sym_dir = tmp_path / "other_sym"
    (_, other_digest), _, _ = _make_day_coded_panel(other_sym_dir, days, ["ZZZ"])
    monkeypatch.setenv(panel_signals.ENV, str(other_sym_dir))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=other_digest)),
            ),
        ),
    )
    batches = read_all(path)
    tech_width = len(technical_features.VERSIONS[technical_features.NAME_V5])
    charts = batches[0]["inputs"]["charts"]
    base_width = charts.shape[1] - K - tech_width
    np.testing.assert_array_equal(
        charts[0, base_width : base_width + K], np.zeros(K, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 4. Identidad de índices y copias empaquetadas
# ---------------------------------------------------------------------------


def _phase():
    return FinancialPhase(
        "train",
        946_684_800_000_000,
        946_684_800_000_000,
        1_672_531_200_000_000,
        1_672_531_200_000_000,
    )


def test_index_and_packed_archive_identity_bind_panel_name_sha256_and_module_hash(
    tmp_path, monkeypatch
):
    path, meta = _prepared_view(tmp_path)
    _, days = _asset_days_and_prices(meta)
    phase = _phase()

    # (1) Sin panel: identidad de índice e implementación de copia empaquetada sin claves nuevas.
    monkeypatch.delenv(panel_signals.ENV, raising=False)
    dump(path, dict(meta, input_extensions=dict(technical=technical_features.NAME_V5)))
    ds_plain = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    ident_plain = financial_observations.index_identity(ds_plain, phase)
    assert "panel" not in ident_plain
    assert set(ident_plain["code"]) == {
        "financial_observations",
        "corpus_inputs",
        "external_order",
    }
    plain_index = financial_observations.prepare_observation_index(
        ds_plain, tmp_path / "idx-plain", phase=phase
    )
    plain_source = financial_observations.FinancialObservationSource(ds_plain, plain_index)
    assert training_event_cache.source_implementation(plain_source) == {}

    # (2) Dos artefactos distintos (mismo nombre y forma, distinto contenido -> distinto sha256).
    root_a, root_b = tmp_path / "panels_a", tmp_path / "panels_b"
    (_, sha_a), _, _ = _make_day_coded_panel(root_a, days, ["AAA"], offset=0.1)
    (_, sha_b), _, _ = _make_day_coded_panel(root_b, days, ["AAA"], offset=-1.2)
    assert sha_a != sha_b

    shared_root = tmp_path / "shared-indices"
    cache_root = tmp_path / "packed-cache"

    # Construir índice y copia empaquetada con artefacto A.
    monkeypatch.setenv(panel_signals.ENV, str(root_a))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=sha_a)),
            ),
        ),
    )
    ds_a = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    ident_a = financial_observations.index_identity(ds_a, phase)
    assert ident_a["panel"] == dict(name=PANEL, sha256=dict(US=sha_a))
    assert ident_a["code"]["panel_signals"] == sha256(Path(panel_signals.__file__))

    monkeypatch.setenv(training_event_cache.ENV, str(cache_root))
    sources_a = window_indices.phase_sources(
        ds_a,
        {"train": phase},
        shared_root / ds_a.identity[:16],
        train_cache=True,
    )
    packed_a = sources_a["train"]
    assert isinstance(packed_a, training_event_cache.PackedPhaseSource)
    assert training_event_cache.source_implementation(packed_a) == {
        "mars_titan.memory.event_archive": sha256(Path(event_archive.__file__)),
        "mars_titan.memory.training_event_cache": sha256(Path(training_event_cache.__file__)),
        "mars_titan.data.panel_signals": sha256(Path(panel_signals.__file__)),
    }
    plain_with_panel = financial_observations.FinancialObservationSource(ds_a, packed_a.path)
    assert training_event_cache.source_implementation(plain_with_panel) == {
        "mars_titan.data.panel_signals": sha256(Path(panel_signals.__file__)),
    }

    training_event_cache.reset_usage()
    events_a = list(packed_a.batched_events(block_rows=4))
    ref_a = list(packed_a.reference_batched_events(block_rows=4))
    assert len(events_a) == len(ref_a) > 0
    for left, right in zip(events_a, ref_a, strict=True):
        for lb, rb in zip(left.inputs, right.inputs, strict=True):
            np.testing.assert_array_equal(lb["inputs"]["charts"], rb["inputs"]["charts"])
    assert training_event_cache.receipt()["builds"] == 1
    stamp_a = json.loads((cache_root / "train" / "events.json").read_text(encoding="utf-8"))
    assert stamp_a["panel"] == dict(name=PANEL, sha256=dict(US=sha_a))
    assert stamp_a["implementation"]["mars_titan.data.panel_signals"] == sha256(
        Path(panel_signals.__file__)
    )

    # (3) Cambiar al artefacto B: la clave del índice cambia y la copia empaquetada de A NO sirve.
    monkeypatch.setenv(panel_signals.ENV, str(root_b))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=sha_b)),
            ),
        ),
    )
    ds_b = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    ident_b = financial_observations.index_identity(ds_b, phase)
    assert ident_b != ident_a
    assert ds_b.identity != ds_a.identity

    sources_b = window_indices.phase_sources(
        ds_b,
        {"train": phase},
        shared_root / ds_b.identity[:16],
        train_cache=True,
    )
    packed_b = sources_b["train"]
    assert packed_b.path.parent != packed_a.path.parent

    # Reproducir directamente el archivo empaquetado de A con la fuente B se rechaza.
    with pytest.raises(ValueError, match="otro origen"):
        list(
            event_archive.replay(
                packed_b,
                packed_b.specification(),
                cache_root / "train" / "events.arrow",
                stamp_a,
                cursor=0,
                block=4,
            )
        )

    # Al abrir la misma carpeta de caché con B, la copia de A se descarta y se reconstruye con B.
    training_event_cache.reset_usage()
    events_b = list(packed_b.batched_events(block_rows=4))
    assert training_event_cache.receipt()["builds"] == 1
    charts_a = events_a[0].inputs[0]["inputs"]["charts"]
    charts_b = events_b[0].inputs[0]["inputs"]["charts"]
    assert not np.array_equal(charts_a, charts_b)
    ref_b_charts = list(packed_b.reference_batched_events(block_rows=4))[0].inputs[0]["inputs"][
        "charts"
    ]
    np.testing.assert_array_equal(charts_b, ref_b_charts)

    # (4) Alterar panel.npy en disco después de abrir la fuente se detecta al reproducir la copia.
    panel_b_file = root_b / PANEL / "US" / "panel.npy"
    raw = bytearray(panel_b_file.read_bytes())
    raw[-1] ^= 0x01
    panel_b_file.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="panel.npy|cambiado"):
        list(packed_b.batched_events(block_rows=4))

    # (5) Un cambio en la huella de `panel_signals.py` invalida la identidad del índice.
    real_sha = financial_observations.sha256

    def altered_hash(target):
        if Path(target).name == "panel_signals.py":
            return "f" * 64
        return real_sha(target)

    monkeypatch.setattr(financial_observations, "sha256", altered_hash)
    assert financial_observations.index_identity(ds_a, phase) != ident_a
    with pytest.raises(ValueError, match="identidad|cambiaron"):
        financial_observations.FinancialObservationSource(ds_a, packed_a.path)


# ---------------------------------------------------------------------------
# 5. Numérica estricta (FP32 y FP64, determinista, gradientes finitos en cero)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_technical_branch_reads_panel_plus_technical_in_fp32_and_fp64_with_finite_zero_gradients(
    tmp_path, monkeypatch, dtype
):
    path, meta = _prepared_view(tmp_path)
    _, days = _asset_days_and_prices(meta)
    panel_dir = tmp_path / "panels"
    (_, digest), _, _ = _make_day_coded_panel(panel_dir, days, ["AAA"], offset=0.25)
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest)),
            ),
        ),
    )
    ds = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    phase = _phase()
    index = financial_observations.prepare_observation_index(ds, tmp_path / "idx", phase=phase)
    source = financial_observations.FinancialObservationSource(ds, index)
    spec = source.specification()
    tech_width = len(technical_features.VERSIONS[technical_features.NAME_V5])
    branch_cols = K + tech_width
    assert spec.dimensions["charts"] >= branch_cols

    # Extraer el bloque real de entrada [1, C_total] y duplicarlo con variación para sesión n=2.
    first_batch = next(ev.inputs[0] for ev in source.events() if ev.inputs)
    charts_np = first_batch["inputs"]["charts"]
    charts_two = np.concatenate([charts_np, charts_np * np.float32(0.5)], axis=0)
    charts = torch.tensor(charts_two, dtype=dtype, requires_grad=True)

    from mars_titan.models.titans.technical_branch import build_technical_branch

    cfg = TechnicalDenseConfig(columns=branch_cols, hidden=16, depth=2, session_ranks=True)
    branch = build_technical_branch(32, cfg, clip=5.0, seed=42, dtype=dtype)
    # Al arrancar, la última capa vale cero -> residuo exactamente cero y gradientes finitos.
    out = branch(charts, markets=["US", "US"])
    assert out.dtype == dtype and out.shape == (2, 32)
    assert torch.equal(out, torch.zeros((2, 32), dtype=dtype))
    loss = (out - torch.tensor(0.5, dtype=dtype)).pow(2).mean()
    loss.backward()
    for name, param in branch.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    # Verificar que la rama responde exactamente a un cambio en las columnas del panel.
    with torch.no_grad():
        branch.outer.weight.fill_(0.05)
    base_out = branch(charts.detach(), markets=["US", "US"])
    changed = charts.detach().clone()
    changed[0, -branch_cols:-tech_width] += 1.0
    changed_out = branch(changed, markets=["US", "US"])
    assert not torch.equal(base_out, changed_out)
    # Confirmar construcción de FinancialConfig con learning_scale y la rama de K+C columnas.
    fcfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        seed=42,
        learning_scale=LearningScale(clip=5.0),
        technical_branch=cfg,
    )
    predictor = FinancialPredictor(fcfg, dtype=dtype)
    assert predictor.technical_branch.config.columns == branch_cols


def test_joint_campaign_declaration_and_reader_models_validate_with_panel():
    base = json.loads(
        (ROOT / "configs/experiments/us-cn-campaign-v11-inputs-v2.json").read_text(encoding="utf-8")
    )
    recipe = json.loads((ROOT / base["recipes"]["core_nogain"]).read_text(encoding="utf-8"))
    branch_62 = dict(recipe["predictor"]["technical_branch"], columns=46 + K)
    sha_us, sha_cn = "a" * 64, "b" * 64
    joint = copy.deepcopy(base)
    joint["inputs"] = dict(
        technical="long_horizon_v5",
        panel=dict(name=PANEL, sha256=dict(US=sha_us, CN=sha_cn)),
    )
    for model in joint["models"].values():
        model["predictor_changes"] = dict(technical_branch=branch_62)
    joint["recipes"]["reader"] = "configs/titans/episodic-readout-historical-masked.json"
    joint["models"]["mars_titan_m1_panel"] = dict(
        machine="mars_titan",
        tier="r4_nogain",
        runner="reader",
        recipe="reader",
        parent="titans_mac_nogain",
        components=dict(episodic_bank="m1"),
        search_case="lr1e-3",
    )
    campaign.checked_declaration(joint, root=ROOT)

    # Falta CN en campaña US+CN -> rechazo.
    missing_cn = copy.deepcopy(joint)
    missing_cn["inputs"]["panel"]["sha256"] = dict(US=sha_us)
    with pytest.raises(ValueError, match="mercado"):
        campaign.checked_declaration(missing_cn, root=ROOT)

    # Rama técnica sin `columns` en un modelo de GPU -> rechazo.
    no_cols = copy.deepcopy(joint)
    no_cols["models"]["titans_mac_nogain"]["predictor_changes"] = dict(
        technical_branch=dict(hidden=128, depth=2)
    )
    with pytest.raises(ValueError, match="rama técnica"):
        campaign.checked_declaration(no_cols, root=ROOT)


def test_corpus_batches_detect_panel_tampering_after_open(tmp_path, monkeypatch):
    path, meta = _prepared_view(tmp_path)
    _, days = _asset_days_and_prices(meta)
    panel_dir = tmp_path / "panels"
    (_, digest), _, _ = _make_day_coded_panel(panel_dir, days, ["AAA"])
    monkeypatch.setenv(panel_signals.ENV, str(panel_dir))
    dump(
        path,
        dict(
            meta,
            input_extensions=dict(
                technical=technical_features.NAME_V5,
                panel=dict(name=PANEL, sha256=dict(US=digest)),
            ),
        ),
    )
    ds = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    assert len(list(ds.batches(partition="train", batch_size=1, epoch=0, seed=42))) == 1
    panel_file = panel_dir / PANEL / "US" / "panel.npy"
    raw = bytearray(panel_file.read_bytes())
    raw[-1] ^= 0x01
    panel_file.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="panel.npy|cambiado"):
        list(ds.batches(partition="train", batch_size=1, epoch=0, seed=42))
