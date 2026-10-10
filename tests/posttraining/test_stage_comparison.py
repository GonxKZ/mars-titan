"""Comparación de los brazos postentrenados con su padre congelado y su continuación.

La etapa se toma reducida de ``campaign_fixture`` (GRU, semilla 42, dos ventanas US),
pero no se ejecuta: sus recibos, las fuentes de la campaña y las predicciones se escriben
aquí con valores conocidos. Ningún modelo se carga ni se ajusta. Es el plan por etapas de
A, así que solo la segunda ventana tiene trabajos y el padre congelado es el suyo.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import sha256
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix, campaign_stage
from mars_titan.posttraining import stage_comparison as compare
from tests.evaluation.test_comparison_sources import masked_view, save
from tests.posttraining.campaign_fixture import write_configs

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/posttraining/historical-masked-adapter-comparison-a.json"
ASSETS = tuple("ABCDEFG")
OFFSETS = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
ADAPTED = ["gru__head", "gru__fusion", "gru__head_fusion", "gru__fusion_full_rank"]


def keys(segment):
    """Diez instantes de decisión del tramo, con los siete activos en cada uno."""
    first = datetime.fromisoformat(segment[0]).replace(tzinfo=UTC)
    moments = [first + timedelta(days=day, hours=21) for day in range(10)]
    return [(f"US/{asset}", moment) for moment in moments for asset in ASSETS]


def target(window, partition, rows):
    rng = np.random.default_rng(zlib.crc32(f"{window}/{partition}".encode()))
    return np.round(rng.normal(0, 1, rows), 2)


def prediction(arm, window, partition, truth):
    """Padre congelado, base y continuación iguales, cabeza perfecta y el resto desplazado."""
    if arm == "gru__head":
        return truth.copy()
    rng = np.random.default_rng(zlib.crc32(f"parent/{window}/{partition}".encode()))
    parent = np.round(0.5 * truth + rng.normal(0, 0.5, len(truth)), 3)
    return (
        parent if arm in ("gru", "gru__frozen_parent", "gru__full_continuation") else parent + 0.1
    )


def write_table(path, arm, window, partition, segment):
    rows = keys(segment)
    truth = target(window, partition, len(rows))
    point = prediction(arm, window, partition, truth)
    values = dict(
        sample_id=[f"{asset}/{moment.isoformat()}" for asset, moment in rows],
        asset_id=[asset for asset, _ in rows],
        market=["US"] * len(rows),
        prediction_at=pa.array([moment for _, moment in rows], pa.timestamp("us", tz="UTC")),
        target=truth,
        prediction=point,
    )
    values.update(zip(QUANTILE_COLUMNS, (point[:, None] + 0.2 * OFFSETS).T, strict=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(values), path)
    return dict(path=path, sha256=sha256(path))


class Stage:
    """Etapa reducida declarada, fuentes de la campaña y recibos escritos sin ejecutar."""

    def __init__(self, root):
        self.root = root
        _, stage_path = write_configs(root / "config", "A")
        # Sin secciones secundarias: los estratos leerían muestras que estas vistas no tienen.
        comparison = json.loads((root / "config" / "comparison.json").read_text())
        for section in ("modality_strata", "modality_ablation", "long_short"):
            comparison.pop(section)
        save(root / "config" / "comparison.json", dict(comparison, schema_version=1))
        self.declaration = root / "config" / "declaration.json"
        declared = json.loads(DECLARATION.read_text())
        save(self.declaration, dict(declared, stage=stage_path.name))
        self.loaded = compare.load_declaration(self.declaration)
        stage = self.loaded["stage"]
        scope = stage["campaign"]["comparison_config"]["resolved_scopes"]["US"]
        self.windows = scope["windows"]
        protocol = scope["protocols"]["US"]
        self.views = {}
        for window, fold in self.windows.items():
            manifest = dict(
                kind="corpus_supervision",
                cohort_complete=True,
                final_test_opened=False,
                temporal_view=masked_view(protocol, fold),
                **policy_identity(HISTORICAL_MASKED),
            )
            path = root / "views" / f"{window}.json"
            self.views[window] = dict(path=str(path), sha256=save(path, manifest))
        self.base_sources = root / "campaign" / "sources" / "US.json"
        self.output = root / "stage-output"
        self.publish_base()
        self.publish_stage()

    def entry(self, folder, arm, window, prefix):
        entry = dict(input_policy=HISTORICAL_MASKED, view_sha256=self.views[window]["sha256"])
        for partition in ("calibration", "evaluation"):
            path = prefix / f"{partition}-predictions.parquet"
            record = write_table(path, arm, window, partition, self.windows[window][partition])
            entry[partition] = dict(path=str(path.relative_to(folder)), sha256=record["sha256"])
        return entry

    def publish_base(self):
        folder = self.base_sources.parent
        arms = {
            "gru": {
                "42": {
                    window: self.entry(folder, "gru", window, folder / "gru" / window)
                    for window in self.windows
                }
            }
        }
        save(
            self.base_sources,
            dict(
                schema_version=1,
                kind="walk_forward_prediction_sources",
                scope="US",
                input_policy=HISTORICAL_MASKED,
                windows={w: dict(view=view) for w, view in self.views.items()},
                arms=arms,
            ),
        )

    def publish_stage(self, **changes):
        marker = dict(
            schema_version=1,
            kind=campaign_stage.RUN_KIND,
            stage_sha256=self.loaded["stage"]["sha256"],
            views={"US": {w: view["sha256"] for w, view in self.views.items()}},
            final_test_opened=False,
        )
        marker.update(changes)
        save(self.output / "stage.json", marker)
        identity = campaign_stage._digest(marker)
        for job in self.loaded["groups"]["gru"]["jobs"]["US"]:
            folder = self.output / "jobs" / job["id"]
            entry = self.entry(self.output, job["arm"], job["window"], folder)
            receipt = dict(
                schema_version=1,
                kind=campaign_stage.RECEIPT_KIND,
                status="completed",
                identity=dict(
                    id=job["id"],
                    stage_identity_sha256=identity,
                    view_sha256=entry["view_sha256"],
                ),
                predictions={part: entry[part] for part in ("calibration", "evaluation")},
                final_test_opened=False,
            )
            save(folder / "receipt.json", receipt)

    def sources(self):
        return compare.write_sources(
            self.declaration,
            "US",
            "gru",
            base_sources=self.base_sources,
            stage_output=self.output,
        )


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    stage = Stage(tmp_path_factory.mktemp("stage-comparison"))
    sources = stage.sources()
    config, report, sessions, portfolio = compare.evaluate(stage.declaration, sources, "US", "gru")
    return stage, sources, config, report, sessions, portfolio


def by_hand_session_mae(stage, arm, windows):
    sessions = []
    for window in windows:
        rows = keys(stage.windows[window]["evaluation"])
        truth = target(window, "evaluation", len(rows))
        error = np.abs(prediction(arm, window, "evaluation", truth) - truth)
        sessions.extend(error.reshape(-1, len(ASSETS)).mean(axis=1))
    return float(np.mean(sessions)), len(sessions)


def matrix_arms(stage, base_arm, family):
    """Brazos adaptados de una red según la matriz que declara la etapa, en su orden.

    Son los casos de una semilla sin papel de control, con el nombre que les da la etapa.
    """
    items = adapter_matrix.cases(
        stage["matrix"], stage["matrix_sha256"], family, head=QUANTILE_HEAD
    )
    return [
        campaign_stage.arm_name(base_arm, item["id"].split("/", 1)[1])
        for item in items
        if item["case"]["seed"] == 42 and item["control"] is None
    ]


def test_repository_declaration_derives_every_parent_from_the_stage_plan():
    loaded = compare.load_declaration(DECLARATION)
    # Ridge y XGBoost solo tienen el padre congelado en su cadena y no entran en los contrastes.
    assert list(loaded["groups"]) == [
        "rnn",
        "lstm",
        "gru",
        "dlinear",
        "transformer_compact",
        "titans_transformer_direct",
        "titans_mac_disabled",
        "titans_mac_frozen",
        "titans_mac_online",
    ]
    transformer = loaded["groups"]["transformer_compact"]
    assert transformer["full_continuation"] == "transformer_compact__full_continuation"
    # Los brazos adaptados salen de la matriz, con la variedad de #444 incluida. Solo
    # transformer_compact tiene los puntos de lectura.
    for base_arm, family in loaded["stage"]["campaign"]["neural"]["arms"].items():
        assert loaded["groups"][base_arm]["adapted"] == matrix_arms(
            loaded["stage"], base_arm, family
        )
    assert {"transformer_compact__readout", "transformer_compact__readout_dora"} <= set(
        transformer["adapted"]
    )
    assert not any("readout" in arm for arm in loaded["groups"]["gru"]["adapted"])
    for base_arm, config in loaded["configs"].items():
        group = loaded["groups"][base_arm]
        inherited = loaded["stage"]["campaign"]["comparison_config"]
        frozen = group["frozen_parent"]
        assert frozen == f"{base_arm}__frozen_parent"
        assert config["arms"][base_arm] == inherited["arms"][base_arm]
        assert set(config["arms"]) == {
            base_arm,
            frozen,
            group["full_continuation"],
            *group["adapted"],
        }
        assert all(arm["seeds"] == [42, 43, 44] for arm in config["arms"].values())
        families = config["resolved_families"]
        assert set(families["versus_frozen_parent"]) == {
            f"{arm}-{frozen}" for arm in [*group["adapted"], group["full_continuation"]]
        }
        # La base reentrenada en cada ventana queda como nivel, fuera de las familias.
        assert base_arm in families["levels"]
        # Sin postentrenamiento en la primera ventana, la comparación empieza en la segunda.
        for scope, resolved in config["resolved_scopes"].items():
            assert (
                list(resolved["windows"])
                == list(inherited["resolved_scopes"][scope]["windows"])[1:]
            )
        assert set(families["versus_full_continuation"]) == {
            f"{arm}-{group['full_continuation']}" for arm in group["adapted"]
        }
        # Todo lo demás se hereda de la comparación de la campaña.
        for field in ("metrics", "calibration", "input_policy", "long_short"):
            assert config[field] == inherited[field]
        assert {k: v for k, v in config["comparison"].items() if k != "families"} == {
            k: v for k, v in inherited["comparison"].items() if k != "families"
        }


def test_frozen_parent_continuation_and_adapters_are_contrasted_by_role(study):
    stage, _, config, report, _, _ = study
    assert set(report["arms"]) == {"gru", "gru__frozen_parent", "gru__full_continuation", *ADAPTED}
    assert report["posttraining"]["roles"] == dict(
        frozen_parent="gru__frozen_parent",
        full_continuation="gru__full_continuation",
        adapted=ADAPTED,
    )
    assert report["posttraining"]["windows"] == ["fold-001"]
    assert report["final_test_opened"] is False
    mae, _ = by_hand_session_mae(stage, "gru", ["fold-001"])
    contrasts = report["contrasts"]["US"]
    for family, base in (
        ("versus_frozen_parent", "gru__frozen_parent"),
        ("versus_full_continuation", "gru__full_continuation"),
    ):
        rows = {row["name"]: row for row in contrasts[family]["mae"]["contrasts"]}
        assert rows[f"gru__head-{base}"]["estimate"] == pytest.approx(-mae, abs=1e-12)
        assert contrasts[family]["mae"]["multiplicity"]["family_size"] == len(rows)
    rows = {row["name"]: row for row in contrasts["versus_frozen_parent"]["mae"]["contrasts"]}
    assert len(rows) == 5 and set(contrasts) >= {"versus_frozen_parent", "levels"}
    # La continuación repite las predicciones del padre: su contraste es exactamente cero.
    assert rows["gru__full_continuation-gru__frozen_parent"]["estimate"] == 0.0
    assert config["name"] == "historical-masked-2000-adapters-a-gru"


def test_sources_point_to_the_campaign_parent_and_the_stage_receipts(study):
    stage, sources, _, _, _, _ = study
    manifest = json.loads(sources.read_text())
    assert sources == stage.output / "sources" / "US" / "gru.json"
    assert set(manifest["arms"]) == {
        "gru",
        "gru__frozen_parent",
        "gru__full_continuation",
        *ADAPTED,
    }
    assert list(manifest["windows"]) == ["fold-001"]
    parent = manifest["arms"]["gru"]["42"]["fold-001"]["evaluation"]["path"]
    assert (sources.parent / parent).resolve() == (
        stage.base_sources.parent / "gru" / "fold-001" / "evaluation-predictions.parquet"
    ).resolve()
    frozen = manifest["arms"]["gru__frozen_parent"]["42"]["fold-001"]["evaluation"]["path"]
    assert (sources.parent / frozen).resolve() == (
        stage.output
        / "jobs"
        / "US/fold-001/gru__frozen_parent/frozen-s42"
        / "evaluation-predictions.parquet"
    ).resolve()
    head = manifest["arms"]["gru__head"]["42"]["fold-001"]["calibration"]["path"]
    assert (sources.parent / head).resolve() == (
        stage.output / "jobs" / "US/fold-001/gru__head/fit-s42" / "calibration-predictions.parquet"
    ).resolve()


def test_a_stage_with_other_identity_or_views_is_rejected(tmp_path):
    stage = Stage(tmp_path)
    stage.publish_stage(stage_sha256="0" * 64)
    with pytest.raises(ValueError, match="no pertenece a la etapa"):
        stage.sources()
    stage.publish_stage(views={"US": {"fold-000": "0" * 64}})
    with pytest.raises(ValueError, match="otras vistas"):
        stage.sources()


def test_a_receipt_from_another_run_or_view_is_rejected(tmp_path):
    stage = Stage(tmp_path)
    job = stage.loaded["groups"]["gru"]["jobs"]["US"][3]
    path = stage.output / "jobs" / job["id"] / "receipt.json"
    receipt = json.loads(path.read_text())
    for change in (
        dict(stage_identity_sha256="0" * 64),
        dict(view_sha256="0" * 64),
        dict(id="US/fold-000/other"),
    ):
        save(path, dict(receipt, identity=dict(receipt["identity"], **change)))
        with pytest.raises(ValueError, match=f"recibo de {job['id']}"):
            stage.sources()
    save(path, dict(receipt, status="running"))
    with pytest.raises(ValueError, match="no corresponde a la etapa"):
        stage.sources()
    assert not (stage.output / "sources" / "US" / "gru.json").exists()


def test_a_missing_prediction_leaves_no_candidate_manifest(tmp_path):
    stage = Stage(tmp_path)
    job = stage.loaded["groups"]["gru"]["jobs"]["US"][2]
    (stage.output / "jobs" / job["id"] / "calibration-predictions.parquet").unlink()
    with pytest.raises(ValueError, match="no es un archivo regular"):
        stage.sources()
    assert list((stage.output / "sources" / "US").iterdir()) == []


def test_a_control_without_a_declared_role_stops_the_derivation(monkeypatch):
    jobs = [
        dict(base_arm="gru", arm=f"gru__{name}", control=control, scope="US")
        for name, control in (
            ("full_continuation", "full_continuation"),
            ("head", None),
            ("linear_residual", "linear_residual"),
        )
    ]
    monkeypatch.setattr(compare, "plan_stage", lambda stage: jobs)
    # Sin brazos de la cadena trivial: todos los trabajos entran en los contrastes.
    monkeypatch.setattr(compare, "stage_arms", lambda stage: ({}, {}))
    with pytest.raises(ValueError, match="linear_residual de gru__linear_residual"):
        compare._groups({})
    monkeypatch.setattr(compare, "plan_stage", lambda stage: jobs[:2])
    group = compare._groups({})["gru"]
    assert group["full_continuation"] == "gru__full_continuation"
    assert group["adapted"] == ["gru__head"]


def test_adapters_must_evaluate_the_same_rows_as_the_parent(tmp_path):
    stage = Stage(tmp_path)
    job = stage.loaded["groups"]["gru"]["jobs"]["US"][1]
    folder = stage.output / "jobs" / job["id"]
    path = folder / "evaluation-predictions.parquet"
    table = pq.read_table(path)
    shifted = pa.array(table["target"].to_numpy() + 1.0)
    column = table.schema.get_field_index("target")
    pq.write_table(table.set_column(column, "target", shifted), path)
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["predictions"]["evaluation"]["sha256"] = sha256(path)
    save(folder / "receipt.json", receipt)
    sources = stage.sources()
    with pytest.raises(ValueError, match="70 filas con otro objetivo"):
        compare.evaluate(stage.declaration, sources, "US", "gru")


@pytest.mark.parametrize(
    "families",
    [
        {"wrong": {"base": "adapted", "variants": ["full_continuation"]}},
        {"wrong": {"base": "frozen_parent", "variants": ["frozen_parent"]}},
        {"wrong": {"base": "frozen_parent", "variants": ["linear_residual"]}},
        {"levels": {"base": "frozen_parent", "variants": ["adapted"]}},
        {},
    ],
)
def test_declared_families_only_use_known_roles(tmp_path, families):
    declared = json.loads(DECLARATION.read_text())
    path = tmp_path / "declaration.json"
    stage = (DECLARATION.parent / declared["stage"]).resolve()
    save(path, dict(declared, stage=str(stage), families=families))
    with pytest.raises(ValueError):
        compare.load_declaration(path)


def test_cli_checks_the_declaration_without_reading_predictions(capsys):
    assert compare.main(["check", "--declaration", str(DECLARATION)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["final_test_opened"] is False
    # La base, su padre congelado, la continuación completa y los brazos de la matriz.
    stage = campaign_stage.load_stage(
        DECLARATION.parent / json.loads(DECLARATION.read_text())["stage"]
    )
    assert result["parents"]["gru"]["arms"] == 3 + len(matrix_arms(stage, "gru", "gru"))
