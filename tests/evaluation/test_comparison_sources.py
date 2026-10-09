"""Admisión de recibos congelados sin abrir modelos ni predicciones."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.evaluation.splits import build_folds

# Fuentes del contrato temporal estricto (versión 1). La admisión de recibos valida el
# contrato completo pero no abre estas rutas, por eso pueden apuntar a archivos ausentes.
STRICT_SOURCES = dict(
    macro_path="/unavailable/macro.parquet",
    macro_sha256="c" * 64,
    admission_path="/unavailable/admission.json",
    admission_sha256="d" * 64,
    parent_manifest="/unavailable/parent/manifest.json",
    parent_sha256="e" * 64,
)


def strict_view(protocol):
    """Vista estricta completa de la primera ventana, como la escribe la preparación."""
    return dict(
        schema_version=1, protocol=protocol, fold=build_folds(protocol)[0], **STRICT_SOURCES
    )


def masked_protocol(market, first_validation_start="2023-10-01", **changes):
    """Protocolo v2 de la edición desde 2000 con ventanas mensuales para los fixtures."""
    protocol = dict(
        schema_version=2,
        market=market,
        train_start="2000-01-01",
        first_validation_start=first_validation_start,
        validation_months=1,
        calibration_months=1,
        evaluation_months=1,
        step_months=1,
        minimum_train_months=240,
        purge="label_interval",
        final_test_start="2024-01-01",
        final_test_end="2025-01-01",
        primary_metric="session_mae",
        selection=dict(
            metric="session_mae", stopping="fixed_budget", patience=5, min_delta=1e-5, max_epochs=30
        ),
        seeds=[42, 43, 44],
    )
    protocol.update(changes)
    return protocol


def masked_view(protocol, fold=None):
    """Vista histórica con máscaras (versión 2), como la escribe la preparación."""
    return dict(
        schema_version=2,
        **policy_identity(HISTORICAL_MASKED),
        protocol=protocol,
        fold=build_folds(protocol)[0] if fold is None else fold,
        selection_partition="validation",
        recover_annual_boundaries=True,
        parent_manifest="/unavailable/parent/manifest.json",
        parent_sha256="e" * 64,
    )


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class Campaign:
    def __init__(
        self, root, *, materialize=False, matching=False, markets=("US",), policy=STRICT_INPUTS
    ):
        self.materialize = materialize
        self.matching = matching
        self.markets = markets
        self.arm = "US+CN" if len(markets) == 2 else markets[0]
        self.reference = root / "reference"
        self.completion = root / "completion"
        self.fold = "fold-000"
        self.counts = {
            name: count * len(markets)
            for name, count in dict(train=4, validation=2, calibration=2, evaluation=2).items()
        }
        protocol = dict(
            schema_version=1,
            market=markets[0],
            train_start="2022-01-01",
            first_validation_start="2023-10-01",
            validation_months=1,
            calibration_months=1,
            evaluation_months=1,
            step_months=1,
            minimum_train_months=6,
            gap_sessions=1,
            final_test_start="2024-01-01",
            final_test_end="2025-01-01",
            primary_metric="session_mae",
            seeds=[42, 43, 44],
        )
        masked = policy == HISTORICAL_MASKED
        if masked:
            protocol = masked_protocol(markets[0])
        self.manifest_path = self.reference / self.fold / "views" / f"{self.arm}.json"
        self.manifest = dict(
            kind="corpus_supervision",
            scope="full_corpus",
            cohort_complete=True,
            final_test_opened=False,
            cohort_id="fixture",
            counts=self.counts,
            selected_arm=self.arm,
            source_manifest_sha256="b" * 64,
            temporal_view=masked_view(protocol) if masked else strict_view(protocol),
            **policy_identity(policy),
        )
        if len(markets) == 2:
            contract = self.manifest.pop("temporal_view")
            self.manifest.update(
                markets=list(markets),
                assets=[
                    dict(
                        market=market,
                        symbol=symbol,
                        counts=dict(train=2, validation=1, calibration=1, evaluation=1),
                    )
                    for market in markets
                    for symbol in ("A", "B")
                ],
                temporal_views={
                    market: dict(contract, protocol=dict(protocol, market=market))
                    for market in markets
                },
            )
        self.manifest_hash = save(self.manifest_path, self.manifest)
        self.rows = {stage: [] for stage in ("reference", "tabular", "posttraining")}
        self.originals = {}
        self.add("reference", "search/winner", "rnn", "search", 42)
        self.add("reference", "search/loser", "rnn", "search", 42, loss="mse")
        self.add("reference", "finalist/rnn-s43", "rnn", "finalist", 43)
        self.add(
            "reference",
            "posttraining/rnn-mae-s43",
            "rnn",
            "posttraining",
            43,
            parent="finalist/rnn-s43",
        )
        self.add("tabular", "ridge", "ridge", "search", None)
        self.add("posttraining", "rnn/seed-42/real/neural_mae", "rnn", "posttraining", 42)
        self.add("posttraining", "rnn/seed-42/real/mae", "rnn", "posttraining", 42)
        if matching:
            self.add("posttraining", "rnn/seed-43/real/neural_mae", "rnn", "posttraining", 43)
        self.result_changes = {}
        self.job_changes = {}
        self.publish()

    def folder(self, stage):
        return (
            self.reference / self.fold
            if stage == "reference"
            else self.completion / self.fold / stage
        )

    def add(self, stage, name, family, phase, seed, *, loss="mae", parent=None):
        case = dict(kind=family, seed=seed, loss=loss, epochs=5 if parent else 100)
        if stage == "posttraining":
            case = dict(mode=name.rsplit("/", 1)[1], condition="real", seed=seed, epochs=50)
        elif stage == "tabular":
            case = {"alpha": 1.0}
        checkpoint = dict(
            path="missing-weights.pt", sha256=hashlib.sha256(name.encode()).hexdigest()
        )
        identity = dict(manifest_sha256=self.manifest_hash, case=case, initialization=None)
        if stage == "posttraining":
            identity["grid"] = dict(
                schema_version=1,
                values=[index / 10 for index in range(-10, 11)],
                scale=0.1,
                source_sha256=self.manifest_hash,
                training_samples=self.counts["train"],
            )
        report = dict(
            status="completed",
            final_test_opened=False,
            identity=identity,
            scope="full_corpus",
            cohort_complete=True,
            samples=self.counts,
            checkpoint=checkpoint,
            selection=dict(best_epoch=0, last_epoch=5),
            stopped_early=False,
            global_step=20,
            attempts=[dict(seconds=1.5)],
            model=family,
        )
        if stage == "tabular":
            report.update(alpha=1.0, manifest_sha256=self.manifest_hash)
            report.pop("identity")
            report.pop("selection")
        if parent:
            identity["initialization"] = dict(
                parent_checkpoint_sha256=self.originals[f"reference/{parent}"]["checkpoint"][
                    "sha256"
                ]
            )
        row = dict(id=name, kind=family, stage=phase, status="completed", case=case)
        if parent:
            row["parent"] = parent
        self.rows[stage].append(row)
        self.originals[f"{stage}/{name}"] = report

    def publish(self):
        proofs = {}
        jobs = []
        summaries = {}
        for stage in self.rows:
            entries = [] if stage != "posttraining" else {}
            for row in self.rows[stage]:
                key = f"{stage}/{row['id']}"
                report = self.originals[key]
                if stage == "posttraining":
                    parent_id = (
                        "finalist/rnn-s43"
                        if self.matching and row["case"]["seed"] == 43
                        else "search/winner"
                    )
                    parent = self.originals[f"reference/{parent_id}"]
                    parent_report = self.folder("reference") / "runs" / parent_id / "run.json"
                    report["identity"]["parent"] = dict(
                        ordered_manifest_sha256="c" * 64,
                        source_sha256=self.manifest_hash,
                        parent_report_sha256=hashlib.sha256(parent_report.read_bytes()).hexdigest(),
                        checkpoint_sha256=parent["checkpoint"]["sha256"],
                        model="rnn",
                        counts=self.counts,
                        cohort_id="fixture",
                        final_test_opened=False,
                    )
                    report["identity"]["dataset"] = dict(
                        train_sha256="c" * 64,
                        validation_sha256="c" * 64,
                        parent_sha256=parent["checkpoint"]["sha256"],
                        synthetic=None,
                    )
                    report.update(
                        kind="paired_posttraining", domain="real", mode=row["case"]["mode"]
                    )
                path = self.folder(stage) / "runs" / row["id"] / "run.json"
                signature = save(path, report)
                comparator = None
                if "parent" in row:
                    parent_path = self.folder(stage) / "runs" / row["parent"] / "run.json"
                    comparator = dict(
                        path=str(parent_path),
                        sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                    )
                job = dict(
                    id=key,
                    stage=stage,
                    report=str(path),
                    sha256=signature,
                    comparator=comparator,
                    phase=row["stage"],
                )
                job.update(self.job_changes.get(key, {}))
                jobs.append(job)
                if stage == "posttraining":
                    entries[row["id"]] = dict(
                        status="completed",
                        path=str(path.relative_to(self.folder(stage))),
                        sha256=signature,
                    )
                else:
                    entry = copy.deepcopy(row)
                    entry["report_sha256"] = signature
                    relative = str(path.parent.relative_to(self.folder(stage)))
                    entry.update(
                        dict(path=relative)
                        if stage == "reference"
                        else dict(attempts=[dict(path=relative, status="completed")])
                    )
                    entries.append(entry)
                if key in ("reference/search/winner", "tabular/ridge"):
                    proofs[row["kind"]] = dict(report=str(path), sha256=signature, run_id=row["id"])
            summary = dict(
                status="completed",
                final_test_opened=False,
                runs=entries,
                planned_runs=len(entries),
                completed_runs=len(entries),
                scope="full_corpus",
                cohort_complete=True,
                identity=dict(
                    manifest_sha256="b" * 64 if stage == "reference" else self.manifest_hash
                ),
            )
            if stage != "posttraining":
                summary["selected"] = (
                    {f"{self.arm}-natural/rnn": "search/winner"}
                    if stage == "reference"
                    else {"ridge": "ridge"}
                )
            if stage == "reference":
                summary["kind"] = "reference_search"
            elif stage == "tabular":
                summary["counts"] = self.counts
            else:
                summary["identity"] = dict(proof=self.proof | {"parents": proofs})
                if self.matching:
                    paired = {}
                    for family, chosen in proofs.items():
                        paired[family] = {}
                        for seed in (42, 43):
                            record = dict(chosen)
                            if family == "rnn" and seed == 43:
                                parent_path = (
                                    self.folder("reference") / "runs/finalist/rnn-s43/run.json"
                                )
                                record.update(
                                    report=str(parent_path),
                                    sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                                    run_id="finalist/rnn-s43",
                                )
                            record.update(
                                parent_seed=None if family == "ridge" else seed,
                                shared_deterministic=family == "ridge",
                            )
                            paired[family][str(seed)] = record
                    summary["identity"]["proof"].update(
                        schema_version=2, parent_seed_policy="matching", parents_by_seed=paired
                    )
            path = self.folder(stage) / "summary.json"
            signature = save(path, summary)
            summaries[stage] = dict(path=str(path), sha256=signature)
            if stage == "reference":
                self.proof = dict(
                    manifest=str(self.manifest_path),
                    manifest_sha256=self.manifest_hash,
                    reference_summary_sha256=signature,
                    confirmed_runs=len(entries),
                    matched_runs=len(entries),
                    arm=self.arm,
                    counts=self.counts,
                    final_test_opened=False,
                )
            if stage == "tabular":
                self.proof["tabular_summary_sha256"] = signature
        evaluated = {}
        metrics = {
            name: dict(
                samples=2 * len(self.markets),
                session_count=len(self.markets),
                session_mae=0.1,
                session_mse=0.01,
            )
            for name in ("prediction", "parent", "zero")
        }
        for job in jobs:
            original = self.originals[job["id"]]
            report = dict(
                status="completed",
                final_test_opened=False,
                job=job,
                family=original["model"],
                case=original.get("identity", {}).get("case", {"alpha": original.get("alpha")}),
                checkpoint=original["checkpoint"],
                selection=original.get("selection"),
                primary="median" if job["stage"] == "posttraining" else "continuous",
                predictions={
                    part: dict(
                        path=f"{part}.parquet", sha256="d" * 64, metrics=copy.deepcopy(metrics)
                    )
                    for part in ("calibration", "evaluation")
                },
            )
            report.update(self.result_changes.get(job["id"], {}))
            path = self.completion / self.fold / "evaluation/runs" / job["id"] / "run.json"
            if self.materialize:
                from datetime import UTC, datetime

                import pyarrow as pa
                import pyarrow.parquet as pq

                path.parent.mkdir(parents=True, exist_ok=True)
                for month, part in ((11, "calibration"), (12, "evaluation")):
                    table = pa.table(
                        dict(
                            sample_id=[
                                f"{part}/{market}/{name}"
                                for market in self.markets
                                for name in ("a", "b")
                            ],
                            asset_id=[
                                f"{market}/{name}" for market in self.markets for name in ("A", "B")
                            ],
                            market=[market for market in self.markets for _ in range(2)],
                            prediction_at=pa.array(
                                [datetime(2023, month, 2, tzinfo=UTC)] * (2 * len(self.markets)),
                                pa.timestamp("us", tz="UTC"),
                            ),
                            target=[0.1] * (2 * len(self.markets)),
                            prediction=[0.0] * (2 * len(self.markets)),
                            parent=[0.0] * (2 * len(self.markets)),
                            zero=[0.0] * (2 * len(self.markets)),
                        )
                    )
                    parquet = path.parent / f"{part}.parquet"
                    pq.write_table(table, parquet)
                    report["predictions"][part]["sha256"] = hashlib.sha256(
                        parquet.read_bytes()
                    ).hexdigest()
            signature = save(path, report)
            evaluated[job["id"]] = dict(
                path=str(path.relative_to(self.completion / self.fold / "evaluation")),
                sha256=signature,
            )
        evaluation = dict(
            kind="frozen_temporal_evaluation",
            status="completed",
            final_test_opened=False,
            identity=dict(
                sources=summaries,
                jobs=jobs,
                ordered_sha256="c" * 64,
                manifest_sha256=self.manifest_hash,
                partitions=["calibration", "evaluation"],
            ),
            runs=evaluated,
            planned_runs=len(evaluated),
            completed_runs=len(evaluated),
        )
        evaluation_hash = save(self.completion / self.fold / "evaluation/summary.json", evaluation)
        count = len(self.rows["reference"])
        reference = dict(
            kind="temporal_reference_search",
            status="completed",
            final_test_opened=False,
            planned_runs=count,
            completed_runs=count,
            folds=[
                dict(id=self.fold, status="completed", planned_runs=count, completed_runs=count)
            ],
            identity=dict(manifests={self.fold: "b" * 64}),
        )
        reference_hash = save(self.reference / "summary.json", reference)
        stages = [
            dict(
                fold=self.fold,
                stage=stage,
                status="completed",
                planned_runs=len(self.rows[stage]),
                completed_runs=len(self.rows[stage]),
                sha256=summaries[stage]["sha256"],
            )
            for stage in ("tabular", "posttraining")
        ]
        stages.append(
            dict(
                fold=self.fold,
                stage="evaluation",
                status="completed",
                planned_runs=len(jobs),
                completed_runs=len(jobs),
                sha256=evaluation_hash,
            )
        )
        count = sum(row["planned_runs"] for row in stages)
        save(
            self.completion / "summary.json",
            dict(
                kind="temporal_posttraining_completion",
                status="completed",
                final_test_opened=False,
                identity=dict(
                    reference_sha256=reference_hash,
                    references={
                        self.fold: {
                            k: v for k, v in self.proof.items() if k != "tabular_summary_sha256"
                        }
                    },
                ),
                stages=stages,
                planned_runs=count,
                completed_runs=count,
            ),
        )


@pytest.fixture
def campaign(tmp_path):
    return Campaign(tmp_path)


def sources(campaign):
    from mars_titan.evaluation.comparison_sources import predictive_sources

    return predictive_sources(campaign.reference, campaign.completion)


def test_includes_search_winner_and_preserves_distinct_continuations(campaign):
    rows, provenance = sources(campaign)
    assert len(rows) == 14
    by_id = {(row["id"], row["partition"]): row for row in rows}
    assert by_id["reference/search/winner", "evaluation"]["included"]
    assert not by_id["reference/search/loser", "evaluation"]["included"]
    assert by_id["reference/finalist/rnn-s43", "evaluation"]["included"]
    old = by_id["reference/posttraining/rnn-mae-s43", "evaluation"]
    new = by_id["posttraining/rnn/seed-42/real/neural_mae", "evaluation"]
    assert old["method"] == "reference_mae"
    assert new["method"] == "neural_mae"
    assert old["metadata"]["case"]["loss"] == "mae"
    assert old["checkpoint_sha256"] == campaign.originals[old["id"]]["checkpoint"]["sha256"]
    assert (
        old["source_report_sha256"]
        == hashlib.sha256(
            (campaign.folder("reference") / "runs/posttraining/rnn-mae-s43/run.json").read_bytes()
        ).hexdigest()
    )
    assert old["parent_id"] == "reference/finalist/rnn-s43"
    assert new["parent_id"] == "reference/search/winner"
    assert new["bounds"] == ["2023-12-01", "2024-01-01"]
    assert isinstance(new["path"], Path)
    assert str(campaign.reference.parent) not in json.dumps(provenance)


def test_matching_campaign_keeps_finalist_parents_and_temporal_population(tmp_path):
    campaign = Campaign(tmp_path, matching=True)
    rows, provenance = sources(campaign)
    child = next(row for row in rows if row["id"] == "posttraining/rnn/seed-43/real/neural_mae")
    assert child["parent_id"] == "reference/finalist/rnn-s43"
    assert child["seed"] == 43
    assert child["bounds"] == ["2023-11-01", "2023-12-01"]
    assert provenance["final_test_opened"] is False
    campaign.originals["reference/finalist/rnn-s43"]["identity"]["case"]["loss"] = "mse"
    campaign.publish()
    with pytest.raises(ValueError, match="configuración"):
        sources(campaign)


def test_admission_does_not_require_opening_weights_or_prediction_files(campaign):
    weights = campaign.reference / campaign.fold / "runs/search/winner/missing-weights.pt"
    assert not weights.exists()
    rows, _ = sources(campaign)
    assert all(not row["path"].exists() for row in rows)


def test_normalizes_the_declared_xgboost_backend_to_its_family(campaign):
    campaign.rows["tabular"][0]["kind"] = "xgboost"
    campaign.originals["tabular/ridge"]["model"] = "xgboost_external_cuda"
    campaign.publish()
    rows, _ = sources(campaign)
    tabular = [row for row in rows if row["stage"] == "tabular"]
    assert {row["family"] for row in tabular} == {"xgboost"}
    assert {row["method"] for row in tabular} == {"xgboost"}


def test_changed_original_json_is_rejected(campaign):
    path = campaign.reference / campaign.fold / "runs/search/winner/run.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="huella"):
        sources(campaign)


@pytest.mark.parametrize("defect", ["test", "duplicate", "parent", "job", "outside", "predictions"])
def test_rejects_invalid_receipts_before_returning_sources(campaign, defect):
    key = "reference/posttraining/rnn-mae-s43"
    if defect == "test":
        campaign.originals[key]["final_test_opened"] = True
    elif defect == "duplicate":
        campaign.rows["reference"].append(copy.deepcopy(campaign.rows["reference"][0]))
    elif defect == "parent":
        campaign.originals[key]["identity"]["initialization"]["parent_checkpoint_sha256"] = "e" * 64
    elif defect == "job":
        campaign.result_changes[key] = dict(job=dict(id=key, phase="search"))
    elif defect == "outside":
        campaign.job_changes[key] = dict(report=str(campaign.reference.parent / "outside.json"))
    else:
        campaign.result_changes[key] = dict(predictions={"test": {"path": "test.parquet"}})
    campaign.publish()
    with pytest.raises(ValueError):
        sources(campaign)


def test_external_prediction_path_is_rejected_without_reading_it(campaign):
    evaluation = campaign.completion / campaign.fold / "evaluation/summary.json"
    key = "reference/search/winner"
    report = json.loads(
        (evaluation.parent / json.loads(evaluation.read_text())["runs"][key]["path"]).read_text()
    )
    report["predictions"]["evaluation"]["path"] = "../../../../../../outside.parquet"
    campaign.result_changes[key] = dict(predictions=report["predictions"])
    campaign.publish()
    with pytest.raises(ValueError, match="predicci|ruta|partición"):
        sources(campaign)


def test_preserves_training_grid_baseline_and_resource_evidence(campaign):
    original = campaign.originals["posttraining/rnn/seed-42/real/neural_mae"]
    grid = original["identity"]["grid"] | {"scale": 0.2}
    original["identity"]["grid"] = grid
    original["baseline"] = dict(primary="median", session_mae=0.2)
    original["budget"] = dict(updates=20)
    original["stop_reason"] = "validation_plateau"
    campaign.publish()
    rows, _ = sources(campaign)
    metadata = next(row["metadata"] for row in rows if row["method"] == "neural_mae")
    assert metadata["grid"] == grid
    assert metadata["baseline"] == dict(primary="median", session_mae=0.2)
    assert metadata["budget"] == dict(updates=20)
    assert metadata["stop_reason"] == "validation_plateau"
    assert metadata["attempts"] == [dict(seconds=1.5)]


@pytest.mark.parametrize("defect", ["missing", "invalid", "source", "population"])
def test_adjustment_grid_must_belong_to_the_declared_training_cohort(campaign, defect):
    identity = campaign.originals["posttraining/rnn/seed-42/real/neural_mae"]["identity"]
    if defect == "missing":
        identity.pop("grid")
    elif defect == "invalid":
        identity["grid"]["scale"] = 0.0
    elif defect == "source":
        identity["grid"]["source_sha256"] = "f" * 64
    else:
        identity["grid"]["training_samples"] = 999
    campaign.publish()
    with pytest.raises(ValueError, match="rejilla"):
        sources(campaign)


def test_rejects_duplicate_json_keys(campaign):
    path = campaign.reference / "summary.json"
    path.write_text('{"status":"completed","status":"running"}')
    with pytest.raises(ValueError, match="duplicada"):
        sources(campaign)


def test_prediction_symlink_is_rejected_without_opening_its_target(campaign):
    path = (
        campaign.completion
        / campaign.fold
        / "evaluation/runs/reference/search/winner/evaluation.parquet"
    )
    path.symlink_to(campaign.reference.parent / "private-test.parquet")
    with pytest.raises(ValueError, match="enlace"):
        sources(campaign)


def test_only_the_explicit_verified_temporal_manifest_may_be_external(campaign):
    external = campaign.reference.parent / "metadata" / "manifest.json"
    campaign.manifest_path = external
    campaign.manifest_hash = save(external, campaign.manifest)
    campaign.publish()
    rows, provenance = sources(campaign)
    assert len(rows) == 14
    assert str(external) not in json.dumps(provenance)
    external.write_text(external.read_text() + " ")
    with pytest.raises(ValueError, match="huella"):
        sources(campaign)


def republish_manifest(campaign):
    """Volver a firmar el manifiesto y enlazar todos los recibos a la nueva huella."""
    campaign.manifest_hash = save(campaign.manifest_path, campaign.manifest)
    for original in campaign.originals.values():
        if "identity" in original:
            original["identity"]["manifest_sha256"] = campaign.manifest_hash
            if "grid" in original["identity"]:
                original["identity"]["grid"]["source_sha256"] = campaign.manifest_hash
        else:
            original["manifest_sha256"] = campaign.manifest_hash
    campaign.publish()


def test_does_not_follow_other_paths_inside_the_temporal_metadata(campaign):
    campaign.manifest["roots"] = dict(labels="/unavailable/private-test")
    campaign.manifest["temporal_view"]["macro_path"] = "/unavailable/private-macro.parquet"
    republish_manifest(campaign)
    rows, provenance = sources(campaign)
    assert len(rows) == 14
    assert "/unavailable/" not in json.dumps(provenance)


@pytest.mark.parametrize("change", [*STRICT_SOURCES, "schema_version", "extra"])
def test_incomplete_or_extended_strict_view_is_rejected_with_consistent_receipts(campaign, change):
    view = campaign.manifest["temporal_view"]
    if change == "extra":
        view["input_policy"] = "strict_inputs_v1"
    elif change == "schema_version":
        view["schema_version"] = 2
    else:
        del view[change]
    republish_manifest(campaign)
    with pytest.raises(ValueError, match="contrato"):
        sources(campaign)


def test_temporal_metadata_cannot_move_the_final_test_boundary(campaign):
    campaign.manifest["temporal_view"]["protocol"].update(
        final_test_start="2025-01-01", final_test_end="2026-01-01"
    )
    campaign.manifest_hash = save(campaign.manifest_path, campaign.manifest)
    campaign.publish()
    with pytest.raises(ValueError, match="temporal|reservado"):
        sources(campaign)


def test_temporal_proof_cannot_authorize_opening_a_non_json_artifact(campaign):
    campaign.manifest_path = campaign.reference.parent / "never-open.parquet"
    campaign.publish()
    with pytest.raises(ValueError, match="metadatos JSON"):
        sources(campaign)


def test_frozen_prediction_population_must_match_the_temporal_manifest(campaign):
    evaluation = campaign.completion / campaign.fold / "evaluation/summary.json"
    key = "reference/search/winner"
    saved = json.loads(evaluation.read_text())["runs"][key]
    result = json.loads((evaluation.parent / saved["path"]).read_text())
    result["predictions"]["evaluation"]["metrics"]["prediction"]["samples"] = 3
    campaign.result_changes[key] = dict(predictions=result["predictions"])
    campaign.publish()
    with pytest.raises(ValueError, match="población"):
        sources(campaign)


def test_evaluation_must_declare_its_frozen_temporal_contract(campaign):
    path = campaign.completion / campaign.fold / "evaluation/summary.json"
    report = json.loads(path.read_text())
    report["kind"] = "unverified"
    signature = save(path, report)
    root = campaign.completion / "summary.json"
    summary = json.loads(root.read_text())
    for stage in summary["stages"]:
        if stage["stage"] == "evaluation":
            stage["sha256"] = signature
    save(root, summary)
    with pytest.raises(ValueError, match="congelado"):
        sources(campaign)


@pytest.mark.parametrize("markets", [("US",), ("US", "CN")])
def test_masked_views_are_admitted_only_under_their_declared_policy(tmp_path, markets):
    from mars_titan.evaluation.comparison_sources import predictive_sources

    masked = Campaign(tmp_path / "masked", markets=markets, policy=HISTORICAL_MASKED)
    strict = Campaign(tmp_path / "strict", markets=markets)
    rows, provenance = predictive_sources(
        masked.reference, masked.completion, input_policy=HISTORICAL_MASKED
    )
    strict_rows, strict_provenance = predictive_sources(strict.reference, strict.completion)
    assert provenance["input_policy"] == HISTORICAL_MASKED
    assert provenance["mask_contract"] == policy_identity(HISTORICAL_MASKED)["mask_contract"]
    assert "input_policy" not in strict_provenance and "mask_contract" not in strict_provenance
    # Mismas ventanas mensuales: solo cambian la política y las huellas de los recibos.
    assert [(r["id"], r["partition"], r["bounds"]) for r in rows] == [
        (r["id"], r["partition"], r["bounds"]) for r in strict_rows
    ]
    with pytest.raises(ValueError, match="política de entradas"):
        predictive_sources(masked.reference, masked.completion)
    with pytest.raises(ValueError, match="política de entradas"):
        predictive_sources(strict.reference, strict.completion, input_policy=HISTORICAL_MASKED)


def test_unknown_input_policy_is_rejected_before_opening_any_receipt(tmp_path):
    from mars_titan.evaluation.comparison_sources import predictive_sources

    with pytest.raises(ValueError, match="no está admitida"):
        predictive_sources(tmp_path / "a", tmp_path / "b", input_policy="historical_masked")


@pytest.mark.parametrize(
    "defect,message",
    [
        ("strict_view", "no cumple su contrato"),
        ("strict_protocol", "no conserva el inicio"),
        ("missing_mask_contract", "adhesión y contrato exactos"),
    ],
)
def test_masked_admission_rejects_views_that_mix_both_contracts(tmp_path, defect, message):
    from mars_titan.evaluation.comparison_sources import predictive_sources

    campaign = Campaign(tmp_path, policy=HISTORICAL_MASKED)
    if defect == "strict_view":
        campaign.manifest["temporal_view"] = strict_view(
            campaign.manifest["temporal_view"]["protocol"]
        )
    elif defect == "strict_protocol":
        view = campaign.manifest["temporal_view"]
        view["protocol"] = dict(view["protocol"], train_start="2022-01-01", minimum_train_months=6)
        view["fold"] = build_folds(view["protocol"])[0]
    else:
        campaign.manifest.pop("mask_contract")
    campaign.manifest_hash = save(campaign.manifest_path, campaign.manifest)
    for original in campaign.originals.values():
        if "identity" in original:
            original["identity"]["manifest_sha256"] = campaign.manifest_hash
            if "grid" in original["identity"]:
                original["identity"]["grid"]["source_sha256"] = campaign.manifest_hash
        else:
            original["manifest_sha256"] = campaign.manifest_hash
    campaign.publish()
    with pytest.raises(ValueError, match=message):
        predictive_sources(campaign.reference, campaign.completion, input_policy=HISTORICAL_MASKED)
