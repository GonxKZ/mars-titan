"""Admisión de recibos congelados sin abrir modelos ni predicciones."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.evaluation.splits import build_folds


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class Campaign:
    def __init__(self, root):
        self.reference = root / "reference"
        self.completion = root / "completion"
        self.fold = "fold-000"
        self.counts = dict(train=4, validation=2, calibration=2, evaluation=2)
        protocol = dict(
            schema_version=1,
            market="US",
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
        self.manifest_path = self.reference / self.fold / "views/US.json"
        self.manifest = dict(
            kind="corpus_supervision",
            scope="full_corpus",
            cohort_complete=True,
            final_test_opened=False,
            cohort_id="fixture",
            counts=self.counts,
            selected_arm="US",
            source_manifest_sha256="b" * 64,
            temporal_view=dict(schema_version=1, protocol=protocol, fold=build_folds(protocol)[0]),
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
                    parent = self.originals["reference/search/winner"]
                    report["identity"]["parent"] = dict(
                        ordered_manifest_sha256="c" * 64,
                        source_sha256=self.manifest_hash,
                        parent_report_sha256=proofs["rnn"]["sha256"],
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
                    {"US-natural/rnn": "search/winner"}
                    if stage == "reference"
                    else {"ridge": "ridge"}
                )
            if stage == "reference":
                summary["kind"] = "reference_search"
            elif stage == "tabular":
                summary["counts"] = self.counts
            else:
                summary["identity"] = dict(proof=self.proof | {"parents": proofs})
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
                    arm="US",
                    counts=self.counts,
                    final_test_opened=False,
                )
            if stage == "tabular":
                self.proof["tabular_summary_sha256"] = signature
        evaluated = {}
        metrics = {
            name: dict(samples=2, session_count=1, session_mae=0.1, session_mse=0.01)
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
    assert old["parent_id"] == "reference/finalist/rnn-s43"
    assert new["parent_id"] == "reference/search/winner"
    assert new["bounds"] == ["2023-12-01", "2024-01-01"]
    assert isinstance(new["path"], Path)
    assert str(campaign.reference.parent) not in json.dumps(provenance)


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
    grid = dict(offsets=[-1.0, 0.0, 1.0], scale=0.2)
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


def test_does_not_follow_other_paths_inside_the_temporal_metadata(campaign):
    campaign.manifest["roots"] = dict(labels="/unavailable/private-test")
    campaign.manifest["temporal_view"]["macro_path"] = "/unavailable/macro.parquet"
    campaign.manifest_hash = save(campaign.manifest_path, campaign.manifest)
    for original in campaign.originals.values():
        if "identity" in original:
            original["identity"]["manifest_sha256"] = campaign.manifest_hash
        else:
            original["manifest_sha256"] = campaign.manifest_hash
    campaign.publish()
    rows, provenance = sources(campaign)
    assert len(rows) == 14
    assert "/unavailable/" not in json.dumps(provenance)


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
