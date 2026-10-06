"""Padres finalistas emparejados sin cambiar la selección de la búsqueda."""

import copy
import json

import pytest

from mars_titan.data.storage import atomic_json, sha256
from tests.training.test_klpo_queue import parents


def matched_campaign(tmp_path):
    reference, tabular, view = parents(tmp_path)
    for path in (reference, tabular):
        summary = json.loads(path.read_text())
        originals = list(summary["runs"])
        finalists = []
        for row in originals:
            neural = path == reference
            kind = row["case"]["kind"] if neural else row["kind"]
            field = "case" if neural else "parameters"
            relative = row["path"] if neural else row["attempts"][-1]["path"]
            report = json.loads((path.parent / relative / "run.json").read_text())
            if kind != "ridge":
                row[field].update(seed=42)
                if neural:
                    row[field].update(epochs=5, learning_rate=0.001)
                report["identity"]["case" if neural else "options"] = copy.deepcopy(row[field])
            atomic_json(path.parent / relative / "run.json", report)
            row["report_sha256"] = sha256(path.parent / relative / "run.json")
            if neural:
                finalists.append(
                    dict(arm="US", weighting="natural", kind=kind, seed=42, run_id=row["id"])
                )
            if kind == "ridge":
                continue
            for seed in (43, 44):
                replica = copy.deepcopy(row)
                replica["id"] = f"{kind}-s{seed}"
                replica["stage"] = "finalist"
                replica[field]["seed"] = seed
                destination = f"runs/{kind}-s{seed}" + ("" if neural else "/attempt-0001")
                folder = path.parent / destination
                folder.mkdir(parents=True)
                if neural:
                    replica["path"] = destination
                    finalists.append(
                        dict(
                            arm="US",
                            weighting="natural",
                            kind=kind,
                            seed=seed,
                            run_id=replica["id"],
                        )
                    )
                else:
                    replica["attempts"] = [dict(path=destination, status="completed")]
                inherited = copy.deepcopy(report)
                inherited["identity"]["case" if neural else "options"] = copy.deepcopy(
                    replica[field]
                )
                for name in ("model.pt", "train.parquet", "validation.parquet"):
                    (folder / name).write_bytes((path.parent / relative / name).read_bytes())
                (folder / "model.pt").write_bytes(f"{kind}-{seed}".encode())
                inherited["checkpoint"]["sha256"] = sha256(folder / "model.pt")
                atomic_json(folder / "run.json", inherited)
                replica["report_sha256"] = sha256(folder / "run.json")
                summary["runs"].append(replica)
        summary.update(planned_runs=len(summary["runs"]), completed_runs=len(summary["runs"]))
        if path == reference:
            summary["finalists"] = finalists
        atomic_json(path, summary)
    return reference, tabular, view


def update_finalist(reference, change):
    summary = json.loads(reference.read_text())
    row = next(r for r in summary["runs"] if r["id"] == "gru-s43")
    report_path = reference.parent / row["path"] / "run.json"
    report = json.loads(report_path.read_text())
    change(row, report)
    atomic_json(report_path, report)
    row["report_sha256"] = sha256(report_path)
    atomic_json(reference, summary)


def test_matching_uses_each_finalist_and_identifies_shared_deterministic_ridge(tmp_path):
    from mars_titan.posttraining.parent_selection import matching_parents, parent_for_seed

    reference, tabular, _ = matched_campaign(tmp_path)
    proof = matching_parents(reference, tabular, seeds=[42, 43, 44])
    assert proof["schema_version"] == 2
    assert proof["parent_seed_policy"] == "matching"
    for family in ("rnn", "lstm", "gru", "dlinear", "xgboost"):
        records = [parent_for_seed(proof, family, seed) for seed in (42, 43, 44)]
        assert [record["parent_seed"] for record in records] == [42, 43, 44]
        assert len({record["report"] for record in records}) == 3
        assert records[1]["run_id"] == f"{family}-s43"
    ridge = [parent_for_seed(proof, "ridge", seed) for seed in (42, 43, 44)]
    assert ridge[0] == ridge[1] == ridge[2]
    assert ridge[0]["parent_seed"] is None
    assert ridge[0]["shared_deterministic"] is True


@pytest.mark.parametrize(
    "defect",
    ["seed", "configuration", "receipt", "selection", "checkpoint", "population", "incomplete"],
)
def test_matching_rejects_a_false_or_changed_finalist(tmp_path, defect):
    from mars_titan.posttraining.parent_selection import matching_parents

    reference, tabular, _ = matched_campaign(tmp_path)

    def change(row, report):
        if defect == "seed":
            row["case"]["seed"] = report["identity"]["case"]["seed"] = 42
        elif defect == "configuration":
            row["case"]["learning_rate"] = report["identity"]["case"]["learning_rate"] = 0.2
        elif defect == "receipt":
            report["identity"]["case"]["seed"] = 44
        elif defect == "selection":
            report["predictions"]["validation"]["metrics"]["session_mae"] = 0.9
        elif defect == "checkpoint":
            report["checkpoint"]["sha256"] = "f" * 64
        elif defect == "population":
            report["samples"]["train"] = 99
        else:
            report["status"] = "running"

    update_finalist(reference, change)
    with pytest.raises(ValueError):
        matching_parents(reference, tabular, seeds=[42, 43, 44])


def test_legacy_parent_binding_keeps_the_search_parent_for_all_adjustment_seeds(tmp_path):
    from mars_titan.posttraining.parent_selection import parent_for_seed
    from mars_titan.training.klpo_queue import selected_parents

    reference, tabular, _ = matched_campaign(tmp_path)
    proof = selected_parents(reference, tabular)
    assert "schema_version" not in proof
    assert parent_for_seed(proof, "gru", 43) == proof["parents"]["gru"]


@pytest.mark.parametrize(
    "change", [{"schema_version": 1}, {"parent_seed_policy": "shared"}, {"parents_by_seed": {}}]
)
def test_partial_or_changed_matching_proof_cannot_fall_back_to_legacy(tmp_path, change):
    from mars_titan.posttraining.parent_selection import matching_parents, parent_for_seed

    reference, tabular, _ = matched_campaign(tmp_path)
    proof = matching_parents(reference, tabular, seeds=[42, 43, 44])
    proof.update(change)
    with pytest.raises(ValueError):
        parent_for_seed(proof, "gru", 43)


@pytest.mark.parametrize("defect", ["score", "checkpoint"])
def test_matching_neural_parent_keeps_its_selected_epoch_and_checkpoint(tmp_path, defect):
    from mars_titan.posttraining.parent_selection import matching_parents

    reference, tabular, _ = matched_campaign(tmp_path)
    summary = json.loads(reference.read_text())
    for row in summary["runs"]:
        if row["case"]["kind"] != "gru":
            continue
        folder = reference.parent / row["path"]
        report = json.loads((folder / "run.json").read_text())
        row["case"]["selection"] = dict(metric="session_mae", patience=2, min_delta=0.0)
        report["identity"]["case"] = copy.deepcopy(row["case"])
        report["selection"] = dict(best_epoch=1, last_epoch=3, best_score=0.1)
        (folder / "checkpoints").mkdir()
        checkpoint = folder / "checkpoints/best.pt"
        checkpoint.write_bytes((folder / "model.pt").read_bytes())
        report["checkpoint"] = dict(path="checkpoints/best.pt", sha256=sha256(checkpoint))
        atomic_json(
            folder / "checkpoints/latest.json",
            dict(best=dict(name="best.pt", sha256=sha256(checkpoint))),
        )
        atomic_json(folder / "run.json", report)
        row["report_sha256"] = sha256(folder / "run.json")
    atomic_json(reference, summary)
    assert (
        matching_parents(reference, tabular, seeds=[42, 43, 44])["parents_by_seed"]["gru"]["43"][
            "parent_seed"
        ]
        == 43
    )
    if defect == "score":
        update_finalist(reference, lambda _row, report: report["selection"].update(best_score=0.9))
    else:
        folder = reference.parent / "runs/gru-s43"
        atomic_json(
            folder / "checkpoints/latest.json", dict(best=dict(name="other.pt", sha256="a" * 64))
        )
    with pytest.raises(ValueError, match="seleccionad|selección"):
        matching_parents(reference, tabular, seeds=[42, 43, 44])
