"""Disco de la retención v2 sobre el plan de la campaña A v2, sin leer datos ni modelos.

Se ejecuta con `PYTHONPATH` en `src` de una copia de `feat/campaign-a-joint-design`, que
aporta el plan, el orden por ventanas y las etapas de A v2. `rolling_storage.py` y la huella
de `campaign_storage.py` se cargan desde este repositorio. La variante tabular toma la huella
de una copia de `perf/campaign-tabular` (páginas densas de XGBoost sin caché de validación en
disco), con la misma liberación de los boosters de recuperación tras el recibo.

    PYTHONPATH=<diseño conjunto>/src uv run --no-sync python \
        reports/engineering/rolling-retention-20261009/estimate_a_v2.py \
        --joint <diseño conjunto> --tabular <campaign-tabular> --counts <recuentos> \
        --output reports/engineering/rolling-retention-20261009/estimate-a-v2.json

`--counts` tiene el formato de `run_masked_campaign.py storage --counts`, con los recuentos
de `reports/data/campaign-a-v2-window-counts-20261009.json` del diseño conjunto y los flujos.
"""

import argparse
import importlib.util
import json
from pathlib import Path

ROLLING = Path(__file__).resolve().parents[3]
CAMPAIGN = "configs/baselines/historical-masked-campaign-a-v2.json"
STAGES = dict(
    ablation="configs/evaluation/historical-masked-ablation-stage-a-v2.json",
    adapters="configs/posttraining/historical-masked-adapter-stage-a-v2.json",
    rl="configs/simulation/historical-masked-rl-stage-a-v2.json",
)
GB = 1e9


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tabular_footprint(tabular):
    def footprint(job, counts, storage, *, release=None, prediction_bytes=None):
        value = tabular.job_footprint(
            job, counts, storage, release=release, prediction_bytes=prediction_bytes
        )
        release = storage["release_on_confirmation"] if release is None else release
        if job["model"] == "xgboost" and job.get("kind") != "carry" and release:
            extra = storage["state_bytes"]["xgboost"] * (storage["retained_states"]["xgboost"] - 1)
            value["retained"]["states"] -= extra
            value["transient"]["recovery"] += extra
            value["retained_bytes"] -= extra
            value["transient_bytes"] += extra
        return value

    return footprint


def summary(estimate, module):
    return {
        s: dict(
            retained_gb=round(estimate[s]["retained_bytes"] / GB, 2),
            peak_gb=round(estimate[s]["peak_bytes"] / GB, 2),
            peak_at=estimate[s]["peak_at"],
            increment_gb=round(module.window_increment(estimate, s)["bytes"] / GB, 2),
            increment_window=module.window_increment(estimate, s)["window"],
        )
        for s in module.SCENARIOS
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--joint", "--tabular", "--counts", "--output"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args(argv)
    # Plan, orden y etapas de A v2: módulos de la copia del diseño conjunto en PYTHONPATH.
    from mars_titan.training.campaign_schedule import window_schedule

    from mars_titan.training.campaign_plan import load_campaign, plan_campaign
    from mars_titan.training.storage_budget import (
        ablation_estimate,
        adapter_estimate,
        base_estimate,
        formula_windows,
    )

    JOINT, TABULAR = args.joint, args.tabular  # noqa: N806
    campaign = load_campaign(JOINT / CAMPAIGN)
    jobs = plan_campaign(campaign)
    windows = [dict(id=r["window"], scopes=r["scopes"]) for r in window_schedule(campaign, jobs)]
    counts = json.loads(args.counts.read_text())
    report = ROLLING / "reports/engineering/campaign-storage-20261009"
    coefficients = json.loads((report / "row-bytes.json").read_text())
    extras = json.loads((report / "extras.json").read_text())
    measured = formula_windows(counts, coefficients)
    rolling = load("rolling_storage", ROLLING / "src/mars_titan/training/rolling_storage.py")
    paths = {k: JOINT / v for k, v in STAGES.items()}
    stages = rolling.rolling_inputs(jobs, windows, extras, **paths)
    develop = load(
        "campaign_storage_rolling", ROLLING / "src/mars_titan/training/campaign_storage.py"
    )
    tabular = load(
        "campaign_storage_tabular", TABULAR / "src/mars_titan/training/campaign_storage.py"
    )
    variants = dict(
        develop_xgboost_cache=(
            develop.job_footprint,
            develop.load_storage(
                ROLLING / "configs/baselines/historical-masked-campaign-storage.json"
            ),
        ),
        tabular_dense_pages=(
            tabular_footprint(tabular),
            tabular.load_storage(
                TABULAR / "configs/baselines/historical-masked-campaign-storage.json"
            ),
        ),
    )
    combos = dict(
        base=dict(),
        base_rl=dict(policy_reads=stages["policy_reads"], tapes=stages["tapes"]),
        base_rl_ablation=dict(
            policy_reads=stages["policy_reads"],
            tapes=stages["tapes"],
            ablation_jobs=stages["ablation_jobs"],
        ),
        all_ordered_copy=dict(stages, ordered_copy=True),
        all_blocks=dict(stages, ordered_copy=False),
    )
    results, details = {}, {}
    for name, (footprint, storage) in variants.items():
        module = load(f"rolling_{name}", ROLLING / "src/mars_titan/training/rolling_storage.py")
        module.job_footprint = footprint
        for combo, options in combos.items():
            estimate = module.rolling_estimate(
                jobs, windows, counts, measured, storage, extras, **options
            )
            results[f"{name}/{combo}"] = summary(estimate, module)
            if combo in ("base_rl_ablation", "all_blocks"):
                details[f"{name}/{combo}"] = {
                    s: [
                        {k: (round(v / GB, 2) if isinstance(v, int) else v) for k, v in row.items()}
                        for row in estimate[s]["windows"]
                    ]
                    for s in module.SCENARIOS
                }
        parts = {}
        for job in jobs:
            f = footprint(
                job,
                counts[job["scope"]][job["window"]],
                storage,
                release=True,
                prediction_bytes=lambda *_: 0,
            )
            parts.setdefault(job["model"], dict(jobs=0, states=0, reports=0, sessions=0))
            for key in ("states", "reports", "sessions"):
                parts[job["model"]][key] += f["retained"][key]
            parts[job["model"]]["jobs"] += 1
        details[f"{name}/kept_per_model_gb"] = {
            m: dict(
                jobs=v["jobs"],
                **{k: round(v[k] / GB, 3) for k in ("states", "reports", "sessions")},
            )
            for m, v in parts.items()
        }
    # Referencia: la retención por etapas de la declaración actual, sin recorrido por ventanas.
    reports = {s: {w: dict(counts=c) for w, c in v.items()} for s, v in counts.items()}
    storage = variants["develop_xgboost_cache"][1]
    base = base_estimate(campaign, jobs, reports, measured, storage, extras)["totals"]
    ablation = ablation_estimate(paths["ablation"], reports, measured, storage)["layouts"]
    adapters = adapter_estimate(paths["adapters"], reports, extras)["layouts"]
    by_stage = dict(
        base={
            k: dict(
                retained_gb=round(v["retained_bytes"] / GB, 2),
                peak_gb=round(v["peak_bytes"] / GB, 2),
            )
            for k, v in base.items()
        },
        ablation={
            k: dict(
                retained_gb=round(v["retained_bytes"] / GB, 2),
                peak_gb=round(v["peak_bytes"] / GB, 2),
            )
            for k, v in ablation.items()
        },
        adapters={
            k: dict(
                retained_gb=round(v["retained_bytes"] / GB, 2),
                peak_gb=round(v["peak_bytes"] / GB, 2),
            )
            for k, v in adapters.items()
        },
    )
    aggregates = sum(
        int(
            extras["aggregate_bytes_per_row"]
            * sum(counts[j["scope"]][j["window"]][p] for p in rolling._partitions(j))
        )
        for j in jobs
    )
    result = dict(
        schema_version=1,
        kind="historical_masked_rolling_retention_estimate",
        recorded_at="2026-10-09",
        campaign=dict(
            path=CAMPAIGN,
            sha256=campaign["sha256"],
            branch="feat/campaign-a-joint-design",
            commit="4f04aebb",
        ),
        stages=STAGES,
        counts="reports/data/campaign-a-v2-window-counts-20261009.json del diseño conjunto",
        row_bytes="reports/engineering/campaign-storage-20261009/row-bytes.json",
        extras="reports/engineering/campaign-storage-20261009/extras.json",
        tabular=dict(branch="perf/campaign-tabular", commit="496cc512"),
        jobs=len(jobs),
        windows=[w["id"] for w in windows],
        ablation_jobs=len(stages["ablation_jobs"]),
        adapter_jobs=len(stages["adapter_jobs"]),
        policy_inputs=len(stages["policy_reads"]),
        tape_gb=round(sum(stages["tapes"].values()) / GB, 2),
        aggregates_gb=round(aggregates / GB, 2),
        by_stage_without_rolling=by_stage,
        results=results,
        details=details,
        training_executed=False,
        final_test_opened=False,
    )
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    for key, value in results.items():
        print(key, json.dumps(value))


if __name__ == "__main__":
    main()
