"""Selección con la validación anterior a cada ventana, sobre recibos escritos en la prueba.

Los recibos e informes tienen solo los campos que lee `integrity.window_selection`. No hay
modelos ajustados ni pasos de optimizador.
"""

import json

from mars_titan.integrity import window_selection as selection


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def _fit(output, job_id, score, case):
    """Recibo y su informe de un ajuste, con la puntuación de su validación."""
    report = dict(predictions=dict(validation=dict(metrics=dict(session_mae=score))))
    _write(output / "jobs" / job_id / "attempt-0001" / "run.json", report)
    receipt = dict(
        status="completed",
        score=score,
        identity=dict(case=case, seed=case["seed"]),
        report=dict(path=f"jobs/{job_id}/attempt-0001/run.json"),
    )
    _write(output / "jobs" / job_id / "receipt.json", receipt)


def test_a_finalist_must_repeat_the_case_chosen_by_validation(tmp_path):
    common = dict(scope="US", window="fold-000", arm="gru", kind="fit", anchor="fold-000")
    jobs = [
        dict(common, id="US/fold-000/gru/search-a", stage="search", seed=42),
        dict(common, id="US/fold-000/gru/search-b", stage="search", seed=42),
        dict(common, id="US/fold-000/gru/finalist-s43", stage="finalist", seed=43),
    ]
    _fit(tmp_path, jobs[0]["id"], 0.03, dict(rate=0.1, seed=42))
    _fit(tmp_path, jobs[1]["id"], 0.02, dict(rate=0.01, seed=42))
    _fit(tmp_path, jobs[2]["id"], 0.025, dict(rate=0.01, seed=43))
    campaign = dict(comparison_config=dict(arms={}))
    result = selection.check_selection(campaign, jobs, tmp_path)
    assert result["passed"], result["failures"]
    assert result["chosen"] == {"US/fold-000/gru": dict(job=jobs[1]["id"], searches=2)}
    # El finalista repite la tasa de la búsqueda que no ganó.
    _fit(tmp_path, jobs[2]["id"], 0.025, dict(rate=0.1, seed=43))
    result = selection.check_selection(campaign, jobs, tmp_path)
    assert result["failures"] == [f"{jobs[2]['id']}: el finalista no repite el caso elegido"]
