"""Memoria Permanente de Investigación y Ledger de Experimentos (`ResearchMemoryLedger`).

Persiste cada experimento, hipótesis, fallo, causa raíz, bug detectado y ganancia verificada
a lo largo de las **4 familias estrictamente independientes** construidas sobre el fundamento
Titans (`titans_original`, `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`), con
separación estricta de mercados (`US` vs `CN`) y sello temporal `2024+`, antes de promover
campeones certificados por `GainValidationGate` a Google Colab RTX PRO 6000 (`sm_120`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESEARCH_FAMILIES: tuple[str, ...] = (
    "titans_original",
    "holographic_brain",
    "mmarrs_titan",
    "titans_cognitive_core",
)

MARKETS: tuple[str, ...] = ("US", "CN")

MARKET_TRAIN_WINDOWS: dict[str, str] = {
    "US": "2000-2022",
    "CN": "2006-2022",
}

DEFAULT_COLAB_PRECISION_MODE: str = "bf16_amp"

EXPERIMENT_VERDICTS: tuple[str, ...] = (
    "KEEP",
    "REJECT",
    "INVESTIGATE",
    "BUG_CAUGHT",
)

_FAMILY_TITLES: dict[str, str] = {
    "titans_original": "Family 1 — `titans_original` (Titans-MAC + MARS-TITAN + DGX v12)",
    "holographic_brain": "Family 2 — `holographic_brain` (Holographic 3L + HRR + Neuro-AI)",
    "mmarrs_titan": "Family 3 — `mmarrs_titan` (High-SNR MoE + PC-TRM Multi-Loop + Hypergraph)",
    "titans_cognitive_core": "Family 4 — `titans_cognitive_core` (Weight-Tied Core + 6 GPU Tools)",
}

_LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    {
        "crps",
        "mae",
        "rmse",
        "pinball",
        "pinball_loss",
        "ece",
        "max_drawdown",
        "turnover",
        "brier",
    }
)

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "rank_ic_mean": ("rank_ic_mean", "rank_ic", "session_ic"),
    "rank_ic": ("rank_ic", "rank_ic_mean", "session_ic"),
    "pure_alpha_ic_mean": ("pure_alpha_ic_mean", "pure_alpha_ic"),
    "pure_alpha_ic": ("pure_alpha_ic", "pure_alpha_ic_mean"),
    "rank_ic_t_nw": ("rank_ic_t_nw", "newey_west_t", "t_stat"),
    "cov_80": ("cov_80", "cov80"),
}


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _json_sanitize(obj: Any) -> Any:
    """Convierte estructuras arbitrarias a JSON estricto RFC 8259 (NaN/Inf -> `None`)."""
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_sanitize(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_sanitize(obj.item())
        except Exception:
            pass
    return str(obj)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(text.encode("utf-8"))
    os.replace(tmp, path)


def _is_gzip_file(path: Path) -> bool:
    """Detecta archivo gzip por bytes mágicos `\\x1f\\x8b` (o sufijo `.gz` si aún vacío)."""
    if path.is_file() and path.stat().st_size >= 2:
        with path.open("rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    return path.suffix == ".gz"


def _is_gain_gate_certified(rec: ExperimentRecord) -> bool:
    """Exige `verdict=KEEP`, `gain_gate_passed=True` Y `gain_verdict` PASS/promoted real."""
    if rec.verdict != "KEEP" or not rec.gain_gate_passed:
        return False
    gv = rec.gain_verdict
    return (
        isinstance(gv, Mapping)
        and gv.get("verdict") == "PASS"
        and gv.get("promoted") is True
    )


def canonical_experiment_fingerprint(**fields: Any) -> str:
    """Calcula una huella criptográfica SHA-256 determinista para deduplicar experimentos."""
    payload = _json_sanitize(fields)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _extract_metric_value(metrics_6d: Mapping[str, Any], metric: str) -> float | None:
    """Extrae una métrica numérica finita de un informe 6D (anidado o plano)."""
    candidates = _METRIC_ALIASES.get(metric, (metric,))
    root = metrics_6d.get("granular_report", metrics_6d)
    if not isinstance(root, Mapping):
        return None

    for key in candidates:
        val = root.get(key)
        if isinstance(val, int | float) and not isinstance(val, bool) and math.isfinite(float(val)):
            return float(val)

    for sub in root.values():
        if isinstance(sub, Mapping):
            for key in candidates:
                val = sub.get(key)
                if (
                    isinstance(val, int | float)
                    and not isinstance(val, bool)
                    and math.isfinite(float(val))
                ):
                    return float(val)
    return None


def _is_higher_better(metric: str, explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return metric.lower() not in _LOWER_IS_BETTER_METRICS


def _infer_record_precision_mode(rec: ExperimentRecord) -> str | None:
    """Infiere el modo de precisión (`bf16_amp`, `fp16_amp`, `fp32_strict`) del registro."""
    explicit = rec.metrics_6d.get("precision_mode")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    env = rec.hardware_env.lower()
    if "bf16" in env:
        return "bf16_amp"
    if "fp16" in env:
        return "fp16_amp"
    if "fp32" in env:
        return "fp32_strict"
    return None


@dataclass(frozen=True)
class ExperimentRecord:
    """Registro inmutable de un experimento, hipótesis, bug o hallazgo verificado."""

    experiment_id: str
    timestamp_utc: str
    family: str
    variation: str
    market: str
    hypothesis: str
    changes_made: str
    metrics_6d: dict[str, Any]
    vs_controls_delta: dict[str, Any]
    verdict: str
    failure_mode_or_root_cause: str
    lessons_learned: str
    next_hypothesis: str
    hardware_env: str
    fingerprint_sha256: str
    gain_gate_passed: bool = False
    gain_verdict: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("experiment_id no puede estar vacío")
        if not self.timestamp_utc or not self.timestamp_utc.strip():
            raise ValueError("timestamp_utc no puede estar vacío")
        if self.family not in RESEARCH_FAMILIES:
            raise ValueError(
                f"family inválida {self.family!r}; válidas: {RESEARCH_FAMILIES}"
            )
        if not self.variation or not self.variation.strip():
            raise ValueError("variation no puede estar vacía")
        if self.market not in MARKETS:
            raise ValueError(
                f"market inválido {self.market!r}; válidos: {MARKETS} (US y CN siempre separados)"
            )
        if not self.hypothesis or not self.hypothesis.strip():
            raise ValueError("hypothesis no puede estar vacía")
        if self.verdict not in EXPERIMENT_VERDICTS:
            raise ValueError(
                f"verdict inválido {self.verdict!r}; válidos: {EXPERIMENT_VERDICTS}"
            )
        if not self.fingerprint_sha256 or not self.fingerprint_sha256.strip():
            raise ValueError("fingerprint_sha256 no puede estar vacío")
        if not isinstance(self.metrics_6d, dict):
            raise ValueError("metrics_6d debe ser un dict")
        if not isinstance(self.vs_controls_delta, dict):
            raise ValueError("vs_controls_delta debe ser un dict")

        if self.gain_verdict is not None:
            if not isinstance(self.gain_verdict, dict):
                raise ValueError("gain_verdict debe ser un dict o None")
            gv_verdict = self.gain_verdict.get("verdict")
            gv_promoted = self.gain_verdict.get("promoted")
            if self.gain_gate_passed and (gv_verdict != "PASS" or gv_promoted is False):
                raise ValueError(
                    "Contradicción: gain_gate_passed=True pero gain_verdict no es PASS/promoted"
                )

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        family: str,
        variation: str,
        market: str,
        hypothesis: str,
        changes_made: str,
        metrics_6d: dict[str, Any],
        vs_controls_delta: dict[str, Any],
        verdict: str,
        failure_mode_or_root_cause: str = "none",
        lessons_learned: str = "",
        next_hypothesis: str = "",
        hardware_env: str = "cpu_fp32_strict",
        timestamp_utc: str | None = None,
        fingerprint_sha256: str | None = None,
        gain_gate_passed: bool = False,
        gain_verdict: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Fábrica ergonómica con sello UTC y huella SHA-256 automática si se omite."""
        ts = timestamp_utc or _utc_now_iso()
        fp = fingerprint_sha256 or canonical_experiment_fingerprint(
            family=family,
            variation=variation,
            market=market,
            hypothesis=hypothesis,
            changes_made=changes_made,
        )
        return cls(
            experiment_id=experiment_id,
            timestamp_utc=ts,
            family=family,
            variation=variation,
            market=market,
            hypothesis=hypothesis,
            changes_made=changes_made,
            metrics_6d=dict(metrics_6d),
            vs_controls_delta=dict(vs_controls_delta),
            verdict=verdict,
            failure_mode_or_root_cause=failure_mode_or_root_cause,
            lessons_learned=lessons_learned,
            next_hypothesis=next_hypothesis,
            hardware_env=hardware_env,
            fingerprint_sha256=fp,
            gain_gate_passed=gain_gate_passed,
            gain_verdict=dict(gain_verdict) if gain_verdict is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializa el registro a diccionario listo para JSONL estricto."""
        return _json_sanitize(
            dict(
                experiment_id=self.experiment_id,
                timestamp_utc=self.timestamp_utc,
                family=self.family,
                variation=self.variation,
                market=self.market,
                hypothesis=self.hypothesis,
                changes_made=self.changes_made,
                metrics_6d=self.metrics_6d,
                vs_controls_delta=self.vs_controls_delta,
                verdict=self.verdict,
                failure_mode_or_root_cause=self.failure_mode_or_root_cause,
                lessons_learned=self.lessons_learned,
                next_hypothesis=self.next_hypothesis,
                hardware_env=self.hardware_env,
                fingerprint_sha256=self.fingerprint_sha256,
                gain_gate_passed=self.gain_gate_passed,
                gain_verdict=self.gain_verdict,
            )
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentRecord:
        """Reconstruye un `ExperimentRecord` desde un diccionario JSONL."""
        return cls(
            experiment_id=str(data["experiment_id"]),
            timestamp_utc=str(data["timestamp_utc"]),
            family=str(data["family"]),
            variation=str(data["variation"]),
            market=str(data["market"]),
            hypothesis=str(data["hypothesis"]),
            changes_made=str(data["changes_made"]),
            metrics_6d=dict(data.get("metrics_6d") or {}),
            vs_controls_delta=dict(data.get("vs_controls_delta") or {}),
            verdict=str(data["verdict"]),
            failure_mode_or_root_cause=str(data.get("failure_mode_or_root_cause", "")),
            lessons_learned=str(data.get("lessons_learned", "")),
            next_hypothesis=str(data.get("next_hypothesis", "")),
            hardware_env=str(data.get("hardware_env", "")),
            fingerprint_sha256=str(data["fingerprint_sha256"]),
            gain_gate_passed=bool(data.get("gain_gate_passed", False)),
            gain_verdict=(
                dict(data["gain_verdict"]) if data.get("gain_verdict") is not None else None
            ),
        )


class ResearchMemoryLedger:
    """Base de datos append-only `.jsonl`/`.jsonl.gz` + consultas + renderizador Markdown."""

    def __init__(self, ledger_path: str | Path, *, compress: bool | None = None) -> None:
        self._path = Path(ledger_path)
        self._compress = bool(compress) if compress is not None else _is_gzip_file(self._path)
        self._records: list[ExperimentRecord] = []
        self._by_fingerprint: dict[str, list[ExperimentRecord]] = {}
        if self._path.is_file():
            self._load_existing()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def compressed(self) -> bool:
        return self._compress

    def _load_existing(self) -> None:
        opener = gzip.open if _is_gzip_file(self._path) else open
        with opener(self._path, "rt", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    rec = ExperimentRecord.from_dict(data)
                except Exception as exc:
                    raise ValueError(
                        f"Entrada inválida en {self._path} línea {line_no}: {exc}"
                    ) from exc
                self._records.append(rec)
                self._by_fingerprint.setdefault(rec.fingerprint_sha256, []).append(rec)

    def already_tested(self, fingerprint_sha256: str) -> bool:
        """Indica si ya existe un experimento registrado con esta huella SHA-256."""
        return fingerprint_sha256 in self._by_fingerprint

    def get_by_fingerprint(self, fingerprint_sha256: str) -> list[ExperimentRecord]:
        """Devuelve todos los registros que comparten `fingerprint_sha256`."""
        return list(self._by_fingerprint.get(fingerprint_sha256, []))

    def record(
        self,
        entry: ExperimentRecord,
        *,
        allow_duplicate_fingerprint: bool = True,
        fsync: bool = False,
    ) -> ExperimentRecord:
        """Añade un `ExperimentRecord` al JSONL/JSONL.GZ append-only sin sobrescribir."""
        if not allow_duplicate_fingerprint and self.already_tested(entry.fingerprint_sha256):
            raise ValueError(
                f"Experimento duplicado: fingerprint_sha256={entry.fingerprint_sha256!r} "
                "ya registrado en el ledger"
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = entry.to_dict()
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        if self._compress or self._path.suffix == ".gz":
            # Miembro gzip concatenado RFC 1952: append-only real, compacto y sin pérdida
            with self._path.open("ab") as raw_fh:
                with gzip.GzipFile(fileobj=raw_fh, mode="wb", compresslevel=6) as gz_fh:
                    gz_fh.write(line.encode("utf-8"))
                raw_fh.flush()
                if fsync:
                    os.fsync(raw_fh.fileno())
        else:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                if fsync:
                    os.fsync(fh.fileno())
        self._records.append(entry)
        self._by_fingerprint.setdefault(entry.fingerprint_sha256, []).append(entry)
        return entry

    def record_with_gain_gate(
        self,
        entry: ExperimentRecord,
        gain_verdict: Any,
        *,
        allow_duplicate_fingerprint: bool = True,
        fsync: bool = False,
    ) -> ExperimentRecord:
        """Certifica `entry` con un `GainVerdict` de `GainValidationGate` y lo persiste."""
        gv_dict = (
            gain_verdict.to_dict()
            if hasattr(gain_verdict, "to_dict") and callable(gain_verdict.to_dict)
            else dict(gain_verdict)
        )
        passed = bool(gv_dict.get("promoted") and gv_dict.get("verdict") == "PASS")
        new_verdict = entry.verdict
        new_failure = entry.failure_mode_or_root_cause
        if not passed and entry.verdict == "KEEP":
            new_verdict = "REJECT"
            failed_list = gv_dict.get("failed_criteria") or []
            reason = "; ".join(str(x) for x in failed_list) or "GainValidationGate REJECT"
            new_failure = (
                f"{entry.failure_mode_or_root_cause} | GainValidationGate: {reason}"
                if entry.failure_mode_or_root_cause
                and entry.failure_mode_or_root_cause.lower() != "none"
                else f"GainValidationGate: {reason}"
            )

        merged_deltas = dict(entry.vs_controls_delta)
        gv_deltas = gv_dict.get("deltas_vs_controls")
        if isinstance(gv_deltas, Mapping) and gv_deltas:
            merged_deltas.update({str(k): dict(v) for k, v in gv_deltas.items()})

        certified = replace(
            entry,
            verdict=new_verdict,
            vs_controls_delta=merged_deltas,
            failure_mode_or_root_cause=new_failure,
            gain_gate_passed=passed,
            gain_verdict=gv_dict,
        )
        return self.record(
            certified,
            allow_duplicate_fingerprint=allow_duplicate_fingerprint,
            fsync=fsync,
        )

    def all_records(
        self,
        *,
        family: str | None = None,
        market: str | None = None,
        verdict: str | None = None,
    ) -> list[ExperimentRecord]:
        """Filtra registros en orden cronológico de inserción."""
        if family is not None and family not in RESEARCH_FAMILIES:
            raise ValueError(f"family inválida {family!r}; válidas: {RESEARCH_FAMILIES}")
        if market is not None and market not in MARKETS:
            raise ValueError(f"market inválido {market!r}; válidos: {MARKETS}")
        if verdict is not None and verdict not in EXPERIMENT_VERDICTS:
            raise ValueError(f"verdict inválido {verdict!r}; válidos: {EXPERIMENT_VERDICTS}")

        out: list[ExperimentRecord] = []
        for rec in self._records:
            if family is not None and rec.family != family:
                continue
            if market is not None and rec.market != market:
                continue
            if verdict is not None and rec.verdict != verdict:
                continue
            out.append(rec)
        return out

    def best_by_family(
        self,
        metric: str = "rank_ic_mean",
        *,
        market: str | None = None,
        verdicts: tuple[str, ...] | None = ("KEEP",),
        require_gain_gate: bool = False,
        higher_is_better: bool | None = None,
    ) -> dict[str, ExperimentRecord]:
        """Devuelve el mejor experimento por cada familia según `metric`."""
        higher = _is_higher_better(metric, higher_is_better)
        best: dict[str, tuple[float, ExperimentRecord]] = {}

        for rec in self._records:
            if market is not None and rec.market != market:
                continue
            if verdicts is not None and rec.verdict not in verdicts:
                continue
            if require_gain_gate and not _is_gain_gate_certified(rec):
                continue
            score = _extract_metric_value(rec.metrics_6d, metric)
            if score is None:
                continue
            prev = best.get(rec.family)
            if prev is None or (score > prev[0] if higher else score < prev[0]):
                best[rec.family] = (score, rec)

        return {fam: best[fam][1] for fam in RESEARCH_FAMILIES if fam in best}

    def rejected_hypotheses(
        self,
        family: str | None = None,
        *,
        market: str | None = None,
        include_bugs: bool = True,
    ) -> list[ExperimentRecord]:
        """Lista hipótesis rechazadas (y bugs capturados) para no repetir experimentos fallidos."""
        if family is not None and family not in RESEARCH_FAMILIES:
            raise ValueError(f"family inválida {family!r}; válidas: {RESEARCH_FAMILIES}")
        if market is not None and market not in MARKETS:
            raise ValueError(f"market inválido {market!r}; válidos: {MARKETS}")

        allowed = ("REJECT", "BUG_CAUGHT") if include_bugs else ("REJECT",)
        return [
            rec
            for rec in self._records
            if rec.verdict in allowed
            and (family is None or rec.family == family)
            and (market is None or rec.market == market)
        ]

    def open_hypotheses(self, family: str | None = None) -> list[dict[str, str]]:
        """Cola deduplicada de hipótesis abiertas (`INVESTIGATE` + `next_hypothesis`)."""
        if family is not None and family not in RESEARCH_FAMILIES:
            raise ValueError(f"family inválida {family!r}; válidas: {RESEARCH_FAMILIES}")

        seen: set[tuple[str, str]] = set()
        queue: list[dict[str, str]] = []
        for rec in self._records:
            if family is not None and rec.family != family:
                continue
            if rec.verdict == "INVESTIGATE":
                key = (rec.family, rec.hypothesis.strip())
                if key not in seen:
                    seen.add(key)
                    queue.append(
                        dict(
                            source_experiment_id=rec.experiment_id,
                            family=rec.family,
                            market=rec.market,
                            variation=rec.variation,
                            hypothesis=rec.hypothesis.strip(),
                            status="INVESTIGATE",
                        )
                    )
            nxt = rec.next_hypothesis.strip()
            if nxt and nxt.lower() not in ("none", "n/a", "-"):
                key = (rec.family, nxt)
                if key not in seen:
                    seen.add(key)
                    queue.append(
                        dict(
                            source_experiment_id=rec.experiment_id,
                            family=rec.family,
                            market=rec.market,
                            variation=rec.variation,
                            hypothesis=nxt,
                            status="QUEUED_NEXT",
                        )
                    )
        return queue

    def export_colab_champions_manifest(
        self,
        top_k_per_family: int = 2,
        *,
        metric: str = "rank_ic_mean",
        market: str | None = None,
        output_path: str | Path | None = None,
        higher_is_better: bool | None = None,
        precision_mode: str | None = None,
    ) -> dict[str, Any]:
        """Exporta EXCLUSIVAMENTE variaciones `KEEP` certificadas `PASS` por `GainValidationGate`.

        Listo para lanzar pre-entrenamiento completo (`2000–2022` US / `2006–2022` CN -> val `2023`,
        `2024+` sellado) en Google Colab RTX PRO 6000 Blackwell (`sm_120`), conservando las 4
        familias y ambos mercados (`US`, `CN`) estrictamente separados.
        """
        if top_k_per_family < 1:
            raise ValueError("top_k_per_family debe ser >= 1")
        if market is not None and market not in MARKETS:
            raise ValueError(f"market inválido {market!r}; válidos: {MARKETS}")
        if precision_mode is not None and (
            not isinstance(precision_mode, str) or not precision_mode.strip()
        ):
            raise ValueError("precision_mode debe ser una cadena no vacía o None")

        higher = _is_higher_better(metric, higher_is_better)
        target_markets = (market,) if market is not None else MARKETS
        train_windows_by_market = {m: MARKET_TRAIN_WINDOWS[m] for m in target_markets}
        train_window: str | dict[str, str] = (
            MARKET_TRAIN_WINDOWS[market] if market is not None else dict(train_windows_by_market)
        )

        champions_by_family: dict[str, list[dict[str, Any]]] = {}
        champions_by_family_and_market: dict[str, dict[str, list[dict[str, Any]]]] = {}
        exported_records: list[ExperimentRecord] = []
        total_champions = 0

        for fam in RESEARCH_FAMILIES:
            fam_by_mkt: dict[str, list[dict[str, Any]]] = {}
            fam_combined: list[dict[str, Any]] = []
            # Seleccionar `top_k_per_family` estrictamente por cada par (familia, mercado)
            for mkt in target_markets:
                best_per_var: dict[str, tuple[float, ExperimentRecord]] = {}
                for rec in self._records:
                    if rec.family != fam or rec.market != mkt:
                        continue
                    if not _is_gain_gate_certified(rec):
                        continue
                    score = _extract_metric_value(rec.metrics_6d, metric)
                    if score is None:
                        continue
                    prev = best_per_var.get(rec.variation)
                    if prev is None or (score > prev[0] if higher else score < prev[0]):
                        best_per_var[rec.variation] = (score, rec)

                ranked = sorted(
                    best_per_var.values(),
                    key=lambda item: (-item[0] if higher else item[0], item[1].variation),
                )[:top_k_per_family]

                mkt_entries: list[dict[str, Any]] = []
                for score, rec in ranked:
                    exported_records.append(rec)
                    rec_prec = _infer_record_precision_mode(rec)
                    entry = dict(
                        experiment_id=rec.experiment_id,
                        family=rec.family,
                        variation=rec.variation,
                        market=rec.market,
                        train_window=MARKET_TRAIN_WINDOWS[rec.market],
                        precision_mode=rec_prec,
                        hardware_env=rec.hardware_env,
                        verdict=rec.verdict,
                        gain_gate_passed=rec.gain_gate_passed,
                        ranking_metric=metric,
                        ranking_score=score,
                        metrics_6d=_json_sanitize(rec.metrics_6d),
                        vs_controls_delta=_json_sanitize(rec.vs_controls_delta),
                        fingerprint_sha256=rec.fingerprint_sha256,
                        gain_verdict=_json_sanitize(rec.gain_verdict),
                    )
                    mkt_entries.append(entry)
                    fam_combined.append(entry)
                fam_by_mkt[mkt] = mkt_entries
                total_champions += len(mkt_entries)

            champions_by_family[fam] = fam_combined
            champions_by_family_and_market[fam] = fam_by_mkt

        if precision_mode is not None:
            resolved_precision_mode = precision_mode.strip()
        else:
            inferred_modes = {
                m
                for rec in exported_records
                if (m := _infer_record_precision_mode(rec)) is not None
            }
            resolved_precision_mode = (
                next(iter(inferred_modes))
                if len(inferred_modes) == 1
                else DEFAULT_COLAB_PRECISION_MODE
            )

        for fam_list in champions_by_family.values():
            for entry in fam_list:
                if entry["precision_mode"] is None:
                    entry["precision_mode"] = resolved_precision_mode

        contract_body = dict(
            schema_version=1,
            hardware_target="NVIDIA RTX PRO 6000 Blackwell (sm_120 / GB202 / 96GB GDDR7 / 188 SMs)",
            precision_mode=resolved_precision_mode,
            train_window=train_window,
            train_windows_by_market=train_windows_by_market,
            val_window="2023",
            sealed_test_start_year=2024,
            markets_strictly_separate=True,
            families_strictly_independent=True,
            selection_metric=metric,
            top_k_per_family=int(top_k_per_family),
            market_filter=market,
            total_champions=total_champions,
            champions_by_family=champions_by_family,
            champions_by_family_and_market=champions_by_family_and_market,
        )
        manifest_sha256 = canonical_experiment_fingerprint(**contract_body)
        manifest = dict(
            **contract_body,
            manifest_sha256=manifest_sha256,
            exported_at_utc=_utc_now_iso(),
        )

        if output_path is not None:
            out_p = Path(output_path)
            if out_p.suffix == ".gz":
                compact = json.dumps(
                    manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                _atomic_write_gzip_text(out_p, compact + "\n")
            else:
                _atomic_write_text(
                    out_p,
                    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                )
        return manifest

    def render_markdown(
        self,
        path: str | Path | None = None,
        *,
        title: str = "MMARRS-TITAN Permanent Research Memory & Experiment Ledger",
    ) -> str:
        """Renderiza el ledger completo en Markdown estructurado por las 4 familias."""
        counts_verdict = {v: 0 for v in EXPERIMENT_VERDICTS}
        counts_family = {f: 0 for f in RESEARCH_FAMILIES}
        for rec in self._records:
            counts_verdict[rec.verdict] += 1
            counts_family[rec.family] += 1

        def _md_cell(text: str) -> str:
            return " ".join(str(text).splitlines()).replace("|", "\\|").strip()

        def _fmt_metric(rec: ExperimentRecord, key: str, digits: int = 4) -> str:
            val = _extract_metric_value(rec.metrics_6d, key)
            return f"{val:.{digits}f}" if val is not None else "—"

        def _fmt_delta(rec: ExperimentRecord) -> str:
            if not rec.vs_controls_delta:
                return "—"
            parts: list[str] = []
            for k, v in rec.vs_controls_delta.items():
                if isinstance(v, Mapping):
                    inner = ", ".join(
                        f"{ik}={iv:+.4f}" if isinstance(iv, int | float) else f"{ik}={iv}"
                        for ik, iv in v.items()
                    )
                    parts.append(f"{k}({inner})")
                elif isinstance(v, int | float):
                    parts.append(f"{k}={v:+.4f}")
                else:
                    parts.append(f"{k}={v}")
            return "; ".join(parts)

        lines: list[str] = [
            f"# {title}",
            "",
            "> **Strict Research Invariants:** 4 strictly independent Titans-foundation families "
            "(`titans_original`, `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core` — "
            "zero soup-blending); `US` and `CN` trained and evaluated strictly separately (FM-8); "
            "full-history `2000–2022` train -> `2023` val/cal -> **`2024+` strictly sealed**; "
            "promotion to Google Colab RTX PRO 6000 (`sm_120`) requires `verdict=KEEP` + "
            "`GainValidationGate PASS`.",
            "",
            "## Ledger Summary",
            "",
            f"- **Total Records:** `{len(self._records)}` "
            f"(`KEEP={counts_verdict['KEEP']}`, `REJECT={counts_verdict['REJECT']}`, "
            f"`INVESTIGATE={counts_verdict['INVESTIGATE']}`, "
            f"`BUG_CAUGHT={counts_verdict['BUG_CAUGHT']}`)",
            "- **By Family:** "
            + ", ".join(f"`{f}={counts_family[f]}`" for f in RESEARCH_FAMILIES),
            "",
            "## Verified Gains Leaderboard",
            "",
            "| Family | Market | Variation | Experiment ID | Rank/Session IC | Net Sharpe 1x | "
            "GainGate | Delta vs Controls | Key Lesson |",
            "| :--- | :---: | :--- | :--- | ---: | ---: | :---: | :--- | :--- |",
        ]

        def _keep_sort_key(rec: ExperimentRecord) -> tuple[int, int, float, str]:
            val = _extract_metric_value(rec.metrics_6d, "rank_ic_mean")
            return (
                RESEARCH_FAMILIES.index(rec.family),
                0 if val is not None else 1,
                -(val if val is not None else 0.0),
                rec.experiment_id,
            )

        keeps = [r for r in self._records if r.verdict == "KEEP"]
        keeps_sorted = sorted(keeps, key=_keep_sort_key)
        if keeps_sorted:
            for rec in keeps_sorted:
                gate_str = "PASS" if _is_gain_gate_certified(rec) else "PENDING"
                lines.append(
                    f"| `{rec.family}` | `{rec.market}` | `{rec.variation}` | "
                    f"`{rec.experiment_id}` | {_fmt_metric(rec, 'rank_ic_mean')} | "
                    f"{_fmt_metric(rec, 'net_sharpe_1x', 2)} | **{gate_str}** | "
                    f"{_md_cell(_fmt_delta(rec))} | {_md_cell(rec.lessons_learned)} |"
                )
        else:
            lines.append("| — | — | — | — | — | — | — | — | No `KEEP` entries yet |")

        lines.extend(["", "## Per-Family Experiment Ledgers", ""])
        for fam in RESEARCH_FAMILIES:
            lines.extend([f"### {_FAMILY_TITLES[fam]}", ""])
            fam_recs = [r for r in self._records if r.family == fam]
            if not fam_recs:
                lines.extend(["_No records logged yet for this family._", ""])
                continue
            lines.extend(
                [
                    "| ID | Market | Variation | Verdict | Rank IC | Hypothesis | "
                    "Changes Made | Root Cause / Lesson |",
                    "| :--- | :---: | :--- | :---: | ---: | :--- | :--- | :--- |",
                ]
            )
            for rec in fam_recs:
                summary_note = (
                    f"**Root cause:** {rec.failure_mode_or_root_cause}. **Lesson:** "
                    f"{rec.lessons_learned}"
                    if rec.failure_mode_or_root_cause
                    and rec.failure_mode_or_root_cause.lower() != "none"
                    else rec.lessons_learned
                )
                lines.append(
                    f"| `{rec.experiment_id}` | `{rec.market}` | `{rec.variation}` | "
                    f"**{rec.verdict}** | {_fmt_metric(rec, 'rank_ic_mean')} | "
                    f"{_md_cell(rec.hypothesis)} | {_md_cell(rec.changes_made)} | "
                    f"{_md_cell(summary_note)} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Graveyard of Rejected Ideas & Root Causes (Never Repeat!)",
                "",
                "| ID | Family | Market | Verdict | Variation | Failure Mode / Root Cause | "
                "Permanent Rule / Lesson Learned |",
                "| :--- | :--- | :---: | :---: | :--- | :--- | :--- |",
            ]
        )
        graveyard = self.rejected_hypotheses(include_bugs=True)
        if graveyard:
            for rec in graveyard:
                lines.append(
                    f"| `{rec.experiment_id}` | `{rec.family}` | `{rec.market}` | "
                    f"**{rec.verdict}** | `{rec.variation}` | "
                    f"{_md_cell(rec.failure_mode_or_root_cause)} | "
                    f"{_md_cell(rec.lessons_learned)} |"
                )
        else:
            lines.append("| — | — | — | — | — | — | None |")

        lines.extend(
            [
                "",
                "## Open Hypotheses Queue",
                "",
                "| Source ID | Family | Market | Status | Variation | Next Hypothesis to Test |",
                "| :--- | :--- | :---: | :---: | :--- | :--- |",
            ]
        )
        open_q = self.open_hypotheses()
        if open_q:
            for item in open_q:
                lines.append(
                    f"| `{item['source_experiment_id']}` | `{item['family']}` | "
                    f"`{item['market']}` | `{item['status']}` | `{item['variation']}` | "
                    f"{_md_cell(item['hypothesis'])} |"
                )
        else:
            lines.append("| — | — | — | — | — | None |")

        text = "\n".join(lines) + "\n"
        if path is not None:
            _atomic_write_text(Path(path), text)
        return text


def build_initial_session_records() -> list[ExperimentRecord]:
    """Construye la semilla empírica verificada de esta sesión y de las ablaciones reales 2013."""
    ts = "2026-11-23T11:00:00+00:00"
    records: list[ExperimentRecord] = [
        # --- FAMILY 1: titans_original (Verified Gains & Real 2013 Ablations) ---
        ExperimentRecord.create(
            experiment_id="dgx_2013_t01_xrank_ic",
            timestamp_utc=ts,
            family="titans_original",
            variation="t02_titans_mac_online_xrank_ic",
            market="US",
            hypothesis=(
                "TechnicalDense (128x2) + Session IC term (w=3) + cross-sectional Gaussian "
                "session ranks (`xrank`) lifts Titans-MAC above LightGBM OOS on US 2013"
            ),
            changes_made=(
                "Added 46 causal technical features (`long_horizon_v2`), `TechnicalDense` 128x2, "
                "Session IC loss w=3, and session Gaussian ranks (`xrank`)"
            ),
            metrics_6d=dict(rank_ic_mean=0.1163, pure_alpha_ic_mean=0.1107, rank_ic_t_nw=4.4),
            vs_controls_delta=dict(
                original_recipe=dict(delta_rank_ic=0.0387),
                no_xrank_base=dict(delta_rank_ic=0.0055),
                lightgbm_oos=dict(delta_rank_ic=0.0041),
            ),
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "Session IC rose 0.0776 -> 0.1107 -> 0.1163 (+0.0055, t=4.4), first Titans "
                "beating LightGBM OOS (0.1122, p=0.017) on US 2013"
            ),
            next_hypothesis=(
                "Verify `t10_titans_tabm8` + `long_horizon_v5` + `masked_v1` ranks on clean "
                "2014-2023 US full-history origins"
            ),
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_2013_t02_3seed_online_mem",
            timestamp_utc=ts,
            family="titans_original",
            variation="t02_titans_mac_online_3seed_rank_ens",
            market="US",
            hypothesis=(
                "3-seed cross-sectional rank ensembling reduces seed variance and cleanly "
                "separates online Titans memory from `mac_frozen` and `mac_disabled`"
            ),
            changes_made="Averaged cross-sectional ranks across seeds (42, 43, 44)",
            metrics_6d=dict(rank_ic_mean=0.1193, rank_ic_t_nw=3.8),
            vs_controls_delta=dict(
                mac_frozen=dict(delta_rank_ic=0.0031, t_stat=3.5),
                mac_disabled=dict(delta_rank_ic=0.0044, t_stat=3.8),
            ),
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "3-seed rank ensembles add +0.002..+0.004 IC and unlock statistically significant "
                "online memory gain (+0.0031 vs frozen t=3.5; +0.0044 vs disabled t=3.8)"
            ),
            next_hypothesis=(
                "Test `t11_titans_crmm`, `t12_titans_lfm_sw`, and `t14_titans_neuromod_sota` "
                "with mandatory 3-seed rank ensembling on RTX PRO 6000"
            ),
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_2013_t03_wp5_combined_organ",
            timestamp_utc=ts,
            family="titans_original",
            variation="wp5_combined_mature_memory_organ",
            market="US",
            hypothesis=(
                "Delayed matured-residual per-asset EMA (h=21, kappa=0.1) + neighbor bank "
                "(W=63, k=256) + HRR-delta post-training organ adds real OOS order"
            ),
            changes_made=(
                "Combined `asset_lab` EMA, 63d 256-NN mature-residual bank, and HRR-delta rule "
                "with strict permuted-asset and permuted-label placebos"
            ),
            metrics_6d=dict(rank_ic_mean=0.1165, rank_ic_t_nw=2.42),
            vs_controls_delta=dict(
                static_base=dict(delta_rank_ic=0.0027),
                permuted_asset_placebo=dict(delta_rank_ic=-0.0010, t_stat=-4.53),
            ),
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "Adds +0.0020..+0.0034 IC across 13 models with negative placebos on all 13; "
                "online gain is asset-specific persistent residual (+0.0017, t=2.42)"
            ),
            next_hypothesis=(
                "Validate WP5 combined organ on 2008/2020/2022 crisis walk-forward origins"
            ),
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_2013_t04_f7b_order_level_decouple",
            timestamp_utc=ts,
            family="titans_original",
            variation="t03_f7b_lad_noic_mondrian_aci",
            market="US",
            hypothesis=(
                "Decoupling cross-sectional order (IC model) from return level (LAD scale) and "
                "quantile widths (`noic` pinball twin + Mondrian+ACI) resolves FM-7 scale inflation"
            ),
            changes_made=(
                "Paired IC-order predictor with LAD point recalibration, `noic` pinball-only "
                "widths, and volatility-decile Mondrian+ACI conformal intervals"
            ),
            metrics_6d=dict(ece=0.028, cov_90=0.9003, pinball_reduction_pct=43.0),
            vs_controls_delta=dict(
                coupled_ic_quantiles=dict(delta_ece=-0.120, pinball_pct=-43.0),
            ),
            verdict="KEEP",
            failure_mode_or_root_cause="FM-7 order-vs-level scale inflation (1.1-1.9 vs 0.14)",
            lessons_learned=(
                "FM-7 solved: Session IC loss inflates raw dispersion ~10x (1.1-1.9 vs 0.14); "
                "decoupling order from level + `noic` pinball twin cuts pinball -43%, ECE "
                "0.148->0.028, achieving 90.03% calibrated coverage"
            ),
            next_hypothesis="Run F7-b Stage-2 calibration over every Stage-1 champion family",
            hardware_env="dgx_fp32_strict_posttraining_v10",
        ),
        # --- FAMILY 1 REJECTED IDEAS (FM-7, FM-8, PLR, H1-a, H3-a, H5-a, v1 MAE stop) ---
        ExperimentRecord.create(
            experiment_id="dgx_rej_fm8_joint_us_cn",
            timestamp_utc=ts,
            family="titans_original",
            variation="joint_us_cn_5008_assets",
            market="CN",
            hypothesis="Joint US (4,213) + CN (810) training improves cross-market generalization",
            changes_made="Pooled US and CN sessions into shared backbone training weights",
            metrics_6d=dict(rank_ic_t_nw=-2.6),
            vs_controls_delta=dict(
                us_separate=dict(delta_rank_ic=-0.0045),
                cn_separate=dict(delta_rank_ic=-0.0085),
            ),
            verdict="REJECT",
            failure_mode_or_root_cause=(
                "FM-8 cross-market interference: joint US+CN hurts BOTH markets (-0.004..-0.005 "
                "US, -0.007..-0.010 CN, t < -2.5) due to microstructure, T+1/price limits & CSI 300"
            ),
            lessons_learned=(
                "NEVER pool US and CN training weights (FM-8). Always train US and CN strictly "
                "separately and enforce CSI 300 factor (`market_model: required`) for CN"
            ),
            next_hypothesis="Run independent CN-only ablations with CSI 300 factor neutralization",
            hardware_env="dgx_fp32_strict_us_cn",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_rej_plr_periodic_embeddings",
            timestamp_utc=ts,
            family="titans_original",
            variation="titans_plr_periodic_embeddings",
            market="US",
            hypothesis="PLR periodic numerical embeddings improve tabular financial features",
            changes_made="Replaced linear/dense numerical input projection with PLR embeddings",
            metrics_6d=dict(rank_ic_mean=0.1020),
            vs_controls_delta=dict(dense_base=dict(delta_rank_ic=-0.0105)),
            verdict="REJECT",
            failure_mode_or_root_cause=(
                "PLR periodic embeddings overfit high-frequency noise in non-stationary returns "
                "(-0.004 to -0.017 OOS Session IC)"
            ),
            lessons_learned=(
                "Permanently avoid PLR periodic embeddings on financial cross-sections; use "
                "session Gaussian ranks (`xrank`, `masked_v1`) + `TechnicalDense` / `TabM`"
            ),
            next_hypothesis="none",
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_rej_h1a_shared_online_head",
            timestamp_utc=ts,
            family="titans_original",
            variation="h1a_shared_online_linear_head",
            market="US",
            hypothesis="Re-fitting shared linear head online on matured daily labels adds IC",
            changes_made="Online ridge head update (halflife=63, shrink=0.3) across 2011-2013",
            metrics_6d=dict(rank_ic_mean=0.1148, rank_ic_t_nw=0.05),
            vs_controls_delta=dict(
                static_base=dict(delta_rank_ic=0.0001),
                shuffled_placebo=dict(delta_rank_ic=-0.1337),
            ),
            verdict="REJECT",
            failure_mode_or_root_cause=(
                "Overfits calibration (+0.003) with zero eval lift (+0.0001, t=0.05); shared "
                "feature-return slope does not drift fast enough to beat static weights"
            ),
            lessons_learned=(
                "Do not pursue naive session-level online head refitting or uncalibrated "
                "abstention; focus online adaptation on per-asset residuals and episodic recall"
            ),
            next_hypothesis="none",
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_rej_h3a_daily_macro_diff",
            timestamp_utc=ts,
            family="titans_original",
            variation="h3a_daily_macro_diff_interaction",
            market="US",
            hypothesis=(
                "Daily macro differences crossed with per-asset betas add cross-sectional IC"
            ),
            changes_made="Added daily shock x per-asset beta interaction term",
            metrics_6d=dict(rank_ic_mean=0.1133, rank_ic_t_nw=-1.86),
            vs_controls_delta=dict(static_base=dict(delta_rank_ic=-0.0014)),
            verdict="REJECT",
            failure_mode_or_root_cause=(
                "139 of 140 macro indicators publish weekly/monthly (only 1 changes daily), so "
                "daily diff interactions inject pure noise (-0.0002 to -0.0014, t=-1.86)"
            ),
            lessons_learned=(
                "Never compute naive daily macro diffs; model macro via release-day event "
                "surprises (when data age resets to 0) crossed with historical betas"
            ),
            next_hypothesis=(
                "H3-b: Event-day macro release surprise x causal per-asset sensitivity"
            ),
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="dgx_rej_h5a_adam_beta2_and_mae_stop",
            timestamp_utc=ts,
            family="titans_original",
            variation="h5a_adam_beta2_095_and_v1_mae_stop",
            market="US",
            hypothesis=(
                "Adam beta2=0.95 / 5-session accum or MAE early stopping improves convergence"
            ),
            changes_made="Tested beta2=0.95, 5-session gradient accum, wd=0.01, and MAE stopping",
            metrics_6d=dict(rank_ic_mean=0.1136, rank_ic_t_nw=-1.39),
            vs_controls_delta=dict(
                base_adamw=dict(delta_rank_ic=-0.0011),
            ),
            verdict="REJECT",
            failure_mode_or_root_cause=(
                "beta2=0.95 (-0.0018 cal), 5-session accum (-0.0011), wd=0.01 (-0.0001); "
                "MAE stopping (v1) froze episodic readers at epoch 0 because MAE ~ rho^2/2 "
                "is blind to cross-sectional IC"
            ),
            lessons_learned=(
                "Keep standard AdamW(fused=True) with 3x-lower router LR and select epochs by "
                "WP9 adaptive val Session Rank IC stopping (smoothing=3, patience=5, top-3 SWA)"
            ),
            next_hypothesis="Evaluate Muon for 2D hidden/expert matrices vs fused AdamW baseline",
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        # --- BUGS CAUGHT & FIXED (Permanent Regression Shield!) ---
        ExperimentRecord.create(
            experiment_id="bug_01_alpha_eval_covariate_vs_target",
            timestamp_utc=ts,
            family="titans_original",
            variation="eval_protocol_audit",
            market="US",
            hypothesis=(
                "Audit point-in-time separation of `alpha_evaluation` covariates vs `target`"
            ),
            changes_made=(
                "Enforced strict separation between causal input covariates and matured "
                "market-residual target y_{i,t} = r^OC_{i,t+1} - alpha_hat - beta_hat * r_m"
            ),
            metrics_6d=dict(),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "`alpha_evaluation` covariate vs `target` misalignment/leakage: treating "
                "evaluation alpha / target proxy as covariate corrupts causality & inflates IC"
            ),
            lessons_learned=(
                "Strictly separate causal point-in-time covariates from `target`; always run "
                "bit-for-bit causality tests (`mutating t > t0` alters 0 bits at `<= t0`)"
            ),
            next_hypothesis=(
                "Enforce automated causality + seal gates in every training entry point"
            ),
            hardware_env="cpu_audit_fp32_strict",
        ),
        ExperimentRecord.create(
            experiment_id="bug_02_almgren_chriss_zscored_units",
            timestamp_utc=ts,
            family="mmarrs_titan",
            variation="granular_evaluator_p_plus_costs",
            market="US",
            hypothesis=(
                "Audit physical units of Almgren-Chriss market impact in net Sharpe 1x/2x/3x"
            ),
            changes_made=(
                "Separated raw positive physical volatility/ADV units for Almgren-Chriss "
                "square-root impact from z-scored neural network features"
            ),
            metrics_6d=dict(),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "Z-scored liquidity/volatility passed into Almgren-Chriss market impact breaks "
                "physical units (produces negative/distorted impact costs instead of bps impact)"
            ),
            lessons_learned=(
                "NEVER feed z-scored liq/vol into Almgren-Chriss impact formulas; use strictly "
                "positive physical daily vol and ADV/turnover units for 1x/2x/3x net Sharpe"
            ),
            next_hypothesis="Audit liquid-third net Sharpe 1x/2x/3x across all 4 families",
            hardware_env="cpu_audit_fp32_strict",
        ),
        ExperimentRecord.create(
            experiment_id="bug_03_triton_constexpr_fp64_truncation",
            timestamp_utc=ts,
            family="mmarrs_titan",
            variation="blackwell_kernels_sm120",
            market="US",
            hypothesis="Verify IEEE-754 FP64 parity (atol=1e-12) in sm_120 Triton fused kernels",
            changes_made=(
                "Converted `clip_val`, `eps`, and `scale_over_q` to `tl.constexpr` materialized "
                "via `tl.full([], c, x.dtype)` + `force_clone=True` for CUDA Graphs (#23)"
            ),
            metrics_6d=dict(tests_passed=32.0),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "Passing Python float constants as runtime Triton args silently marshaled via "
                "FP32, truncating FP64 parity from 1e-15 to 1e-7; CUDA Graph capture also "
                "aliased exact-bucket caller tensors without `force_clone=True`"
            ),
            lessons_learned=(
                "Always declare Triton scalar constants as `tl.constexpr` + `tl.full(..., dtype)` "
                "(verified `0x3FCC71C71C71C71C` in offline `sm_120` TTIR) and clone static buffers"
            ),
            next_hypothesis=(
                "Benchmark fused kernels on live RTX PRO 6000 `sm_120` via `nsys`/`ncu`"
            ),
            hardware_env="rtx_pro_6000_sm120_offline_compile",
        ),
        ExperimentRecord.create(
            experiment_id="bug_04_cuda_pinned_double_buffer_dma_race",
            timestamp_utc=ts,
            family="mmarrs_titan",
            variation="zero_copy_session_store_h2d",
            market="US",
            hypothesis="Audit async pinned double-buffered H2D pipeline for CUDA DMA races",
            changes_made=(
                "Added per-slot `torch.cuda.Event()` (`slot_events[slot].synchronize()` before "
                "CPU staging overwrite, `.record(stream)` after `.to(..., non_blocking=True)`) "
                "and `tensor.record_stream(cur_stream)` on yielded tensors (#25/#27)"
            ),
            metrics_6d=dict(tests_passed=13.0),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "CUDA pinned double-buffer DMA race: `current_stream.wait_stream(stream)` is "
                "GPU-only and does not block host CPU, allowing CPU `prev.copy_(src)` to corrupt "
                "`staging[0]` while async H2D DMA is still in flight"
            ),
            lessons_learned=(
                "Every pinned double-buffer slot MUST own a `torch.cuda.Event` synchronized "
                "before CPU reuse, plus `record_stream` on every yielded GPU tensor"
            ),
            next_hypothesis="Replace legacy `AsyncSessionPrefetcher` with `ZeroCopySessionStore`",
            hardware_env="rtx_pro_6000_sm120_dma_audit",
        ),
        ExperimentRecord.create(
            experiment_id="bug_05_bf16_linalg_solve_and_fp16_mask_nan",
            timestamp_utc=ts,
            family="titans_cognitive_core",
            variation="c02_cog_core_soft_tools",
            market="US",
            hypothesis="Verify Blackwell BF16/FP16 AMP safety and zero-lag monotonic halt gate",
            changes_made=(
                "Added `_stable_calc_dtype` FP32 upcast for `linalg.solve`, `logsumexp`, gating, "
                "and `torch.where(filled_cnt > 0.5, ..., zeros)`; evaluated `e_cand` directly on "
                "`h_cand` for step-0 monotonic damping (#21/#28)"
            ),
            metrics_6d=dict(tests_passed=7.0),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "`bfloat16`/`float16` `torch.linalg.solve` instability and FP16 `-1e6` mask "
                "overflow (`-inf * 0 = NaN`) in empty HRR-Hopfield lookup"
            ),
            lessons_learned=(
                "Under Blackwell `bf16_amp`, always upcast `linalg.solve`, Cholesky, and "
                "`logsumexp` via `_stable_calc_dtype` and guard empty buffers with `torch.where`"
            ),
            next_hypothesis=(
                "Compare `c02_cog_core_soft_tools` & `c03_cog_core_sparsemax_tools` vs "
                "`c00_cog_core_no_tools` & `c01_cog_core_uniform_tools` on full 2000-2023"
            ),
            hardware_env="rtx_pro_6000_sm120_bf16_audit",
        ),
        ExperimentRecord.create(
            experiment_id="bug_06_cpu_55session_underfit_150k_moe",
            timestamp_utc=ts,
            family="mmarrs_titan",
            variation="v07_v13_cpu_smoke_ablation",
            market="US",
            hypothesis="Diagnose why 55-session CPU smoke tests underfit 150K-param MoE+loop",
            changes_made=(
                "Documented sample-complexity threshold and built `LongRunningTitansHarness` + "
                "`ResearchTelemetryMonitor` + MPS launcher for full 2000-2023 (~5,700 sessions)"
            ),
            metrics_6d=dict(train_sessions_smoke=55.0, full_history_sessions=5700.0),
            vs_controls_delta={},
            verdict="BUG_CAUGHT",
            failure_mode_or_root_cause=(
                "55-session CPU smoke tests severely underfit 150K-param MoE + multi-loop models "
                "(<20 updates per routed expert across 6 experts x 3-4 loops), creating false "
                "negatives vs compact dense baselines"
            ),
            lessons_learned=(
                "Use CPU smoke runs ONLY for unit/causality/gradient/zero-sync verification; "
                "judge MoE, Hypergraph, Holographic 3L, and Cognitive Core capacity ONLY on full "
                "2000-2022 (~5,700 sessions x 4,200 assets) -> 2023 val on RTX PRO 6000"
            ),
            next_hypothesis=(
                "Run full 2000-2022 -> 2023 4-family certification wave on Colab RTX PRO 6000 "
                "with `GainValidationGate` and `ResearchTelemetryMonitor`"
            ),
            hardware_env="cpu_55session_smoke_vs_sm120",
        ),
        # --- FAMILY 2: holographic_brain (Verified Episodic Finding + Frontier Open Hypothesis) ---
        ExperimentRecord.create(
            experiment_id="dgx_2013_h01_h4a_full_session_bank",
            timestamp_utc=ts,
            family="holographic_brain",
            variation="h4a_63d_full_session_episodic_bank",
            market="US",
            hypothesis=(
                "Expanding episodic bank from 1,024 surprise rows to 63 full sessions (~250k "
                "rows) storing matured residuals unlocks episodic readout IC"
            ),
            changes_made=(
                "Replaced tiny surprise-only bank with 63-session full-row mature-residual bank "
                "(256 cosine NN) and verified against permuted-value placebo"
            ),
            metrics_6d=dict(rank_ic_mean=0.1164, rank_ic_t_nw=2.78),
            vs_controls_delta=dict(
                static_base=dict(delta_rank_ic=0.0016),
                permuted_bank_placebo=dict(delta_rank_ic=-0.0006, t_stat=-2.42),
            ),
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "Bank content and size (not key projection complexity) caused inert readers: "
                "random-projection keys over 63 full sessions gained +0.0016 (t=2.78) vs "
                "hidden-state keys +0.0007 (t=1.50) and placebo -0.0006"
            ),
            next_hypothesis=(
                "Evaluate independent Family 2 `StandaloneHolographicBrainModel` (3L + Dendritic "
                "+ Lateral Inhibition + 3-Factor Dopamine + Sharp-Wave Replay) + `HSI`/`CSRF` "
                "on full 2000-2023 RTX PRO 6000"
            ),
            hardware_env="dgx_fp32_strict_us_2013",
        ),
        ExperimentRecord.create(
            experiment_id="wave2_h02_standalone_neuro_brain",
            timestamp_utc=ts,
            family="holographic_brain",
            variation="h07_standalone_holographic_neuro_brain",
            market="US",
            hypothesis=(
                "Independent Family 2 Neuro-AI brain (`HolographicBrain3L` + Dendritic Coincidence "
                "+ Lateral Inhibition + 3-Factor Dopamine PT1 + Hippocampal Replay, NO MoE) beats "
                "H1/H2/H3/H4 controls on crisis retention (2008/2020/2022)"
            ),
            changes_made=(
                "Built `neuro_brain_sota.py` strictly isolated from MoE and Cognitive Core, "
                "hard cap <= 4 PC cycles, zero host syncs"
            ),
            metrics_6d=dict(),
            vs_controls_delta={},
            verdict="INVESTIGATE",
            failure_mode_or_root_cause="none",
            lessons_learned="Ready for full 2000-2023 multi-crisis walk-forward test on sm_120",
            next_hypothesis=(
                "Ablate `use_dendritic`, `use_lateral_inhibition`, `use_three_factor`, and "
                "`use_replay` against matched controls on US and CN separately"
            ),
            hardware_env="pending_colab_rtx_pro_6000_sm120",
        ),
        # --- FAMILY 3: mmarrs_titan (Wave-1/2 Verified Infra + Open Full-History Hypothesis) ---
        ExperimentRecord.create(
            experiment_id="wave1_m01_mmarrs_v13_full_sm120",
            timestamp_utc=ts,
            family="mmarrs_titan",
            variation="v13b_mmarrs_multimodal_sota",
            market="US",
            hypothesis=(
                "Independent Family 3 (`HighSNRMoELayer` Aux-Loss-Free + PC-TRM `max_loops=3` + "
                "RMT Hypergraph + `CrossModalConflictFusion` + `SkewStudentTDistributionHead`) "
                "passes `GainValidationGate` against all 5 matched controls on 2000-2023"
            ),
            changes_made=(
                "Verified zero-sync Blackwell kernels (32 tests), long-running harness + 2024+ "
                "seal (11 tests), telemetry monitor (6 tests), zero-copy store (13 tests) (#31)"
            ),
            metrics_6d=dict(),
            vs_controls_delta={},
            verdict="INVESTIGATE",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "All hot-path `.item()`/`.cpu()` syncs in `mor_loom_pc.py`, `neuromod_memory.py`, "
                "`hypergraph_hrr.py`, `model.py` must be eliminated (`audit_zero_host_syncs == 0`) "
                "before Colab launch (#32)"
            ),
            next_hypothesis=(
                "Execute paired 14-variation + `v13b_mmarrs_multimodal_sota` MPS waves (6-8 "
                "workers in 96GB GDDR7) on 2000-2022 -> 2023 val with 3-seed rank ensembles"
            ),
            hardware_env="pending_colab_rtx_pro_6000_sm120",
        ),
        # --- FAMILY 4: titans_cognitive_core (Verified Unit GREEN + Open Hypothesis) ---
        ExperimentRecord.create(
            experiment_id="wave1_c01_cognitive_core_tools_sm120",
            timestamp_utc=ts,
            family="titans_cognitive_core",
            variation="c02_cog_core_soft_tools",
            market="US",
            hypothesis=(
                "Independent Family 4 (`TitansCognitiveCore` weight-tied K=1..3 + "
                "`ContextEngineerGPU` + 6 GPU differentiable quant tools + `AdaptiveHaltGate`) "
                "beats `c00_cog_core_no_tools` and `c01_cog_core_uniform_tools` on 2000-2023"
            ),
            changes_made=(
                "Verified 7/7 tests GREEN (#28/#31): weight-tied parameter efficiency, zero host "
                "syncs, BF16/FP16 `_stable_calc_dtype` safety, zero-lag monotonic damping"
            ),
            metrics_6d=dict(),
            vs_controls_delta={},
            verdict="INVESTIGATE",
            failure_mode_or_root_cause="none",
            lessons_learned=(
                "Compact weight-tied core keeps parameter count constant from K=1 to K=4 while "
                "delegating exact math (RMT decomp, HRR lookup, P+ feasibility) to 6 GPU tools"
            ),
            next_hypothesis=(
                "Run paired Family 4 ablation (`c00_no_tools`, `c01_uniform`, `c02_soft`, "
                "`c03_sparsemax`, `c04_no_context_eng`) on 2000-2022 -> 2023 val (US & CN)"
            ),
            hardware_env="pending_colab_rtx_pro_6000_sm120",
        ),
    ]
    return records


__all__ = [
    "DEFAULT_COLAB_PRECISION_MODE",
    "EXPERIMENT_VERDICTS",
    "MARKETS",
    "MARKET_TRAIN_WINDOWS",
    "RESEARCH_FAMILIES",
    "ExperimentRecord",
    "ResearchMemoryLedger",
    "build_initial_session_records",
    "canonical_experiment_fingerprint",
]
