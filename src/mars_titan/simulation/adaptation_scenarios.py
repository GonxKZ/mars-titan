"""Escenarios técnicos para separar memoria, adaptación y fricción de ejecución."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, sha256

from .market import MarketTape
from .portfolio import CorporateAction
from .storage import write_tape

FAMILIES = (
    "no_signal",
    "known_signal",
    "delayed_cue",
    "signal_reversal",
    "regime_recurrence",
    "source_conflict",
    "volatility_shift",
    "execution_friction",
)
SPLITS = ("train", "validation", "audit")
PARENT = "adaptation-positive-analytic-v1"
DAY = 86_400_000_000
FIELDS = (
    ("source_a_signal", "signed_score"),
    ("source_b_signal", "signed_score"),
    ("source_a_reliability", "probability"),
    ("source_b_reliability", "probability"),
    ("market_return", "return"),
    ("market_volatility", "return"),
    ("cross_section_dispersion", "return"),
    ("us_treasury_2y", "percent"),
    ("us_treasury_10y", "percent"),
    ("us_curve_10y_2y", "percentage_points"),
    ("trading_enabled", "boolean"),
)
MARKET_FEATURE_INDICES = (4, 5, 6)


def cases(settings):
    """Asignar semillas únicas sin aceptar etiquetas de partición proporcionadas a mano."""
    if (
        set(settings)
        != {"schema_version", "families", "generator", "worlds", "seed_base", "final_test_opened"}
        or type(settings["schema_version"]) is not int
        or settings["schema_version"] != 1
        or settings["final_test_opened"] is not False
        or not isinstance(settings["families"], list)
        or not settings["families"]
        or any(name not in FAMILIES for name in settings["families"])
        or len(set(settings["families"])) != len(settings["families"])
    ):
        raise ValueError("La configuración no conserva las familias y el test cerrado")
    generator = settings["generator"]
    if set(generator) != {"assets", "sessions", "warmup_sessions", "regime_sessions"}:
        raise ValueError("Faltan las dimensiones del generador")
    if any(type(value) is not int for value in generator.values()) or not (
        1 <= generator["assets"] <= 128
        and 8 <= generator["sessions"] <= 364
        and 4 <= generator["warmup_sessions"] < generator["sessions"] - 2
        and 4 <= generator["regime_sessions"] <= 256
    ):
        raise ValueError("Las dimensiones del escenario exceden su presupuesto")
    counts = settings["worlds"]
    if set(counts) != set(SPLITS) or any(
        type(value) is not int or not 1 <= value <= 64 for value in counts.values()
    ):
        raise ValueError("Cada partición necesita entre uno y 64 mundos")
    seed = settings["seed_base"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1 - 7_200_063:
        raise ValueError("La semilla base no admite las semillas independientes del catálogo")
    return [
        dict(
            family=family,
            split=split,
            partition="train" if split == "train" else "validation",
            seed=seed + FAMILIES.index(family) * 1_000_000 + split_index * 100_000 + index,
            **generator,
        )
        for family in settings["families"]
        for split_index, split in enumerate(SPLITS)
        for index in range(counts[split])
    ]


def generate_scenario(case):
    """Generar un mundo sin consultar etiquetas futuras para construir observaciones."""
    family, split = case["family"], case["split"]
    dimensions = {
        key: case[key] for key in ("assets", "sessions", "warmup_sessions", "regime_sessions")
    }
    settings = dict(
        schema_version=1,
        families=[family],
        generator=dimensions,
        worlds=dict(train=1, validation=1, audit=1),
        seed_base=0,
        final_test_opened=False,
    )
    cases(settings)
    if split not in SPLITS or case["partition"] != ("train" if split == "train" else "validation"):
        raise ValueError("La partición declarada no corresponde al uso del mundo")
    if type(case["seed"]) is not int or not 0 <= case["seed"] < 2**32:
        raise ValueError("La semilla del mundo debe ser un entero de 32 bits")
    assets, count, warmup, duration = (
        dimensions[key] for key in ("assets", "sessions", "warmup_sessions", "regime_sessions")
    )
    rng = np.random.default_rng(case["seed"])
    jump_rng = np.random.default_rng(np.random.SeedSequence([case["seed"], 1]))
    delayed_signal = float(rng.choice([-1, 1]))
    cue_at = warmup // 2
    origin = 946_684_800_000_000 if split == "train" else 1_672_531_200_000_000
    close_times = origin + np.arange(count, dtype=np.int64) * DAY + 57_600_000_000
    open_times = close_times - 23_400_000_000
    prices = np.empty((count, assets, 5), dtype=np.float64)
    context = np.zeros((count, len(FIELDS)), dtype=np.float32)
    expected = np.zeros(count, dtype=np.float64)
    signals, relations = np.zeros(count), np.ones(count)
    regimes = np.zeros(count, dtype=np.int32)
    jumps = np.zeros(count, dtype=np.float64)
    realized = np.zeros(count)
    previous = np.linspace(90, 110, assets)
    asset_ids = [f"FIC{index:03d}" for index in range(assets)]
    events, actions = [], []
    for at in range(count):
        phase = max(0, (at - warmup) // duration)
        regime = phase % 2 if family == "regime_recurrence" else int(phase >= 1)
        regimes[at] = regime
        relation = -1 if family in {"signal_reversal", "regime_recurrence"} and regime else 1
        signal = float(rng.choice([-1, 1]))
        error_a, error_b = rng.random(2)
        reliability = (0.9, 0.1) if family == "source_conflict" else (1.0, 1.0)
        first = signal if error_a < reliability[0] else -signal
        second = signal if error_b < reliability[1] else -signal
        if family == "no_signal":
            first = second = 0.0
        if family == "delayed_cue":
            signal = delayed_signal
            first = delayed_signal if at == cue_at else 0.0
            second = 0.0
        signals[at], relations[at] = signal, relation
        active_signal = family != "no_signal" and at >= warmup
        if family == "delayed_cue":
            active_signal = warmup <= at < warmup + duration
        expected[at] = 0.006 * signal * relation if active_signal else 0.0
        sigma = 0.009 if family == "volatility_shift" and regime else 0.002
        idiosyncratic_ratio = 1.5 if family == "volatility_shift" and regime else 0.25
        jump_occurs, jump_sign = jump_rng.random(2)
        if family == "volatility_shift" and regime and jump_occurs < 0.02:
            jumps[at] = 0.04 if jump_sign >= 0.5 else -0.04
        returns = np.clip(
            (expected[at - 1] if at else 0)
            + rng.normal(0, sigma)
            + rng.normal(0, sigma * idiosyncratic_ratio, assets)
            + jumps[at],
            -0.08,
            0.08,
        )
        opening = previous.copy()
        if family == "execution_friction" and at in (warmup + duration, warmup + duration + 1):
            kind = "split" if at == warmup + duration else "dividend"
            amounts = np.full(assets, 2.0) if kind == "split" else opening * 0.002
            for asset, amount in zip(asset_ids, amounts, strict=True):
                identifier = f"{family}:{case['seed']}:{kind}:{asset}"
                actions.append(
                    CorporateAction(
                        identifier,
                        asset,
                        kind,
                        int(open_times[at]),
                        float(amount),
                        pay_at=int(close_times[at] + 2 * DAY) if kind == "dividend" else None,
                        verified=True,
                    )
                )
                events.append(
                    dict(
                        event_id=identifier,
                        session=at,
                        source_id="synthetic_ledger",
                        scope=asset,
                        unit="ratio" if kind == "split" else "USD",
                        value=float(amount),
                        reliability=1.0,
                        available_at=int(open_times[at]),
                        text=f"Operación ficticia {kind} de {asset} aplicada en la apertura.",
                    )
                )
            opening = opening / amounts if kind == "split" else opening - amounts
        closing = opening * (1 + returns)
        prices[at, :, 0] = opening
        prices[at, :, 1] = np.maximum(opening, closing) * 1.001
        prices[at, :, 2] = np.minimum(opening, closing) * 0.999
        prices[at, :, 3] = closing
        prices[at, :, 4] = 10 if family == "execution_friction" and regime else 100_000
        previous = closing
        realized[at] = returns.mean()
        volatility = realized[max(0, at - 15) : at + 1].std()
        rate2, rate10 = 3 + 10 * realized[at], 4 + 5 * realized[at]
        context[at] = (
            first,
            second,
            *reliability,
            realized[at],
            volatility,
            returns.std(),
            rate2,
            rate10,
            rate10 - rate2,
            float(at >= warmup),
        )
        for source, observed, probability in (
            ("source_a", first, reliability[0]),
            ("source_b", second, reliability[1]),
        ):
            if family == "delayed_cue" and (at != cue_at or source != "source_a"):
                continue
            events.append(
                dict(
                    event_id=f"{family}:{case['seed']}:{at}:{source}",
                    session=at,
                    source_id=source,
                    scope="market",
                    unit="signed_score",
                    value=observed,
                    reliability=probability,
                    available_at=int(close_times[at]),
                    text=f"Aviso ficticio de {source}: señal {observed:+.0f}.",
                )
            )
    identity = dict(
        generator=dict(case),
        analysis_domain="technical",
        real_corpus_compatible=False,
        modality_contract="technical_context_only",
        warmup_action=1,
        warmup_policy_loss=False,
        decision_start=warmup,
        simulated_macro=[name for name, _ in FIELDS[7:10]],
        corporate_action_provenance="synthetic_generator",
    )
    tape = MarketTape(
        prices,
        close_times,
        asset_ids,
        np.full((count, assets), 0.01),
        domain="synthetic",
        currency="USD",
        partition=case["partition"],
        parent_id=PARENT,
        source_identity=identity,
        open_times=open_times,
        actions=actions,
    )
    targets = np.full(count, np.nan)
    targets[:-1] = (prices[1:, :, 3] / prices[1:, :, 0] - 1).mean(axis=1)
    maturity = np.zeros(count, dtype=np.int64)
    maturity[:-1] = close_times[1:]
    return dict(
        tape=tape,
        fields=[dict(name=name, unit=unit) for name, unit in FIELDS],
        context=context,
        available_at=np.repeat(close_times[:, None], len(FIELDS), axis=1),
        events=events,
        truth=dict(
            session=np.arange(count, dtype=np.int32),
            latent_signal=signals,
            latent_regime=regimes,
            relation=relations,
            expected_return=expected,
            common_jump=jumps,
            target=targets,
            target_available_at=maturity,
        ),
    )


def _write_context(world, directory):
    sessions, fields = world["context"].shape
    table = pa.table(
        dict(
            session=pa.array(np.repeat(np.arange(sessions), fields), type=pa.int32()),
            feature=pa.array(np.tile(np.arange(fields), sessions), type=pa.int32()),
            value=pa.array(world["context"].reshape(-1), type=pa.float32()),
            present=pa.array(np.ones(sessions * fields, dtype=bool)),
            available_at=pa.array(world["available_at"].reshape(-1), type=pa.int64()),
        )
    )
    path = directory / "context.parquet"
    pq.write_table(table, path, compression="zstd", row_group_size=fields * 16)
    atomic_json(
        directory / "context.json",
        dict(
            schema_version=1,
            domain="synthetic",
            market_manifest_sha256=sha256(directory / "manifest.json"),
            fields=world["fields"],
            file=dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size),
        ),
    )


def fit_hmm(training_features):
    """Ajustar dos emisiones diagonales solo con secuencias declaradas de entrenamiento."""
    if not training_features or any(split != "train" for split, _ in training_features):
        raise ValueError("El HMM solo admite secuencias de entrenamiento")
    arrays = [np.asarray(values, dtype=np.float64) for _, values in training_features]
    if (
        any(
            array.ndim != 2 or array.shape[1] != 3 or len(array) < 2 or not np.isfinite(array).all()
            for array in arrays
        )
        or sum(map(len, arrays)) > 131072
    ):
        raise ValueError("Las observaciones del HMM exceden su contrato")
    data = np.concatenate(arrays)
    center, scale = data.mean(axis=0), data.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("Las emisiones necesitan variación observada en cada variable")
    from importlib.metadata import version

    from hmmlearn.hmm import GaussianHMM
    from threadpoolctl import threadpool_limits

    if version("hmmlearn") != "0.3.3":
        raise ValueError("El ajuste necesita hmmlearn==0.3.3 mediante uv")
    scale = np.maximum(scale, 1e-6)
    model = GaussianHMM(
        n_components=2, covariance_type="diag", n_iter=50, tol=1e-4, min_covar=1e-4, random_state=42
    )
    with threadpool_limits(limits=1):
        model.fit((data - center) / scale, lengths=[len(array) for array in arrays])
    means = model.means_ * scale + center
    variances = np.diagonal(model.covars_, axis1=1, axis2=2) * scale**2
    order = np.argsort(means[:, 1])
    parameters = dict(
        states=2,
        dimensions=3,
        prior=model.startprob_[order].tolist(),
        transitions=model.transmat_[order][:, order].reshape(-1).tolist(),
        means=means[order].reshape(-1).tolist(),
        variances=np.maximum(variances[order], 1e-12).reshape(-1).tolist(),
    )
    if not all(
        np.isfinite(values).all()
        for values in (
            parameters["prior"],
            parameters["transitions"],
            parameters["means"],
            parameters["variances"],
        )
    ):
        raise ValueError("El ajuste HMM produjo parámetros no finitos")
    return dict(
        schema_version=1,
        implementation="hmmlearn==0.3.3",
        fit_split="train",
        fit_scope="offline_train",
        parameters=parameters,
        feature_indices=list(MARKET_FEATURE_INDICES),
        observations=len(data),
        iterations=model.monitor_.iter,
        likelihood_history=list(model.monitor_.history),
        inference="forward_filter_only",
        training_context_point_in_time=False,
    )


def prepare_adaptation_scenarios(settings, output, *, fit_markov=False):
    """Confirmar un catálogo nuevo sin mezclar la verdad del evaluador con el contexto PPO."""
    planned = cases(settings)
    output = Path(output)
    safe_destination(output)
    if output.exists():
        raise ValueError("La salida ya existe y no se sobrescribe")
    output.mkdir(parents=True)
    index = dict(
        schema_version=1,
        kind="adaptation_scenarios",
        domain="synthetic",
        analysis_domain="technical",
        status="preparing",
        parent_id=PARENT,
        real_corpus_compatible=False,
        final_test_opened=False,
        settings=settings,
        macro_coverage=dict(
            simulated_concepts=3,
            catalog_concepts=140,
            simulated_ids=[name for name, _ in FIELDS[7:10]],
            other_concepts="absent",
        ),
        context_fields=[dict(name=name, unit=unit) for name, unit in FIELDS],
        market_feature_indices=list(MARKET_FEATURE_INDICES),
        contract=dict(
            warmup_action=1,
            warmup_policy_loss=False,
            warmup_counts_as_learning=False,
            truth_policy_access=False,
            audit_use="evaluator_only",
            modality_contract="technical_context_only",
        ),
        generator_source_sha256=sha256(Path(__file__)),
        records=[],
    )
    train_features = []
    atomic_json(output / "index.json", index)
    try:
        for case in planned:
            world = generate_scenario(case)
            name = f"{case['family']}-{case['split']}-{case['seed']}"
            directory = output / name
            write_tape(world["tape"], directory)
            _write_context(world, directory)
            pq.write_table(
                pa.Table.from_pylist(world["events"]),
                directory / "events.parquet",
                compression="zstd",
                row_group_size=256,
            )
            evaluator = directory / "evaluator"
            evaluator.mkdir()
            truth = pa.table(
                {key: pa.array(value, from_pandas=True) for key, value in world["truth"].items()}
            )
            pq.write_table(
                truth, evaluator / "truth.parquet", compression="zstd", row_group_size=256
            )
            index["records"].append(
                dict(
                    name=name,
                    path=name,
                    family=case["family"],
                    split=case["split"],
                    partition=case["partition"],
                    seed=case["seed"],
                    manifest_sha256=sha256(directory / "manifest.json"),
                    context_sha256=sha256(directory / "context.json"),
                    events_sha256=sha256(directory / "events.parquet"),
                    truth_sha256=sha256(evaluator / "truth.parquet"),
                    warmup_sessions=case["warmup_sessions"],
                    decision_start=case["warmup_sessions"],
                    evaluator_only=case["split"] == "audit",
                )
            )
            if fit_markov and case["split"] == "train":
                train_features.append(("train", world["context"][:, MARKET_FEATURE_INDICES].copy()))
            atomic_json(output / "index.json", index)
        if fit_markov:
            model = fit_hmm(train_features)
            model["train_manifest_sha256"] = [
                record["manifest_sha256"]
                for record in index["records"]
                if record["split"] == "train"
            ]
            atomic_json(output / "hmm.json", model)
            index["hmm"] = dict(path="hmm.json", sha256=sha256(output / "hmm.json"))
        index["status"] = "completed"
    except Exception:
        index["status"] = "failed"
        atomic_json(output / "index.json", index)
        raise
    atomic_json(output / "index.json", index)
    return index
