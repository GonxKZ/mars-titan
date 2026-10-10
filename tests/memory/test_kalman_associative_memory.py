"""Regla kalman de B6 (PT3) contrastada con formas densas, RLS y el filtro escalar.

Las pruebas comparan una escritura con su ecuación en FP64 sobre tensores pequeños y no
recorren secuencias largas para que la memoria aprenda una señal. La regla no tiene
parámetros entrenables ni gradientes. Las reglas delta y proximal se fijan con las huellas
capturadas antes de añadir kalman.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    KalmanNoise,
    MatureCorrection,
    MatureFeedback,
)
from mars_titan.memory.mars_titan_variant import check_components, load_declaration

# Capturadas en 42e7dbca con la misma traza, antes de añadir la regla kalman.
LEGACY = dict(
    delta="3a8ec2df0d9e5e84463def40a443b34690c1c646e7d03c716530e483dbee1d8c",
    proximal="e26829200d77cf656ecb342bcec78216442b698dc8b12ba92c51766611580c8d",
    correction_identity="0759f11b3f245cf276a4c476e6c33538696408c36d4e3da8156308219a8d763f",
    variant_components="14b653b57020e9128c885a9aa16fd8b4a47b17f1516cf18b03b2700929074fe1",
)
NOISE = dict(process_noise=0.01, observation_noise=0.5, cohort_correlation=0.3, prior_variance=2.0)


def unit_keys(rows, size, generator):
    keys = torch.randn((rows, size), dtype=torch.float64, generator=generator)
    return keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)


def feedback(keys, values, *, start=1, available=None, weights=None):
    rows = keys.shape[0]
    ids = torch.arange(start, start + rows, dtype=torch.int64)
    available = ids * 10 if available is None else torch.tensor(available, dtype=torch.int64)
    return MatureFeedback(
        ids=ids,
        decision_at=available - 5,
        available_at=available,
        keys=keys,
        values=values,
        weights=weights,
    )


def kalman(size=4, values=1, **noise):
    return AssociativeMemoryConfig(
        "kalman", key_size=size, value_size=values, kalman=KalmanNoise(**(NOISE | noise))
    )


def random_state(config, seed):
    generator = torch.Generator().manual_seed(seed)
    size = config.key_size
    matrix = torch.randn((size, config.value_size), dtype=torch.float64, generator=generator)
    factor = torch.randn((size, size), dtype=torch.float64, generator=generator)
    covariance = factor @ factor.T / size + 0.5 * torch.eye(size, dtype=torch.float64)
    covariance = 0.5 * (covariance + covariance.T)
    return AssociativeMemory(config, matrix, covariance=covariance)


def dense_update(matrix, covariance, keys, values, noise, weights=None):
    """Filtro de Kalman en forma de covarianza con Σ densa, como referencia independiente."""
    rows, size = keys.shape
    prior = covariance + noise.process_noise * torch.eye(size, dtype=torch.float64)
    ones = torch.ones((rows, 1), dtype=torch.float64)
    sigma = noise.observation_noise * (
        (1 - noise.cohort_correlation) * torch.eye(rows, dtype=torch.float64)
        + noise.cohort_correlation * ones @ ones.T
    )
    if weights is not None:
        # Pesos de Huber como varianza inflada: Σ_w = W^{-1/2} Σ W^{-1/2}.
        inverse_root = torch.diag(weights.rsqrt())
        sigma = inverse_root @ sigma @ inverse_root
    gain = prior @ keys.T @ torch.linalg.inv(keys @ prior @ keys.T + sigma)
    updated = matrix + gain @ (values - keys @ matrix)
    joseph = torch.eye(size, dtype=torch.float64) - gain @ keys
    posterior = joseph @ prior @ joseph.T + gain @ sigma @ gain.T
    return updated, posterior


def legacy_trace():
    digest = {}
    generator = torch.Generator().manual_seed(4536)
    for rule, rate, forgetting in (("delta", 0.5, 0.1), ("proximal", 2.0, 0.05)):
        config = AssociativeMemoryConfig(
            rule, key_size=8, value_size=2, rate=rate, forgetting=forgetting
        )
        memory = AssociativeMemory(config)
        hasher = hashlib.sha256(config.fingerprint().encode())
        for event in range(4):
            rows = 5 + event
            keys = unit_keys(rows, 8, generator)
            values = torch.randn((rows, 2), dtype=torch.float64, generator=generator)
            ids = torch.arange(1 + 100 * event, 1 + 100 * event + rows, dtype=torch.int64)
            available = 1000 * (event + 1) + torch.arange(rows, dtype=torch.int64)
            weights = (
                torch.full((rows,), 1.0 / rows, dtype=torch.float64) if rule == "proximal" else None
            )
            memory = memory.write(
                MatureFeedback(
                    ids=ids,
                    decision_at=available - 3,
                    available_at=available,
                    keys=keys,
                    values=values,
                    weights=weights,
                ),
                cutoff=1000 * (event + 2),
            )
            hasher.update(memory.matrix.numpy().tobytes())
            hasher.update(memory.read(keys).numpy().tobytes())
        payload = memory.export()
        scalars = {key: value for key, value in payload.items() if key != "matrix"}
        hasher.update(json.dumps(scalars, sort_keys=True).encode())
        digest[rule] = hasher.hexdigest()
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", key_size=64, value_size=1, rate=1.0)
    )
    digest["correction_identity"] = hashlib.sha256(
        json.dumps(correction.identity(), sort_keys=True).encode()
    ).hexdigest()
    components, parsed = check_components(
        load_declaration(),
        {"associative_memory": {"rule": "delta", "key": "codec", "rate": 1, "forgetting": 0.1}},
    )
    digest["variant_components"] = hashlib.sha256(
        json.dumps([components, parsed.identity()], sort_keys=True).encode()
    ).hexdigest()
    return digest


def test_delta_and_proximal_keep_their_outputs_states_and_identities():
    assert legacy_trace() == LEGACY


def test_kalman_is_a_new_identity_without_rate_or_forgetting():
    config = kalman(size=64)
    identity = config.identity()
    assert identity["rule"] == "kalman" and "rate" not in identity and "forgetting" not in identity
    assert identity["kalman"] == dict(NOISE, huber_threshold=None)
    assert identity["weights"] is None and identity["source"] == "post_titans_pt3_v1"
    proximal = AssociativeMemoryConfig("proximal", key_size=64)
    assert "kalman" not in proximal.identity()
    others = {proximal.fingerprint(), kalman(size=64, cohort_correlation=0.0).fingerprint()}
    assert config.fingerprint() not in others
    huber = kalman(size=64, huber_threshold=3).identity()
    assert huber["kalman"]["huber_threshold"] == 3.0
    assert huber["weights"] == "huber_one_step_on_prior_standardized_residual"


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(rule="kalman"), "varianzas"),
        (dict(rule="proximal", kalman=KalmanNoise(**NOISE)), "varianzas"),
        (dict(rule="kalman", rate=0.5, kalman=KalmanNoise(**NOISE)), "rate ni forgetting"),
        (dict(rule="kalman", forgetting=0.1, kalman=KalmanNoise(**NOISE)), "rate ni forgetting"),
    ],
)
def test_invalid_rule_declarations_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        AssociativeMemoryConfig(**options)


@pytest.mark.parametrize(
    "change",
    [
        dict(observation_noise=0.0),
        dict(cohort_correlation=1.0),
        dict(cohort_correlation=-0.1),
        dict(prior_variance=0.0),
        dict(process_noise=-1e-6),
        dict(process_noise=float("nan")),
        dict(huber_threshold=0.0),
    ],
)
def test_impossible_noise_is_rejected(change):
    with pytest.raises(ValueError):
        KalmanNoise(**(NOISE | change))


def test_closed_form_inverse_of_the_equicorrelated_noise():
    sigma2, rho, rows = 0.7, 0.35, 6
    ones = torch.ones((rows, 1), dtype=torch.float64)
    noise = sigma2 * ((1 - rho) * torch.eye(rows, dtype=torch.float64) + rho * ones @ ones.T)
    shrink = rho / (1 - rho + rows * rho)
    closed = (torch.eye(rows, dtype=torch.float64) - shrink * ones @ ones.T) / (sigma2 * (1 - rho))
    torch.testing.assert_close(closed @ noise, torch.eye(rows, dtype=torch.float64))


@pytest.mark.parametrize("huber", [None, 2.0])
def test_information_update_matches_the_dense_covariance_form(huber):
    config = kalman(size=4, values=2, huber_threshold=huber)
    memory = random_state(config, 7)
    generator = torch.Generator().manual_seed(8)
    keys = unit_keys(6, 4, generator)
    values = torch.randn((6, 2), dtype=torch.float64, generator=generator)
    # Una fila atípica, la única que el recorte de Huber debe reponderar.
    values[0] += 15.0
    written = memory.write(feedback(keys, values), cutoff=100)
    weights = None
    if huber is not None:
        noise = config.kalman
        prior = memory.covariance + noise.process_noise * torch.eye(4, dtype=torch.float64)
        spread = (((keys @ prior) * keys).sum(dim=1) + noise.observation_noise).sqrt()
        score = torch.linalg.vector_norm(values - keys @ memory.matrix, dim=1) / spread
        weights = torch.clamp(huber / score, max=1.0)
        # La prueba solo distingue el caso ponderado si alguna fila supera el umbral.
        assert (weights < 1).any() and (weights == 1).any()
    matrix, covariance = dense_update(
        memory.matrix, memory.covariance, keys, values, config.kalman, weights
    )
    torch.testing.assert_close(written.matrix, matrix, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(written.covariance, covariance, rtol=1e-10, atol=1e-12)
    assert torch.equal(written.covariance, written.covariance.T)
    assert written.writes == 6 and memory.writes == 0


def test_a_large_huber_threshold_is_the_unweighted_rule_bit_for_bit():
    generator = torch.Generator().manual_seed(9)
    keys = unit_keys(5, 4, generator)
    values = torch.randn((5, 1), dtype=torch.float64, generator=generator)
    plain = random_state(kalman(), 10).write(feedback(keys, values), cutoff=100)
    wide = AssociativeMemory(
        kalman(huber_threshold=1e6),
        random_state(kalman(), 10).matrix,
        covariance=random_state(kalman(), 10).covariance,
    ).write(feedback(keys, values), cutoff=100)
    assert torch.equal(plain.matrix, wide.matrix)
    assert torch.equal(plain.covariance, wide.covariance)


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.9])
def test_one_outcome_per_cohort_is_the_scalar_kalman_filter(rho):
    config = kalman(size=1, cohort_correlation=rho)
    memory = AssociativeMemory(config)
    estimate, variance = 0.0, NOISE["prior_variance"]
    for step, label in enumerate((1.5, -0.5, 2.0, 0.25), start=1):
        memory = memory.write(
            feedback(
                torch.ones((1, 1), dtype=torch.float64),
                torch.tensor([[label]], dtype=torch.float64),
                start=step,
            ),
            cutoff=100,
        )
        prior = variance + NOISE["process_noise"]
        gain = prior / (prior + NOISE["observation_noise"])
        estimate, variance = estimate + gain * (label - estimate), (1 - gain) * prior
        assert memory.matrix.item() == pytest.approx(estimate, rel=1e-13)
        assert memory.covariance.item() == pytest.approx(variance, rel=1e-13)


def test_zero_correlation_and_process_noise_is_block_rls_with_its_prior():
    config = kalman(size=3, values=2, process_noise=0.0, cohort_correlation=0.0)
    generator = torch.Generator().manual_seed(11)
    cohorts = [
        (unit_keys(rows, 3, generator), torch.randn((rows, 2), dtype=torch.float64))
        for rows in (4, 5)
    ]
    memory, start = AssociativeMemory(config), 1
    for keys, values in cohorts:
        memory = memory.write(feedback(keys, values, start=start), cutoff=1000)
        start += len(keys)
    # Con q = 0 y ϱ = 0, dos cohortes seguidas dan la posterior de una regresión ridge con
    # todas las filas a la vez, λ = σ²/p0.
    keys = torch.cat([keys for keys, _ in cohorts])
    values = torch.cat([values for _, values in cohorts])
    precision = torch.eye(3, dtype=torch.float64) / NOISE["prior_variance"] + keys.T @ keys / 0.5
    torch.testing.assert_close(memory.covariance, torch.linalg.inv(precision))
    torch.testing.assert_close(memory.matrix, torch.linalg.solve(precision, keys.T @ values / 0.5))


@pytest.mark.parametrize("rows", [1, 4, 64, 4096])
def test_cohort_correlation_limits_the_information_of_identical_keys(rows):
    rho, sigma2 = 0.2, 0.5
    config = kalman(size=2, process_noise=0.0, cohort_correlation=rho)
    key = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    keys = key.repeat(rows, 1)
    memory = AssociativeMemory(config).write(
        feedback(keys, torch.ones((rows, 1), dtype=torch.float64)), cutoff=10 * rows + 10
    )
    gained = torch.linalg.inv(memory.covariance) - torch.eye(2, dtype=torch.float64) / 2.0
    # Información n/(σ²(1 + (n − 1)ϱ)) en la dirección de k, como mucho 1/(σ²ϱ).
    expected = rows / (sigma2 * (1 + (rows - 1) * rho)) * key.T @ key
    torch.testing.assert_close(gained, expected, rtol=1e-9, atol=1e-9)
    assert float(key @ gained @ key.T) <= 1 / (sigma2 * rho) + 1e-9


def test_covariance_grows_at_most_linearly_in_unexcited_directions():
    config = kalman(size=3, process_noise=0.05)
    generator = torch.Generator().manual_seed(12)
    memory = AssociativeMemory(config)
    excited = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    for cohort in range(1, 41):
        rows = 3
        values = torch.randn((rows, 1), dtype=torch.float64, generator=generator)
        memory = memory.write(
            feedback(excited.repeat(rows, 1), values, start=rows * cohort), cutoff=10_000
        )
        covariance = memory.covariance
        assert torch.equal(covariance, covariance.T)
        eigenvalues = torch.linalg.eigvalsh(covariance)
        assert eigenvalues.min() > 0
        # P⁺ ⪯ P + qI, así que λ_max(P) ≤ p0 + q · cohortes. Sin claves en una dirección,
        # la cota se alcanza: es el crecimiento declarado de la regla.
        assert eigenvalues.max() <= 2.0 + 0.05 * cohort + 1e-12
        assert covariance[2, 2].item() == pytest.approx(2.0 + 0.05 * cohort, rel=1e-12)
    assert covariance[0, 0] < 0.5


def test_write_order_and_read_purity():
    config = kalman(size=4)
    generator = torch.Generator().manual_seed(13)
    keys = unit_keys(5, 4, generator)
    values = torch.randn((5, 1), dtype=torch.float64, generator=generator)
    available = [50, 20, 40, 10, 30]
    memory = random_state(config, 14)
    first = memory.write(feedback(keys, values, available=available), cutoff=100)
    order = torch.tensor([3, 1, 4, 2, 0])
    second = memory.write(
        MatureFeedback(
            ids=torch.arange(1, 6, dtype=torch.int64)[order],
            decision_at=torch.tensor(available)[order] - 5,
            available_at=torch.tensor(available)[order],
            keys=keys[order],
            values=values[order],
        ),
        cutoff=100,
    )
    assert torch.equal(first.matrix, second.matrix)
    assert torch.equal(first.covariance, second.covariance)
    before = (first.matrix, first.covariance)
    reads = first.read(keys)
    variance = first.variance(keys)
    torch.testing.assert_close(reads, keys @ before[0], rtol=0, atol=0)
    expected = torch.einsum("bi,ij,bj->b", keys, before[1], keys) + NOISE["observation_noise"]
    torch.testing.assert_close(variance, expected, rtol=1e-14, atol=0)
    assert torch.equal(first.matrix, before[0]) and torch.equal(first.covariance, before[1])
    with pytest.raises(ValueError, match="Solo la regla kalman"):
        AssociativeMemory(AssociativeMemoryConfig("proximal", key_size=4)).variance(keys)


def test_immature_repeated_or_weighted_cohorts_are_rejected_without_changing_the_state():
    config = kalman(size=3)
    generator = torch.Generator().manual_seed(15)
    keys = unit_keys(2, 3, generator)
    values = torch.ones((2, 1), dtype=torch.float64)
    memory = random_state(config, 16).write(feedback(keys, values), cutoff=20)
    before = memory.export()
    for fault, message in (
        (lambda: memory.write(feedback(keys, values, start=3), cutoff=35), "madurar"),
        (lambda: memory.write(feedback(keys, values), cutoff=40), "ya se aplicó"),
        (
            lambda: memory.write(
                feedback(keys, values, start=3, weights=torch.full((2,), 0.5)), cutoff=60
            ),
            "pesos",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            fault()
        after = memory.export()
        assert after["matrix_sha256"] == before["matrix_sha256"]
        assert after["covariance_sha256"] == before["covariance_sha256"]
    # Una cohorte vacía no aplica el paseo aleatorio.
    empty = memory.write(
        feedback(
            torch.empty((0, 3), dtype=torch.float64), torch.empty((0, 1), dtype=torch.float64)
        ),
        cutoff=99,
    )
    assert empty is memory


def test_labels_of_a_later_cohort_do_not_change_earlier_reads():
    config = kalman(size=3)
    generator = torch.Generator().manual_seed(17)
    first_keys, later_keys, query = (unit_keys(3, 3, generator) for _ in range(3))
    first = AssociativeMemory(config).write(
        feedback(first_keys, torch.ones((3, 1), dtype=torch.float64)), cutoff=30
    )
    reads = []
    for shift in (0.0, 100.0):
        later = first.write(
            feedback(later_keys, torch.full((3, 1), shift, dtype=torch.float64), start=4),
            cutoff=60,
        )
        reads.append((first.read(query), first.variance(query), later.read(query)))
    assert torch.equal(reads[0][0], reads[1][0]) and torch.equal(reads[0][1], reads[1][1])
    assert not torch.equal(reads[0][2], reads[1][2])


def test_recovery_restores_matrix_covariance_and_cursor_exactly():
    config = kalman(size=3, values=2, huber_threshold=2.0)
    generator = torch.Generator().manual_seed(18)
    cohorts = [
        (unit_keys(4, 3, generator), torch.randn((4, 2), dtype=torch.float64, generator=generator))
        for _ in range(3)
    ]

    def run(restore_after=None):
        memory = AssociativeMemory(config)
        for index, (keys, values) in enumerate(cohorts):
            memory = memory.write(feedback(keys, values, start=1 + 4 * index), cutoff=1000)
            if restore_after == index:
                memory = AssociativeMemory.restore(config, memory.export())
        return memory

    straight, resumed = run(), run(restore_after=0)
    assert torch.equal(straight.matrix, resumed.matrix)
    assert torch.equal(straight.covariance, resumed.covariance)
    assert (straight.writes, straight.cursor) == (resumed.writes, resumed.cursor)
    payload = straight.export()
    tampered = dict(payload, covariance=payload["covariance"] * (1 + 1e-12))
    with pytest.raises(ValueError, match="huella"):
        AssociativeMemory.restore(config, tampered)
    skewed = payload["covariance"].clone()
    skewed[0, 1] += 1e-9
    with pytest.raises(ValueError, match="simétrica"):
        AssociativeMemory.restore(config, dict(payload, covariance=skewed))
    with pytest.raises(ValueError, match="campos"):
        AssociativeMemory.restore(
            config, {k: v for k, v in payload.items() if not k.startswith("covariance")}
        )
    with pytest.raises(ValueError, match="otra configuración"):
        AssociativeMemory.restore(kalman(size=3, values=2), payload)
    with pytest.raises(ValueError, match="Solo la regla kalman"):
        AssociativeMemory(AssociativeMemoryConfig("delta", key_size=3), covariance=torch.eye(3))


def test_mature_correction_and_variant_accept_the_kalman_rule():
    noise = dict(NOISE, huber_threshold=None)
    components, correction = check_components(
        load_declaration(),
        {"associative_memory": {"rule": "kalman", "key": "codec", **noise}},
    )
    assert components["associative_memory"] == {"rule": "kalman", "key": "codec", **noise}
    assert correction.memory.kalman == KalmanNoise(**NOISE)
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn((5, 64), dtype=torch.float32, generator=generator)
    rows = correction.feedback(
        ids=[1, 2, 3, 4, 5],
        decision_at=[1, 2, 3, 4, 5],
        available_at=[6, 7, 8, 9, 10],
        keys=correction.keys(inputs),
        values=[0.1, -0.2, 0.05, 0.3, -0.1],
    )
    assert rows.weights is None
    memory = AssociativeMemory(correction.memory).write(rows, cutoff=10)
    # Con d = 64 el producto TᵀT de BLAS no sale simétrico bit a bit. La escritura debe
    # simetrizarlo para que la covarianza exportada se pueda restaurar.
    assert memory.writes == 5 and torch.equal(memory.covariance, memory.covariance.T)
    restored = AssociativeMemory.restore(correction.memory, memory.export())
    assert torch.equal(restored.covariance, memory.covariance)
    with pytest.raises(ValueError, match="cinco varianzas"):
        check_components(
            load_declaration(),
            {"associative_memory": {"rule": "kalman", "key": "codec", "rate": 1.0, **noise}},
        )


def test_comparison_arm_is_declared_with_its_control_and_rule_but_not_executed():
    root = Path(__file__).resolve().parents[2]
    declaration = json.loads(
        (root / "configs/evaluation/kalman-associative-comparison.json").read_text()
    )
    assert declaration["status"] == "declared_not_executed"
    assert declaration["executions"] == 0 and declaration["final_test_opened"] is False
    roles = {name: arm["role"] for name, arm in declaration["arms"].items()}
    assert sorted(roles.values()) == ["control", "control", "innovation", "trivial_alternative"]
    extensions = load_declaration()
    allowed = extensions["components"]["associative_memory"]["allowed"]
    for arm in declaration["arms"].values():
        value = arm["associative_memory"]
        # Las varianzas y tasas se estiman al ejecutar, así que aquí solo se fija la regla.
        assert value is None or (value["rule"] in allowed and value["key"] == "codec")
    innovation = next(a for a in declaration["arms"].values() if a["role"] == "innovation")
    trivial = next(a for a in declaration["arms"].values() if a["role"] == "trivial_alternative")
    assert innovation["associative_memory"]["rule"] == "kalman"
    assert trivial["associative_memory"]["cohort_correlation"] == 0.0
    ablation = next(a for a in extensions["ablations"] if a["id"] == "A12")
    assert ablation["change"] == {"associative_memory": ["kalman"]}
