"""Contraejemplos del control de contextos, no resultados predictivos de S2."""

import numpy as np


def test_a_plain_maximum_cannot_distinguish_a_permutation_of_the_same_failure_contexts():
    contexts = np.eye(2)
    query = contexts[0]
    kernel = np.exp(-np.square(contexts - query).sum(1) / (2 * 0.25**2))
    permutation = np.array([1, 0])
    assert kernel.max() == kernel[permutation].max() == 1


def test_error_severity_can_make_the_association_control_distinguishable():
    contexts = np.eye(2)
    query = contexts[0]
    kernel = np.exp(-np.square(contexts - query).sum(1) / (2 * 0.25**2))
    errors, threshold = np.array([1.0, 4.0]), 0.5
    severity = 1 - threshold / errors
    original = (severity * kernel).max()
    shuffled = (severity * kernel[[1, 0]]).max()
    assert original == 0.5
    assert shuffled == 0.875
    # Una igualdad de severidades vuelve a hacer equivalente la permutación.
    uniform = np.ones(2)
    assert (uniform * kernel).max() == (uniform * kernel[[1, 0]]).max()
