"""Estado rápido por bloques de `FlowStates`: mismos valores y gradientes que fila a fila.

La versión anterior separaba cada bloque en un estado por fila y lo volvía a apilar. Aquí
se contrasta `gather` con esa misma construcción explícita, en FP64 y sin pasos.
"""

import pytest
import torch

from mars_titan.models.titans.financial import FinancialState
from mars_titan.models.titans.state import MACState, NeuralMemoryState
from mars_titan.training.financial_run import FlowStates


def block(flows, *, offset=0.0, steps=None, grad=True):
    rows = len(flows)
    base = torch.arange(rows * 6, dtype=torch.float64).reshape(rows, 2, 3) + offset
    weights = (base.clone().requires_grad_(grad), (base * 2).clone().requires_grad_(grad))
    momentum = ((base / 3).clone().requires_grad_(grad),)
    counts = torch.tensor(steps or [index + 1 for index in range(rows)], dtype=torch.int64)
    memory = NeuralMemoryState(weights, momentum, counts.clone(), "memory")
    return FinancialState(
        "config",
        "parameters",
        tuple(flows),
        tuple(f"sample-{flow}" for flow in flows),
        tuple(100 + index for index in range(rows)),
        counts,
        MACState(memory, "mac"),
    )


def rows_of(state, flows):
    """Construcción anterior: un estado por fila y su apilado en el orden pedido."""
    index = {flow: row for row, flow in enumerate(state.flow_ids)}
    picked = [index[flow] for flow in flows]
    memory = state.mac.memory

    def stack(values):
        return torch.stack([values[row] for row in picked])

    return (
        tuple(stack(w) for w in memory.weights),
        tuple(stack(m) for m in memory.momentum),
        stack(memory.steps),
        torch.tensor([state.observed_steps[row].item() for row in picked]),
    )


def tensors(state):
    memory = state.mac.memory
    return (*memory.weights, *memory.momentum, memory.steps, state.observed_steps)


def test_whole_block_in_order_is_returned_without_copies():
    states = FlowStates()
    state = block(["a", "b", "c"])
    states.put(state)
    assert states.gather(["a", "b", "c"]) is state
    assert states.gather(["c", "b", "a"]) is not state
    assert states.gather(["a", "b"]) is not state


@pytest.mark.parametrize("flows", [["b"], ["c", "a"], ["b", "c"], ["c", "b", "a"], ["a", "c"]])
def test_gather_matches_the_row_by_row_stack(flows):
    states = FlowStates()
    state = block(["a", "b", "c"])
    states.put(state)
    gathered = states.gather(flows)
    weights, momentum, steps, observed = rows_of(state, flows)
    for got, expected in zip(
        tensors(gathered), (*weights, *momentum, steps, observed), strict=True
    ):
        assert torch.equal(got, expected)
    assert gathered.flow_ids == tuple(flows)
    assert gathered.last_sample_ids == tuple(f"sample-{flow}" for flow in flows)
    order = {"a": 0, "b": 1, "c": 2}
    assert gathered.last_prediction_at == tuple(100 + order[flow] for flow in flows)
    assert [states.steps(flow) for flow in flows] == observed.tolist()


def test_gradients_match_the_row_by_row_stack_across_blocks():
    left, right = block(["a", "b"]), block(["c", "d", "e"], offset=50.0)
    states = FlowStates()
    states.put(left)
    states.put(right)
    flows = ["d", "a", "e", "b"]
    gathered = states.gather(flows)
    weight = torch.linspace(-1, 1, 4 * 6, dtype=torch.float64).reshape(4, 2, 3)
    (gathered.mac.memory.weights[0] * weight).sum().backward()
    new = (left.mac.memory.weights[0].grad.clone(), right.mac.memory.weights[0].grad.clone())
    for value in (*left.mac.memory.weights, *right.mac.memory.weights):
        value.grad = None
    stacked = torch.cat(
        [
            torch.stack([left.mac.memory.weights[0][row] for row in (0, 1)]),
            torch.stack([right.mac.memory.weights[0][row] for row in (0, 1, 2)]),
        ]
    )
    reference = stacked[[3, 0, 4, 1]]
    (reference * weight).sum().backward()
    assert torch.equal(new[0], left.mac.memory.weights[0].grad)
    assert torch.equal(new[1], right.mac.memory.weights[0].grad)


def test_later_blocks_replace_only_their_flows():
    states = FlowStates()
    first, second = block(["a", "b", "c"]), block(["b"], offset=10.0, steps=[7])
    states.put(first)
    states.put(second)
    assert list(states) == ["a", "b", "c"] and len(states) == 3 and "b" in states
    assert states.steps("b") == 7 and states.handle("b")[1] == 0
    assert states.gather(["b"]) is second
    assert len(states.values()) == 2
    assert torch.equal(
        states.gather(["a", "c"]).mac.memory.weights[0], first.mac.memory.weights[0][[0, 2]]
    )
    subset = states.subset(["c"])
    assert list(subset) == ["c"] and subset.gather(["c"]).observed_steps.tolist() == [3]
    states.clear()
    assert not states


def test_detach_keeps_one_block_per_source_and_seals_the_parameters():
    states = FlowStates()
    states.put(block(["a", "b"]))
    states.put(block(["c"], offset=5.0))
    states.detach("sealed")
    values = states.values()
    assert len(values) == 2
    for value in values:
        assert value.parameter_id == "sealed"
        assert all(
            not w.requires_grad for w in (*value.mac.memory.weights, *value.mac.memory.momentum)
        )
    assert states.gather(["a", "b"]) is values[0]
    states.put(block(["d"]), detach=True)
    assert not states.gather(["d"]).mac.memory.weights[0].requires_grad
