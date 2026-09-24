"""Replay circular de estados financieros, sin copiar las modalidades originales."""

import torch


class Replay:
    def __init__(self, capacity, width, *, max_bytes=128 * 1024**2):
        if (
            type(capacity) is not int
            or not 1 <= capacity <= 65536
            or type(width) is not int
            or not 1 <= width <= 32768
            or capacity * (width * 8 + 13) > max_bytes
        ):
            raise ValueError("El replay supera su presupuesto")
        self.capacity, self.width = capacity, width
        self.position, self.size = 0, 0
        self.data = dict(
            observation=torch.zeros(capacity, width, dtype=torch.float32),
            following=torch.zeros(capacity, width, dtype=torch.float32),
            action=torch.zeros(capacity, dtype=torch.int64),
            reward=torch.zeros(capacity, dtype=torch.float32),
            terminated=torch.zeros(capacity, dtype=torch.bool),
        )

    def add(self, observation, following, action, reward, terminated):
        for name, value in zip(
            self.data, (observation, following, action, reward, terminated), strict=True
        ):
            self.data[name][self.position] = torch.as_tensor(value)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, size, rng, device):
        if not self.size or not 1 <= size <= self.capacity:
            raise ValueError("El replay no tiene un lote disponible")
        indices = torch.from_numpy(rng.integers(0, self.size, size=size))
        return {key: value[indices].to(device) for key, value in self.data.items()}

    def snapshot(self):
        return dict(
            position=self.position,
            size=self.size,
            capacity=self.capacity,
            width=self.width,
            data={key: value[: self.size].clone() for key, value in self.data.items()},
        )

    def restore(self, state):
        if (
            state["capacity"] != self.capacity
            or state["width"] != self.width
            or type(state["size"]) is not int
            or not 0 <= state["size"] <= self.capacity
            or type(state["position"]) is not int
            or not 0 <= state["position"] < self.capacity
            or (state["size"] < self.capacity and state["position"] != state["size"])
            or set(state["data"]) != set(self.data)
        ):
            raise ValueError("El replay guardado no conserva su identidad o cursor")
        for key, value in state["data"].items():
            if (
                value.shape != self.data[key][: state["size"]].shape
                or value.dtype != self.data[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("El replay contiene dimensiones o valores inválidos")
        if ((state["data"]["action"] < 0) | (state["data"]["action"] >= 6)).any():
            raise ValueError("El replay contiene una acción fuera del contrato financiero")
        for key, value in state["data"].items():
            self.data[key][: state["size"]].copy_(value)
        self.position, self.size = state["position"], state["size"]
