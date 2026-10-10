"""Lecturas del equipo para el observatorio en directo, sin cambiar nada del sistema.

Las medidas proceden de `/proc`, de `statvfs` y de consultas de lectura de NVML. No se
fijan relojes, perfiles de energía ni límites. Una fuente que falla se publica como
ausencia (NaN en la serie y null en JSON), nunca como cero.
"""

import ctypes
import math
import os
import time
from array import array

FIELDS = (
    "gpu_temp_c",
    "gpu_util_pct",
    "gpu_mem_used_mib",
    "gpu_mem_total_mib",
    "gpu_power_w",
    "gpu_sm_clock_mhz",
    "gpu_events",
    "cpu_util_pct",
    "load_1m",
    "ram_used_mib",
    "ram_total_mib",
    "swap_used_mib",
    "disk_free_gib",
    "server_cpu_pct",
)
# Motivos de reducción de reloj que NVML declara (nvmlClocksEventReasons).
GPU_EVENTS = {
    0x1: "gpu_idle",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake",
}


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    """Consultas de lectura de la primera GPU. Si NVML no está disponible, no hay GPU."""

    def __init__(self, library="libnvidia-ml.so.1"):
        self.handle, self.name = None, None
        try:
            self.lib = ctypes.CDLL(library)
            if self.lib.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return
            buffer = ctypes.create_string_buffer(96)
            if self.lib.nvmlDeviceGetName(handle, buffer, 96) == 0:
                self.name = buffer.value.decode(errors="replace")[:96]
            self.handle = handle
        except OSError:
            self.lib = None

    def _read(self, function, value, *args):
        status = getattr(self.lib, function)(self.handle, *args, ctypes.byref(value))
        return value if status == 0 else None

    def sample(self):
        values = dict.fromkeys(FIELDS[:7], math.nan)
        if self.handle is None:
            return values
        temp = self._read("nvmlDeviceGetTemperature", ctypes.c_uint(), 0)
        memory = self._read("nvmlDeviceGetMemoryInfo", _Memory())
        utilization = self._read("nvmlDeviceGetUtilizationRates", _Utilization())
        power = self._read("nvmlDeviceGetPowerUsage", ctypes.c_uint())
        clock = self._read("nvmlDeviceGetClockInfo", ctypes.c_uint(), 1)
        events = None
        for name in (
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ):
            if hasattr(self.lib, name):
                events = self._read(name, ctypes.c_ulonglong())
                break
        if temp is not None:
            values["gpu_temp_c"] = float(temp.value)
        if memory is not None:
            values["gpu_mem_used_mib"] = memory.used / 2**20
            values["gpu_mem_total_mib"] = memory.total / 2**20
        if utilization is not None:
            values["gpu_util_pct"] = float(utilization.gpu)
        if power is not None:
            values["gpu_power_w"] = power.value / 1000
        if clock is not None:
            values["gpu_sm_clock_mhz"] = float(clock.value)
        if events is not None:
            values["gpu_events"] = float(events.value & sum(GPU_EVENTS))
        return values


def _meminfo(path="/proc/meminfo"):
    fields = {}
    with open(path) as stream:
        for line in stream:
            name, _, rest = line.partition(":")
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                fields[name] = int(rest.split()[0]) / 1024
    return fields


def _cpu_times(path="/proc/stat"):
    with open(path) as stream:
        values = [int(value) for value in stream.readline().split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


class SystemProbe:
    """Muestra puntual del equipo. Los porcentajes de CPU comparan dos lecturas seguidas."""

    def __init__(self, disk_path, *, nvml=None, clock=time.monotonic):
        self.disk_path = os.fspath(disk_path)
        self.nvml = nvml if nvml is not None else Nvml()
        self.clock = clock
        self._cpu = None
        self._process = None

    def sample(self):
        values = self.nvml.sample()
        try:
            total, idle = _cpu_times()
            if self._cpu is not None and total > self._cpu[0]:
                busy = (total - self._cpu[0]) - (idle - self._cpu[1])
                values["cpu_util_pct"] = 100 * busy / (total - self._cpu[0])
            self._cpu = total, idle
        except (OSError, ValueError, IndexError):
            pass
        try:
            values["load_1m"] = os.getloadavg()[0]
        except OSError:
            pass
        try:
            memory = _meminfo()
            values["ram_total_mib"] = memory["MemTotal"]
            values["ram_used_mib"] = memory["MemTotal"] - memory["MemAvailable"]
            values["swap_used_mib"] = memory["SwapTotal"] - memory["SwapFree"]
        except (OSError, KeyError, ValueError):
            pass
        try:
            disk = os.statvfs(self.disk_path)
            values["disk_free_gib"] = disk.f_bavail * disk.f_frsize / 2**30
        except OSError:
            pass
        now, times = self.clock(), os.times()
        spent = times.user + times.system
        if self._process is not None and now > self._process[0]:
            values["server_cpu_pct"] = 100 * (spent - self._process[1]) / (now - self._process[0])
        self._process = now, spent
        return {name: float(values.get(name, math.nan)) for name in FIELDS}


class TelemetryRing:
    """Últimas muestras en columnas de tamaño fijo. La memoria no crece con el tiempo."""

    def __init__(self, capacity):
        if not 1 <= capacity <= 1_000_000:
            raise ValueError("La capacidad de la telemetría debe estar entre 1 y 1.000.000")
        self.capacity = capacity
        self.columns = {name: array("d", [math.nan]) * capacity for name in ("t", *FIELDS)}
        self.start = self.size = 0

    def append(self, timestamp, values):
        index = (self.start + self.size) % self.capacity
        if self.size == self.capacity:
            self.start = (self.start + 1) % self.capacity
        else:
            self.size += 1
        self.columns["t"][index] = timestamp
        for name in FIELDS:
            self.columns[name][index] = values.get(name, math.nan)

    def latest(self, limit):
        """Columnas cronológicas de las últimas `limit` muestras, con null en las ausencias."""
        count = min(limit, self.size)
        first = self.start + self.size - count
        order = [(first + offset) % self.capacity for offset in range(count)]

        def clean(value):
            return None if math.isnan(value) else round(value, 4)

        return {name: [clean(column[i]) for i in order] for name, column in self.columns.items()}
