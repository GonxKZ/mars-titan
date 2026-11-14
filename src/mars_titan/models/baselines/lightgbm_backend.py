"""Selección explícita de OpenCL en doble precisión para el control LightGBM."""

import ctypes as ct
import ctypes.util


def checked_execution(execution):
    """Un perfil acotado. No permite sustituir una GPU ausente por CPU."""
    if not (
        isinstance(execution, dict)
        and set(execution) == {"backend", "platform_id", "device_id"}
        and execution["backend"] == "opencl_float64"
        and all(
            type(execution[k]) is int and 0 <= execution[k] <= 15
            for k in ("platform_id", "device_id")
        )
    ):
        raise ValueError("LightGBM necesita un perfil OpenCL float64 y dispositivos explícitos")
    return execution


def parameters(execution):
    """Cambios del ejecutor sobre los parámetros científicos del control."""
    checked_execution(execution)
    return dict(
        device_type="gpu",
        gpu_use_dp=True,
        deterministic=False,
        gpu_platform_id=execution["platform_id"],
        gpu_device_id=execution["device_id"],
    )


def check_gpu_backend(execution):
    """Consultar el dispositivo del controlador sin construir datos ni ajustar árboles."""
    checked_execution(execution)
    try:
        library = ct.CDLL(ctypes.util.find_library("OpenCL") or "libOpenCL.so.1")
    except OSError as error:
        raise RuntimeError("No está disponible el controlador OpenCL") from error
    pointer = ct.c_void_p
    library.clGetPlatformIDs.argtypes = [ct.c_uint, ct.POINTER(pointer), ct.POINTER(ct.c_uint)]
    library.clGetDeviceIDs.argtypes = [
        pointer,
        ct.c_ulong,
        ct.c_uint,
        ct.POINTER(pointer),
        ct.POINTER(ct.c_uint),
    ]
    library.clGetDeviceInfo.argtypes = [
        pointer,
        ct.c_uint,
        ct.c_size_t,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]

    def checked(code):
        if code != 0:
            raise RuntimeError(f"OpenCL no pudo consultar el dispositivo, código {code}")

    count = ct.c_uint()
    checked(library.clGetPlatformIDs(0, None, ct.byref(count)))
    if not execution["platform_id"] < count.value <= 16:
        raise RuntimeError("No existe la plataforma OpenCL solicitada")
    platforms = (pointer * count.value)()
    checked(library.clGetPlatformIDs(count, platforms, None))
    platform = platforms[execution["platform_id"]]
    checked(library.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None, ct.byref(count)))
    if not execution["device_id"] < count.value <= 16:
        raise RuntimeError("No existe el dispositivo OpenCL solicitado")
    devices = (pointer * count.value)()
    checked(library.clGetDeviceIDs(platform, 0xFFFFFFFF, count, devices, None))
    device = devices[execution["device_id"]]

    def integer(field):
        value = ct.c_ulonglong()
        checked(library.clGetDeviceInfo(device, field, ct.sizeof(value), ct.byref(value), None))
        return value.value

    def text(field):
        value = ct.create_string_buffer(4096)
        checked(library.clGetDeviceInfo(device, field, ct.sizeof(value), value, None))
        return value.value.decode("utf-8", errors="replace").strip()

    name, vendor = text(0x102B), text(0x102C)
    if not integer(0x1000) & 4 or "NVIDIA" not in vendor:
        raise RuntimeError("El dispositivo OpenCL solicitado no es una GPU NVIDIA")
    if not integer(0x1032):
        raise RuntimeError("La GPU OpenCL no declara operaciones en doble precisión")
    return dict(
        execution,
        name=name,
        vendor=vendor,
        driver=text(0x102D),
        global_memory_bytes=integer(0x101F),
        accumulation="float64",
    )
