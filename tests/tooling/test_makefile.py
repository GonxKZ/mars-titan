"""El punto de entrada de órdenes no puede ejecutar pasos de optimizador nativos."""

import re
from pathlib import Path

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"


def test_every_ctest_call_excludes_optimizer_steps_and_cuda():
    text = MAKEFILE.read_text(encoding="utf-8")
    value = re.search(r"^CTEST_EXCLUDED := (.+)$", text, re.M).group(1).replace("$$", "$")
    for label in ("optimizer-steps", "cuda"):
        assert re.search(value, label)
    for label in ("unit", "integration", "ppo", "optimizer"):
        assert not re.search(value, label)
    calls = [line for line in text.splitlines() if "ctest " in line]
    assert calls
    for line in calls:
        # La exclusión puede ir en la línea siguiente de la misma orden.
        command = text[text.index(line) :].split("done", 1)[0]
        assert "-LE '$(CTEST_EXCLUDED)'" in command
