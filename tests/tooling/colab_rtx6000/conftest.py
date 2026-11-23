"""Las pruebas del kit de Colab importan sus módulos por nombre, como los ejecuta el cuaderno."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

for entry in (ROOT / "scripts" / "colab_rtx6000", ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
