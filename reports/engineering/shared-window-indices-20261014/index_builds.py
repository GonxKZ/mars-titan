"""Construcciones de índices por vista en el plan de A v2, con y sin índices compartidos.

Uso: index_builds.py SALIDA.json
Cada ajuste indexado construye los cuatro tramos en su carpeta. La continuación de una meseta
reanuda en la carpeta de la meseta y reutiliza sus índices, así que no construye.
"""

import collections
import json
import sys

from mars_titan.training.campaign_plan import load_campaign, plan_campaign
from mars_titan.training.campaign_storage import INDEXED

jobs = plan_campaign(load_campaign("configs/baselines/historical-masked-campaign-a-v2.json"))
indexed = [job for job in jobs if job["model"] in INDEXED]
builders = [job for job in indexed if not job.get("plateau")]
per_view = collections.Counter((job["scope"], job["window"]) for job in builders)
document = dict(
    per_view={f"{scope}/{window}": n for (scope, window), n in sorted(per_view.items())}
)
with open(sys.argv[1], "w") as stream:
    json.dump(document, stream, indent=1)
print(
    dict(
        indexed=len(indexed),
        continuations=len(indexed) - len(builders),
        builders=len(builders),
        views=len(per_view),
        kinds=dict(collections.Counter(job["kind"] for job in builders)),
    )
)
