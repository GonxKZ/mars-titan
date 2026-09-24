"""Codificación congelada de noticias, gráficos y contexto de empresas ficticias."""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.charts import chart_png
from mars_titan.data.company_factors import FACTOR_CONCEPTS, derive_company_factors
from mars_titan.data.macro_formulas import Formula
from mars_titan.data.samples import FUNDAMENTAL_CONCEPTS, macro_vector, numeric_context
from mars_titan.data.storage import sha256

from .worlds import recipe_fingerprints

CATALOG = Path(__file__).parents[3] / "data/catalogs/macro-indicators.csv"


class EncodedWorld:
    """Compartir un mundo bruto y exigir la versión exacta de sus codificadores."""

    def __init__(self, world, encoders, *, expected_spec, catalog=CATALOG):
        if encoders.spec != expected_spec:
            raise ValueError("La versión del codificador no coincide con la referencia")
        self.world, self.encoders, self.raw_world = world, encoders, world
        self.config, self.index, self.partition = world.config, world.index, world.partition
        self.max_assets = world.max_assets
        self.shapes = {
            **world.shapes,
            "news": (384,),
            "charts": (512,),
            "fundamentals": (45,),
            "macro": (420,),
        }
        with Path(catalog).open() as stream:
            self.catalog = list(csv.DictReader(stream))
        if len(self.catalog) != 140:
            raise ValueError("El catálogo macro no corresponde a las 140 posiciones declaradas")
        curve = next(row for row in self.catalog if row["id"] == "us_curve_10y_2y")
        self.curve = Formula.parse(
            curve["formula"], set(curve["input_ids"].split("|")), curve["unit"]
        )
        self.identity = {
            **world.identity,
            "encoding": hashlib.sha256(
                json.dumps(
                    dict(
                        encoders=expected_spec,
                        fundamental_concepts=FUNDAMENTAL_CONCEPTS + FACTOR_CONCEPTS,
                        macro_catalog_sha256=sha256(Path(catalog)),
                        transformations=recipe_fingerprints(),
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "encoder_spec": expected_spec,
            "encoding_code_sha256": sha256(Path(__file__)),
            "simulated_macro": ["us_treasury_2y", "us_treasury_10y", "us_curve_10y_2y"],
            "macro_units": {
                row["id"]: row["unit"]
                for row in self.catalog
                if row["id"] in {"us_treasury_2y", "us_treasury_10y", "us_curve_10y_2y"}
            },
        }
        self.source_sha256 = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        self.manifest_sha256 = self.source_sha256

    def __len__(self):
        return len(self.world)

    def _fundamentals(self, at):
        world, rows = self.world, []
        period = world.periods[at]
        available = datetime.fromtimestamp(world.fundamental_available_at[at] / 1e6, UTC)
        end = datetime.fromtimestamp(world.fundamental_period_end[at] / 1e6, UTC).date().isoformat()
        age = (world.times[at] - world.fundamental_available_at[at]) / 86_400_000_000
        for i in range(world.count_at(at)):
            assets, liabilities, equity = (
                world.assets[period, i],
                world.liabilities[period, i],
                world.equity[period, i],
            )
            values = [
                assets,
                0.6 * assets,
                liabilities,
                0.6 * liabilities,
                equity,
                0.2 * assets,
                0.3 * liabilities,
                0.2 * assets,
            ]
            facts = [
                dict(
                    concept=concept,
                    unit="USD",
                    value=float(value),
                    period_start=None,
                    period_end=end,
                    filed=available.date().isoformat(),
                    accession=f"fiction-{period}",
                    available_at=available,
                )
                for concept, value in zip(FUNDAMENTAL_CONCEPTS, values, strict=True)
            ]
            ratios = {row["concept"]: row["value"] for row in derive_company_factors(facts)}
            rows.append(
                numeric_context(values + [ratios[key] for key in FACTOR_CONCEPTS], [age] * 15)
            )
        return np.asarray(rows, dtype=np.float32)

    def _macro(self, at):
        available = datetime.fromtimestamp(self.world.times[at] / 1e6, UTC)
        values = {
            "us_treasury_2y": 3 + 10 * self.world.market_returns[at],
            "us_treasury_10y": 4 + 5 * self.world.market_returns[at],
        }
        values["us_curve_10y_2y"] = self.curve.calculate(
            {(name, 0): value for name, value in values.items()}
        )
        rows = [
            dict(
                indicator_id=row["id"],
                value=values.get(row["id"]),
                available_at=available if row["id"] in values else None,
            )
            for row in self.catalog
        ]
        vector, _ = macro_vector(rows, available)
        return np.broadcast_to(
            np.asarray(vector, dtype=np.float32), (self.world.count_at(at), 420)
        ).copy()

    def __call__(self, position):
        raw = self.world(position)
        if raw is None:
            return None
        at = position + self.config.context - 1
        raw["inputs"]["news"] = np.stack(
            [self.encoders.text(text) for text in self.world.events(at)]
        )
        pngs = [
            chart_png(self.world.prices[:, i, :4], end_index=at, context=self.config.context)
            for i in range(self.world.count_at(at))
        ]
        raw["inputs"]["charts"] = np.concatenate(
            [self.encoders.images(pngs[start : start + 64]) for start in range(0, len(pngs), 64)]
        )
        raw["inputs"]["fundamentals"] = self._fundamentals(at)
        raw["inputs"]["macro"] = self._macro(at)
        return raw
