"""Canales contables separados por moneda y contrato de la representación conjunta."""

from .company_factors import FACTOR_CONCEPTS
from .samples import FUNDAMENTAL_CONCEPTS

USD_CONCEPTS = FUNDAMENTAL_CONCEPTS + FACTOR_CONCEPTS
CAD_CONCEPTS = tuple(name.removesuffix(":USD") + ":CAD" for name in FUNDAMENTAL_CONCEPTS)
COMMON_CONCEPTS = USD_CONCEPTS + CAD_CONCEPTS
CNY_CONCEPTS = (
    "cn-reported:Assets:CNY",
    "cn-reported:Liabilities:CNY",
    "cn-reported:EquityIncludingNoncontrollingInterest:CNY",
)
JOINT_CONCEPTS = (*COMMON_CONCEPTS, *CNY_CONCEPTS)
HISTORICAL_ACCOUNTING = "historical_disjoint_accounting_v1"


def historical_accounting_context(market):
    if market not in {"US", "CN"}:
        raise ValueError("El contexto contable necesita un mercado US o CN")
    return dict(
        fundamental_concepts=COMMON_CONCEPTS if market == "US" else CNY_CONCEPTS,
        source_unit="USD" if market == "US" else "CNY",
        company_factors=market == "US",
        target_fundamental_concepts=JOINT_CONCEPTS,
    )
