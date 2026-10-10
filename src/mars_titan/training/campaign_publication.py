"""Paso final de la campaña: comparación, cartera, matriz y comparación postentrenada.

La declaración (``campaign_publication``) fija antes de ver resultados qué campaña se publica,
con qué matriz de comparaciones y con qué comparación de su etapa de adaptadores.
``load_publication`` exige que la matriz lea la misma comparación que la campaña, con la
misma huella, y que la comparación postentrenada derive de una etapa de esa campaña. Así una
matriz declarada para otra campaña no llega a evaluarse.

``run_publication`` se ejecuta cuando la campaña y sus etapas están confirmadas, después del
recorrido ventana a ventana si lo hubo:

1. publica las fuentes de cada ámbito de la campaña con ``masked_campaign.write_sources``;
2. calcula la comparación walk-forward de cada ámbito y, si la comparación la declara, la
   cartera larga y corta, desde los agregados por ventana de la retención v2 cuando se
   indican;
3. publica el manifiesto de fuentes de la matriz con esos informes y evalúa la matriz;
4. publica las fuentes de cada padre de la etapa de adaptadores y su comparación, también
   desde los agregados por ventana que guardó el recorrido.

Todo se escribe en una carpeta provisional junto al destino, que solo toma el nombre del
destino cuando cada paso ha terminado y el recibo guarda la huella de cada archivo. Un fallo
deja la carpeta provisional con su marca, y la siguiente ejecución la descarta y repite.
Nada de este módulo ajusta modelos ni abre la reserva de 2024.
"""

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import comparison_matrix, long_short_comparison
from mars_titan.evaluation import session_table_contrasts as tables
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.posttraining import stage_comparison

from .campaign_plan import load_campaign

KIND = "campaign_publication"
RECEIPT_KIND = "campaign_publication_receipt"
DECLARED = "declared_before_results"
_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "name",
    "campaign",
    "comparison_matrix",
    "posttraining_comparison",
    "final_test_opened",
}
# Marca de la carpeta provisional. Solo una carpeta con ella se descarta al repetir.
PARTIAL_MARK = ".campaign-publication-partial"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_publication(path):
    """Validar la declaración y comprobar que matriz y comparación son de la campaña."""
    path = Path(path)
    document, digest = read_manifest(path, 1024**2)
    _require(
        isinstance(document, dict)
        and set(document) == _FIELDS
        and document["schema_version"] == 1
        and document["kind"] == KIND
        and document["status"] == DECLARED
        and document["final_test_opened"] is False
        and isinstance(document["name"], str)
        and walk._name(document["name"].replace("-", "_"))
        and isinstance(document["campaign"], str)
        and isinstance(document["comparison_matrix"], str)
        and (
            document["posttraining_comparison"] is None
            or isinstance(document["posttraining_comparison"], str)
        ),
        "La publicación de la campaña no cumple su contrato",
    )
    folder = path.parent
    campaign = load_campaign(folder / document["campaign"])
    matrix_path = (folder / document["comparison_matrix"]).resolve()
    matrix = comparison_matrix.load_matrix(matrix_path)
    _require(
        (matrix_path.parent / matrix["comparison"]).resolve() == Path(campaign["comparison_path"])
        and matrix["comparison_config"]["sha256"] == campaign["comparison_config"]["sha256"],
        f"La matriz {matrix['name']} no lee la comparación de la campaña {campaign['name']}",
    )
    posttraining_path, posttraining = None, None
    if document["posttraining_comparison"] is not None:
        posttraining_path = (folder / document["posttraining_comparison"]).resolve()
        posttraining = stage_comparison.load_declaration(posttraining_path)
        stage_campaign = posttraining["stage"]["campaign"]
        _require(
            stage_campaign["path"] == campaign["path"]
            and stage_campaign["sha256"] == campaign["sha256"],
            f"La comparación postentrenada {posttraining['name']} no deriva de una etapa de "
            f"la campaña {campaign['name']}",
        )
    return dict(
        document,
        sha256=digest,
        path=str(path.resolve()),
        campaign_config=campaign,
        matrix_path=str(matrix_path),
        matrix=matrix,
        posttraining_path=None if posttraining_path is None else str(posttraining_path),
        posttraining=posttraining,
    )


def check_publication(path):
    """Resumen de lo que publicará el paso final, sin leer predicciones."""
    loaded = load_publication(path)
    campaign, posttraining = loaded["campaign_config"], loaded["posttraining"]
    config = campaign["comparison_config"]
    return dict(
        status="checked",
        name=loaded["name"],
        declaration_sha256=loaded["sha256"],
        campaign=dict(name=campaign["name"], sha256=campaign["sha256"]),
        comparison=dict(name=config["name"], sha256=config["sha256"]),
        scopes=campaign["scopes"],
        long_short=walk.LONG_SHORT_FIELD in config,
        matrix=comparison_matrix.summary(loaded["matrix"]),
        posttraining=None
        if posttraining is None
        else dict(
            name=posttraining["name"],
            sha256=posttraining["sha256"],
            scopes=posttraining["stage"]["scopes"],
            parents=list(posttraining["configs"]),
        ),
        final_test_opened=False,
    )


def _partial(destination, digest):
    """Carpeta provisional del destino. Una anterior con la marca se descarta."""
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        _require(
            (partial / PARTIAL_MARK).is_file() and not partial.is_symlink(),
            f"{partial} existe y no es una publicación provisional",
        )
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    (partial / PARTIAL_MARK).write_text(digest + "\n", encoding="utf-8")
    return partial


def run_publication(
    path,
    views,
    output,
    destination,
    *,
    adapter_output=None,
    aggregates=None,
    edition=None,
    ablation=None,
    hours=None,
):
    """Publicar comparación, cartera, matriz y comparación postentrenada de la campaña.

    ``views`` y ``output`` son los de la campaña confirmada. ``adapter_output`` es la salida
    de la etapa de adaptadores y se exige si la declaración tiene comparación postentrenada.
    ``aggregates`` es la carpeta de agregados del recorrido ventana a ventana. ``edition`` es
    la edición de precios sin ajustar que pide la cartera. ``ablation`` (``stage`` y
    ``output``) añade la ablación de modalidades al informe walk-forward y debe acompañar a
    unos agregados que la incluyan. ``hours`` es el documento de horas por modelo de la matriz.
    """
    loaded = load_publication(path)
    campaign, posttraining = loaded["campaign_config"], loaded["posttraining"]
    config = campaign["comparison_config"]
    destination = Path(destination)
    safe_destination(destination)
    _require(not destination.exists(), "La publicación debe ser nueva")
    _require(
        (posttraining is None) == (adapter_output is None),
        "La salida de la etapa de adaptadores acompaña a la comparación postentrenada declarada",
    )
    _require(
        walk.LONG_SHORT_FIELD not in config or edition is not None,
        "La comparación declara la cartera larga y corta: falta la edición de precios",
    )
    from . import masked_campaign as engine
    from . import modality_ablation_stage

    partial = _partial(destination, loaded["sha256"])
    matrix = loaded["matrix"]
    published = {}
    base_sources = {}
    for scope in campaign["scopes"]:
        sources = engine.write_sources(campaign["path"], views, output, scope)
        base_sources[scope] = sources
        masked = None
        if ablation is not None:
            masked = modality_ablation_stage.write_sources(
                ablation["stage"], views, output, ablation["output"], scope
            )
        folder = partial / "walk-forward" / scope
        walk.write_walk_forward(
            campaign["comparison_path"],
            sources,
            scope,
            folder,
            ablation_sources=masked,
            aggregates=aggregates,
        )
        reports = [dict(kind=tables.WALK_SOURCE, path=folder / "comparison.json")]
        if walk.LONG_SHORT_FIELD in config:
            folder = partial / "long-short" / scope
            long_short_comparison.write_long_short(
                campaign["comparison_path"], sources, scope, edition, folder, aggregates=aggregates
            )
            reports.append(dict(kind=tables.PORTFOLIO_SOURCE, path=folder / "long_short.json"))
        manifest = tables.write_sources(
            partial / "matrix-sources" / f"{scope}.json",
            scope,
            reports,
            views=matrix["views"],
            config=matrix["comparison_config"],
            hours=hours,
        )
        report = comparison_matrix.write(
            loaded["matrix_path"], manifest, scope, partial / "matrix" / scope
        )
        published[scope] = dict(
            families=len(report["families"]),
            estimable=sum(len(f["estimable"]) for f in report["families"].values()),
        )
    parents = {}
    if posttraining is not None:
        for scope in posttraining["stage"]["scopes"]:
            for base_arm in posttraining["configs"]:
                sources = stage_comparison.write_sources(
                    loaded["posttraining_path"],
                    scope,
                    base_arm,
                    base_sources=base_sources[scope],
                    stage_output=adapter_output,
                    loaded=posttraining,
                )
                report = stage_comparison.write(
                    loaded["posttraining_path"],
                    sources,
                    scope,
                    base_arm,
                    partial / "posttraining" / scope / base_arm,
                    aggregates=aggregates,
                    loaded=posttraining,
                )
                parents[f"{scope}/{base_arm}"] = len(report["windows"])
    receipt = dict(
        schema_version=1,
        kind=RECEIPT_KIND,
        status="completed",
        declaration=dict(name=loaded["name"], sha256=loaded["sha256"]),
        campaign=dict(name=campaign["name"], sha256=campaign["sha256"]),
        comparison_sha256=config["sha256"],
        matrix=dict(name=matrix["name"], sha256=matrix["sha256"]),
        posttraining=None
        if posttraining is None
        else dict(name=posttraining["name"], sha256=posttraining["sha256"], parents=parents),
        from_aggregates=aggregates is not None,
        scopes=published,
        artifacts={
            str(file.relative_to(partial)): sha256(file)
            for file in sorted(partial.rglob("*"))
            if file.is_file() and file.name != PARTIAL_MARK
        },
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
    )
    atomic_json(partial / "publication.json", receipt)
    os.rename(partial, destination)
    # Ya publicada, la marca sobra. Si un corte la deja, el recibo sigue completo.
    (destination / PARTIAL_MARK).unlink()
    return receipt


def main(argv=None):
    from .masked_campaign import _views_argument

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validar la declaración sin leer predicciones")
    run = commands.add_parser("run", help="Publicar comparación, cartera, matriz y etapa")
    for command in (check, run):
        command.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--views", action="append", required=True)
    run.add_argument("--output", type=Path, required=True, help="Salida de la campaña")
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument("--adapter-output", type=Path)
    run.add_argument("--aggregates", type=Path, help="Agregados por ventana de la retención v2")
    run.add_argument("--edition", type=Path, help="Edición de precios sin ajustar")
    run.add_argument("--ablation-stage", type=Path)
    run.add_argument("--ablation-output", type=Path)
    run.add_argument("--hours", type=Path, help="Horas GPU por modelo para la matriz")
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_publication(args.declaration)
    else:
        _require(
            (args.ablation_stage is None) == (args.ablation_output is None),
            "--ablation-stage necesita su salida",
        )
        ablation = None
        if args.ablation_stage is not None:
            ablation = dict(stage=args.ablation_stage, output=args.ablation_output)
        receipt = run_publication(
            args.declaration,
            _views_argument(args.views),
            args.output,
            args.destination,
            adapter_output=args.adapter_output,
            aggregates=args.aggregates,
            edition=args.edition,
            ablation=ablation,
            hours=args.hours,
        )
        result = dict(
            status=receipt["status"],
            destination=str(args.destination),
            scopes=receipt["scopes"],
            posttraining=receipt["posttraining"],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
