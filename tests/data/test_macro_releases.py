"""Preparación reproducible, cobertura del archivo y recuperación sin sobrescrituras."""

import hashlib
import importlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

CATALOG = Path("data/catalogs/macro-indicators.csv")


@pytest.fixture
def release_sources(tmp_path):
    cache = tmp_path / "sources"
    cache.mkdir()
    nbs = {
        "cn_cpi_yoy_published": (
            "Consumer Prices for November 2023",
            "<table><tr><td>Item</td><td>Y/Y (%)</td></tr>"
            "<tr><td>Consumer Prices</td><td>0.0</td></tr></table>",
        ),
        "cn_industrial_yoy_published": (
            "Industrial Production Operation in November 2023",
            "<table><tr><td>Item</td><td>November</td></tr><tr><td></td><td>Y/Y (%)</td></tr>"
            "<tr><td>Value-added of Industry Above Designated Size</td>"
            "<td>6.6</td></tr></table>",
        ),
        "cn_manufacturing_pmi": (
            "Purchasing Managers Index for November 2023",
            "<p>Seasonally Adjusted</p><table><tr><td></td><td>PMI</td></tr>"
            "<tr><td>2023-November</td><td>49.4</td></tr></table>",
        ),
    }
    monetary = {
        "cn_m2_stock": ("2023年11月金融统计数据报告", "11月末，广义货币（M2）余额291.2万亿元。"),
        "cn_tsf_flow": (
            "2023年11月社会融资规模增量统计数据报告",
            "11月份社会融资规模增量为2.45万亿元。",
        ),
        "cn_tsf_stock": (
            "2023年11月社会融资规模存量统计数据报告",
            "11月末社会融资规模存量为376.39万亿元。",
        ),
    }
    docs = []
    for index, (indicator, (title, body)) in enumerate((nbs | monetary).items()):
        if indicator in nbs:
            content = (
                f'<h1 class="con_titles">{title}</h1><div class="info">'
                "National Bureau of Statistics of China 2023-12-13 09:30</div>"
                f'<div class="TRS_Editor">{body}</div>'
            ).encode()
            url = f"https://www.stats.gov.cn/english/PressRelease/202312/t20231213_{index}.html"
        else:
            content = (
                f'<meta name="PubDate" content="2023-12-13"><h2>{title}</h2>'
                '<span id="shijian">2023-12-13 17:00:00</span>'
                f'<div id="zoom">{body}</div>'
            ).encode()
            url = f"https://www.pbc.gov.cn/diaochatongjisi/116219/116225/{index}/index.html"
        digest = hashlib.sha256(content).hexdigest()
        (cache / (digest + ".html")).write_bytes(content)
        docs.append(
            dict(
                url=url,
                sha256=digest,
                indicator_id=indicator,
                reference_periods=["2023-11"],
                availability_bound="2023-12-13",
                exclusion=None,
            )
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                first_reference_period="2023-11",
                last_reference_period="2023-11",
                documents=docs,
            )
        )
    )
    return path, cache


def run(sources, output):
    api = importlib.import_module("mars_titan.data.macro_releases")
    return api.prepare_releases(
        sources[0], CATALOG, sources[1], output, market="US", start="2023-12-12", end="2023-12-18"
    )


def test_preparation_keeps_sources_publishes_six_concepts_and_resumes_offline(
    release_sources, tmp_path
):
    report = run(release_sources, tmp_path / "edition")
    assert report["archive_complete"] is True
    assert report["documents"] == 6
    rows = pq.read_table(tmp_path / "edition/macro.parquet").to_pylist()
    assert len(rows) == 30
    assert sum(r["value"] is not None for r in rows) == 18
    assert len(list((tmp_path / "edition/sources").iterdir())) == 6
    observations = pq.read_table(tmp_path / "edition/observations.parquet").to_pylist()
    money = next(r for r in observations if r["indicator_id"] == "cn_m2_stock")
    assert money["source_value"] == "291.2"
    assert money["source_unit"] == "万亿元"
    nbs = next(r for r in observations if r["indicator_id"] == "cn_cpi_yoy_published")
    assert nbs["archive_path_date"] == "2023-12-13"
    original = (tmp_path / "edition/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(release_sources, tmp_path / "edition")
    assert (tmp_path / "edition/report.json").read_bytes() == original


@pytest.mark.parametrize(
    "defect", ["corrupt", "missing", "duplicate", "period", "future_bound", "unknown_field"]
)
def test_corrupt_or_incomplete_sources_do_not_publish(release_sources, tmp_path, defect):
    path, cache = release_sources
    manifest = json.loads(path.read_text())
    if defect == "corrupt":
        source = cache / (manifest["documents"][0]["sha256"] + ".html")
        source.write_bytes(source.read_bytes().replace(b">0.0<", b">2.0<"))
    elif defect == "missing":
        manifest["documents"].pop()
    elif defect == "duplicate":
        manifest["documents"].append(manifest["documents"][0])
    elif defect == "period":
        manifest["documents"][0]["reference_periods"] = ["2023-10"]
    elif defect == "future_bound":
        manifest["documents"][0]["availability_bound"] = "2023-12-12"
    else:
        manifest["injected"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(release_sources, tmp_path / "edition")
    assert not (tmp_path / "edition").exists()


def test_missing_future_releases_cannot_be_hidden_by_unlimited_forward_fill(
    release_sources, tmp_path
):
    api = importlib.import_module("mars_titan.data.macro_releases")
    with pytest.raises(ValueError, match="no cubre"):
        api.prepare_releases(
            release_sources[0],
            CATALOG,
            release_sources[1],
            tmp_path / "edition",
            market="US",
            start="2023-12-12",
            end="2024-03-01",
        )


def test_downloaded_corruption_does_not_enter_cache(release_sources, tmp_path, monkeypatch):
    api = importlib.import_module("mars_titan.data.macro_releases")
    path, cache = release_sources
    item = json.loads(path.read_text())["documents"][0]
    empty = tmp_path / "empty-cache"
    monkeypatch.setattr(api, "_download", lambda url: b"broken response")
    with pytest.raises(ValueError, match="huella"):
        api._content(item, empty, True)
    assert not empty.exists()


def test_missing_cached_document_is_explicit_and_download_can_recover(
    release_sources, tmp_path, monkeypatch
):
    api = importlib.import_module("mars_titan.data.macro_releases")
    path, cache = release_sources
    item = json.loads(path.read_text())["documents"][0]
    content = (cache / (item["sha256"] + ".html")).read_bytes()
    empty = tmp_path / "empty-cache"
    with pytest.raises(FileNotFoundError):
        api._content(item, empty, False)
    monkeypatch.setattr(api, "_download", lambda url: content)
    assert api._content(item, empty, True) == content
    assert api._content(item, empty, False) == content


@pytest.mark.parametrize("status", [200, 301, 404])
def test_download_checks_status_effective_url_and_does_not_follow_redirects(
    release_sources, monkeypatch, status
):
    from types import SimpleNamespace

    api = importlib.import_module("mars_titan.data.macro_releases")
    item = json.loads(release_sources[0].read_text())["documents"][0]

    def request(command, **kwargs):
        assert "--location" not in command
        assert command[command.index("--max-filesize") + 1] == str(3 * 1024**2)
        assert kwargs["timeout"] == 50
        Path(command[command.index("--output") + 1]).write_bytes(b"source")
        return SimpleNamespace(returncode=0, stdout=f"{status}\n{item['url']}", stderr="")

    monkeypatch.setattr(api.subprocess, "run", request)
    if status == 200:
        assert api._download(item["url"]) == b"source"
    else:
        with pytest.raises(OSError):
            api._download(item["url"])


def test_module_cli_prepares_the_same_artifacts(release_sources, tmp_path, capsys):
    api = importlib.import_module("mars_titan.data.macro_releases")
    assert (
        api.main(
            [
                "--manifest",
                str(release_sources[0]),
                "--catalog",
                str(CATALOG),
                "--source-cache",
                str(release_sources[1]),
                "--output",
                str(tmp_path / "edition"),
                "--market",
                "US",
                "--start",
                "2023-12-12",
                "--end",
                "2023-12-18",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["documents"] == 6
