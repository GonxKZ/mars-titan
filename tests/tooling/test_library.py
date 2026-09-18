"""Contratos que protegen procedencia, integridad y archivos locales."""

import json

import pytest

from scripts import reference_library as library


def test_catalog_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([{"id": "paper"}, {"id": "paper"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicado"):
        library.read_catalogs([path])


@pytest.mark.parametrize("identifier", ["../secret", "/absolute", "paper/name", ""])
def test_catalog_rejects_ids_that_can_escape_library(tmp_path, identifier):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([{"id": identifier}]), encoding="utf-8")
    with pytest.raises(ValueError, match="identificador"):
        library.read_catalogs([path])


def test_catalog_preserves_source_metadata_and_order(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text('[{"id": "paper-1", "year": 2025, "pdf_url": null}]', encoding="utf-8")
    assert library.read_catalogs([path]) == [{"id": "paper-1", "year": 2025, "pdf_url": None}]


def test_pdf_verification_rejects_html_even_with_pdf_suffix(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"<html>Access denied</html>")
    with pytest.raises(ValueError, match="PDF"):
        library.pdf_metadata(path)


def test_pdf_metadata_reports_hash_and_size(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = library.pdf_metadata(path)
    assert result["bytes"] == 15
    assert result["sha256"] == "14bcd090baf31edba64e9cbd8cdfc15f943344aa72cb3675ad8e91bfcbce03ad"


def test_existing_pdf_is_preserved_without_network_access(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = library.download_reference(
        {"id": "paper", "pdf_url": "https://invalid.example/paper.pdf"}, tmp_path
    )
    assert result["status"] == "cached_unverified"
    assert result["url"] is None
    assert path.read_bytes() == b"%PDF-1.4\n%%EOF\n"


def test_source_without_pdf_is_reference_only(tmp_path):
    result = library.download_reference({"id": "book", "pdf_url": None}, tmp_path)
    assert result["status"] == "reference_only"
    assert list(tmp_path.iterdir()) == []


def test_non_https_download_is_rejected_without_creating_file(tmp_path):
    result = library.download_reference({"id": "paper", "pdf_url": "file:///etc/passwd"}, tmp_path)
    assert result["status"] == "failed"
    assert not (tmp_path / "paper.pdf").exists()


def test_truncated_pdf_is_rejected_even_with_valid_header(tmp_path):
    path = tmp_path / "partial.pdf"
    path.write_bytes(b"%PDF-1.4\npartial download")
    with pytest.raises(ValueError, match="incompleto"):
        library.pdf_metadata(path)


def test_changed_source_url_does_not_relabel_cached_bytes(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    (tmp_path / "paper.source.json").write_text(
        json.dumps(
            {
                "url": "https://example.org/v1.pdf",
                "sha256": "14bcd090baf31edba64e9cbd8cdfc15f943344aa72cb3675ad8e91bfcbce03ad",
            }
        ),
        encoding="utf-8",
    )
    result = library.download_reference(
        {"id": "paper", "pdf_url": "https://example.org/v2.pdf"}, tmp_path
    )
    assert result["status"] == "failed"
    assert path.read_bytes() == b"%PDF-1.4\n%%EOF\n"
    assert "procedencia" in result["error"]
