from footnote.corpus.import_researchers import _extract_doi, _license_in_text, repair_text


def test_repair_text_removes_pdf_noise_and_controls() -> None:
    noisy = "A useful opening.\x00\n" + "G " * 40 + "\nA useful closing with enough ordinary words."
    repaired, changes = repair_text(noisy)
    assert "G G G" not in repaired
    assert "\x00" not in repaired
    assert "useful opening" in repaired and "useful closing" in repaired
    assert changes["repetitive_pdf_lines"] == 1
    assert changes["control_characters"] == 1


def test_license_parser_accepts_only_supported_explicit_urls() -> None:
    assert _license_in_text("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert _license_in_text("https://creativecommons.org/licenses/by-sa/4.0/") == "CC-BY-SA-4.0"
    assert _license_in_text("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert _license_in_text("https://creativecommons.org/licenses/by/3.0/") is None


def test_extract_doi_from_resolver_and_publisher_url() -> None:
    assert _extract_doi("https://doi.org/10.1234/example.5") == "10.1234/example.5"
    assert _extract_doi("https://example.test/article?id=10.1371/journal.pcbi.1008601&type=printable") == (
        "10.1371/journal.pcbi.1008601"
    )
