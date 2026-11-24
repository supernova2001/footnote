"""The licences Footnote accepts, and the attribution line each source must carry."""

from __future__ import annotations

from footnote.domain import License, LicenseId

_NAMES: dict[LicenseId, tuple[str, str]] = {
    "CC-BY-4.0": ("Creative Commons Attribution 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "CC-BY-SA-4.0": ("Creative Commons Attribution-ShareAlike 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    "CC0-1.0": (
        "Creative Commons Zero 1.0 (public domain dedication)",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    "PD-US": ("Public domain in the United States", "https://www.gutenberg.org/policy/permission.html"),
}

# Europe PMC licence strings that may be used. Anything else (cc by-nc, cc by-nd, ...) is refused:
# the writer agent adapts text, which -ND forbids, and a portfolio demo should not lean on -NC.
EUROPE_PMC_ACCEPTED: dict[str, LicenseId] = {"cc by": "CC-BY-4.0", "cc0": "CC0-1.0"}


def make_license(license_id: LicenseId, attribution: str) -> License:
    name, url = _NAMES[license_id]
    return License(id=license_id, name=name, url=url, attribution=attribution)
