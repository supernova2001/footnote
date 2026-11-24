from __future__ import annotations

from label_gold import agreement, apply_label


def test_apply_label_and_agreement():
    rows = [
        {"id": "a", "constructed_label": "supported", "method": "verbatim", "gold_label": None},
        {"id": "b", "constructed_label": "partial", "method": "added_year", "gold_label": None},
    ]
    assert apply_label(rows, 0, "s") and apply_label(rows, 1, "U")
    assert not apply_label(rows, 0, "x")
    result = agreement(rows)
    assert result["labelled"] == 2 and result["agreement"] == 0.5
    assert result["disagreements"] == {"partial->unsupported (added_year)": 1}
    assert agreement([{"gold_label": None}])["agreement"] is None
