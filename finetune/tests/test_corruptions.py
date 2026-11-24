from __future__ import annotations

import random
import re

from footnote_ft import corruptions as c
from footnote_ft.text import normalize, sentence_spans

PASSAGE = (
    "Ada Linwood was a teacher and librarian who spent most of her life organising reading groups for working families. "
    "She was elected secretary of the Harlow Reading Society in 1911, which then had about 40 members. "
    "Her letters suggest that she often paid for new books herself. She did not seek public office."
)


def rng() -> random.Random:
    return random.Random(0)


def test_sentence_spans_are_verbatim_and_respect_initials():
    text = "W. E. B. Du Bois wrote the study. Dr. Smith agreed with it in 1903. It sold well!"
    spans = sentence_spans(text)
    assert [s.text for s in spans] == [
        "W. E. B. Du Bois wrote the study.",
        "Dr. Smith agreed with it in 1903.",
        "It sold well!",
    ]
    assert all(text[s.start : s.end] == s.text for s in spans)


def test_numbers_followed_by_punctuation_are_found():
    from footnote_ft.text import numbers

    found = [m.group(0) for m in numbers("In 1911, it had 1,200 books, 3.5 shelves and 40 members.")]
    assert found == ["1911", "1,200", "40"]


def test_verbatim_is_supported_with_itself_as_evidence():
    anchor = "Her letters suggest that she often paid for new books herself."
    cand = c.verbatim(anchor, PASSAGE, rng())
    assert (cand.label, cand.claim, cand.evidence) == ("supported", anchor, anchor)


def test_trim_clause_drops_a_non_restrictive_clause_or_parenthetical():
    cand = c.trim_clause(
        "She was elected secretary of the Harlow Reading Society in 1911, which then had about 40 members.",
        PASSAGE,
        rng(),
    )
    assert cand.claim == "She was elected secretary of the Harlow Reading Society in 1911."
    assert cand.label == "supported"
    paren = c.trim_clause(
        "The society tried to raise a fund of 500 dollars for that purpose (a large sum at the time).", PASSAGE, rng()
    )
    assert paren.claim == "The society tried to raise a fund of 500 dollars for that purpose."
    assert c.trim_clause("She did not seek public office or any paid post.", PASSAGE, rng()) is None


def test_certainty_upgrade_makes_one_hedge_certain():
    anchor = "Her letters suggest that she often paid for new books herself."
    cand = c.certainty_upgrade(anchor, PASSAGE, rng())
    assert cand.label == "partial" and cand.evidence == anchor
    assert cand.claim != anchor
    assert re.search(r"\bprove\b|\balways\b", cand.claim)
    # "May" the month is not a hedge.
    assert c.certainty_upgrade("In May the school opened its doors to every child in the town.", PASSAGE, rng()) is None


def test_added_year_only_where_the_sentence_has_no_date():
    anchor = "Her letters suggest that she often paid for new books herself."
    cand = c.added_year(anchor, PASSAGE, rng())
    year = re.match(r"In (\d{4}), her letters", cand.claim)
    assert year and year.group(1) not in PASSAGE
    assert cand.label == "partial"
    assert c.added_year("She was elected secretary of the society in 1911.", PASSAGE, rng()) is None


def test_added_place_declines_when_the_passage_locates_anything():
    plain = "The society met every week and lent books to its members for a small fee."
    cand = c.added_place(plain, plain, rng())
    place = re.search(r", in ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\.$", cand.claim).group(1)
    assert place not in plain and cand.label == "partial"
    located = "Samuel Okafor taught for thirty years at Bramwell College and never left."
    assert c.added_place(located, located, rng()) is None


def test_added_qualifier_inserts_one_word_into_a_passive():
    anchor = "She was elected secretary of the Harlow Reading Society in 1911."
    cand = c.added_qualifier(anchor, PASSAGE, rng())
    assert re.fullmatch(r"She was (unanimously|immediately|officially|quickly) elected secretary.*", cand.claim)
    assert cand.label == "partial"


def test_negation_adds_or_removes_a_negation():
    assert (
        c.negation("She was elected secretary of the society.", PASSAGE, rng()).claim
        == "She was not elected secretary of the society."
    )
    assert c.negation("She did not seek public office.", PASSAGE, rng()).claim == "She did seek public office."
    assert c.negation("Linwood believed that a library should never charge its readers.", PASSAGE, rng()).claim == (
        "Linwood believed that a library should charge its readers."
    )
    assert c.negation("She founded the reading society for working families.", PASSAGE, rng()).claim == (
        "She never founded the reading society for working families."
    )
    cand = c.negation("She was elected secretary of the society.", PASSAGE, rng())
    assert cand.label == "unsupported" and cand.evidence == ""


def test_number_swap_changes_a_number_to_one_the_passage_lacks():
    anchor = "She was elected secretary of the Harlow Reading Society in 1911, which then had about 40 members."
    for seed in range(20):
        cand = c.number_swap(anchor, PASSAGE, random.Random(seed))
        assert cand.label == "unsupported"
        changed = set(re.findall(r"\d+", cand.claim)) - set(re.findall(r"\d+", anchor))
        assert len(changed) == 1 and changed.pop() not in PASSAGE
    assert c.number_swap("She did not seek public office at any time.", PASSAGE, rng()) is None


def test_entity_swap_uses_a_name_the_passage_never_mentions():
    anchor = "Ada Linwood was a teacher and librarian who organised reading groups."
    cand = c.entity_swap(anchor, PASSAGE, rng(), pool=["Marcus Bell", "Ada Linwood"])
    assert cand.claim == "Marcus Bell was a teacher and librarian who organised reading groups."
    assert cand.label == "unsupported"
    assert c.entity_swap(anchor, PASSAGE, rng(), pool=[]) is None
    assert c.entity_swap(anchor, PASSAGE, rng(), pool=["Ada Linwood"]) is None


def test_no_rule_returns_a_partial_or_unsupported_claim_that_is_in_the_passage():
    for sentence in [s.text for s in sentence_spans(PASSAGE)]:
        for name, rule in c.SENTENCE_RULES.items():
            kwargs = {"pool": ["Marcus Bell"]} if name == "entity_swap" else {}
            cand = rule(sentence, PASSAGE, rng(), **kwargs)
            if cand is not None and cand.label != "supported":
                assert normalize(cand.claim) not in normalize(PASSAGE), (name, cand.claim)
