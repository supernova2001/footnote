"""An invented corpus for tests and smoke runs: nine fictional scholars, three passages each.

Every name, number and event here is made up (no real person, no third-party text), so the fixture
carries no licence or privacy question. The sentences are written to exercise each rule: hedges
("suggests", "may", "often"), passives ("was elected"), years and counts, multi-word names,
negations, non-restrictive clauses and parentheticals, and passages with and without places.

    python -m footnote_ft.demo_corpus data/smoke/chunks.jsonl     # writes the backend's Chunk shape
"""

from __future__ import annotations

import sys
from typing import Any

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "demo-ada-linwood",
        "title": "Ada Linwood (biography)",
        "scholar": "ada-linwood",
        "passages": [
            "Ada Linwood was a teacher and librarian who spent most of her life organising reading groups for working families. "
            "She was elected secretary of the Harlow Reading Society in 1911, which then had about 40 members. "
            "Her letters suggest that she often paid for new books herself. She did not seek public office.",
            "Linwood believed that a lending library should never charge its poorest readers. "
            "The society tried to raise a fund of 500 dollars for that purpose (a large sum at the time). "
            "Some of the members argued that the fee kept the collection in good repair. The debate lasted three years.",
            "In 1919 Ada Linwood published a short guide for reading circles. "
            "The guide was widely copied by other societies, and it may have reached several thousand readers. "
            "She retired from teaching in 1932, at the age of sixty.",
        ],
    },
    {
        "id": "demo-marcus-bell",
        "title": "Marcus Bell, Essays on Night Schools",
        "scholar": "marcus-bell",
        "passages": [
            "Marcus Bell argued that night schools were the most important invention of his generation. "
            "His survey of 12 schools found that most students worked a full day before class. "
            "Attendance was usually highest in the winter months, when farm work was slow.",
            "Bell was appointed inspector of evening classes in 1898. "
            "He visited every school twice a year and wrote a report on each visit. "
            "The reports suggest that teachers were paid less than half of a day-school salary.",
            "Many of the students were adults who had never attended school as children. "
            "Bell believed that they learned faster than younger pupils. "
            "He did not publish his figures on examination results, which have since been lost.",
        ],
    },
    {
        "id": "demo-elena-ruiz",
        "title": "Ruiz, Reading Aloud and Comprehension",
        "scholar": "elena-ruiz",
        "passages": [
            "Elena Ruiz studied how reading aloud affects comprehension in children aged 8 to 10. "
            "The study followed 214 pupils over one school year. "
            "Children who read aloud with a partner scored 11 points higher on the final test than those who read silently.",
            "The authors suggest that the effect may come from the extra attention that partners give each other. "
            "The effect was smaller for pupils who were already strong readers. "
            "Ruiz cautioned that the study was not randomised (classes chose their own method).",
            "The results were published in 2016 and were later discussed by several school boards. "
            "The authors called for a larger trial with random assignment. "
            "They noted that the cost of the method was close to zero.",
        ],
    },
    {
        "id": "demo-samuel-okafor",
        "title": "Samuel Okafor (biography)",
        "scholar": "samuel-okafor",
        "passages": [
            "Samuel Okafor was a mathematician who taught for thirty years at Bramwell College. "
            "He was known for teaching geometry through drawing rather than proofs. "
            "His students often described his classes as the hardest and the most rewarding they took.",
            "Okafor was awarded the college's teaching medal in 1954, the first year it was given. "
            "He declined a later offer to become dean, which surprised many colleagues. "
            "He believed that a teacher's place was in the classroom.",
            "After his retirement Okafor wrote a textbook that was used in some of the region's high schools. "
            "The book was reprinted four times before 1980. "
            "It probably sold more copies than any other work by a Bramwell professor.",
        ],
    },
    {
        "id": "demo-lillian-parks",
        "title": "The Letters of Lillian Parks",
        "scholar": "lillian-parks",
        "passages": [
            "Lillian Parks wrote more than 300 letters to her brother Thomas Parks between 1890 and 1905. "
            "The letters describe her work as a nurse and her efforts to open a clinic for mill workers. "
            "She seems to have kept copies of every letter she sent.",
            "The clinic was opened in 1897 with two beds and a single donated microscope. "
            "Parks hoped that the mill owners would pay for a second nurse. "
            "They did not, and she paid the nurse's wages from her own savings for two years.",
            "Her brother kept the letters, which were donated to a county archive in 1961. "
            "Historians believe that they are the most complete record of the clinic's early years. "
            "Some of the letters were damaged by water and cannot be read.",
        ],
    },
    {
        "id": "demo-harold-vance",
        "title": "Harold Vance, Field Journal",
        "scholar": "harold-vance",
        "passages": [
            "Harold Vance kept a field journal during his survey of river birds in the summer of 1923. "
            "He recorded 87 species, of which 9 had not been described in the region before. "
            "The journal suggests that he worked alone for most of the survey.",
            "Vance was elected to the regional naturalists' club in 1925. "
            "He gave a lecture on his findings, which was attended by about 150 people. "
            "Several of his drawings were later printed in the club's bulletin.",
            "Vance argued that the river's bird population was likely to fall as new mills were built. "
            "He did not live to see the decline he had predicted. "
            "His journal was published in full in 1978.",
        ],
    },
    {
        "id": "demo-grace-moreno",
        "title": "Grace Moreno (biography)",
        "scholar": "grace-moreno",
        "passages": [
            "Grace Moreno was a chemist who studied the purification of drinking water. "
            "She was hired by the city's water board in 1931 as its first woman chemist. "
            "Her early tests showed that the city's water often failed the board's own standards.",
            "Moreno tried to persuade the board to install sand filters across the whole system. "
            "The board agreed to a trial at a single pumping station. "
            "The trial reduced the number of failed tests by about half within a year.",
            "Moreno was promoted to chief chemist in 1940. "
            "She trained some of the engineers who later designed the new treatment plant. "
            "She never published her methods in a journal, which is why her work is little known today.",
        ],
    },
    {
        "id": "demo-james-whitfield",
        "title": "James Whitfield, Lectures on Rhetoric",
        "scholar": "james-whitfield",
        "passages": [
            "James Whitfield delivered a series of twelve lectures on rhetoric at Ashford Institute in 1887. "
            "The lectures were attended mostly by young clerks and teachers. "
            "Whitfield believed that clear speech was a civic duty.",
            "In the fourth lecture Whitfield argued that a speaker should never read from a written text. "
            "He suggested that notes on a single card were enough for most speeches. "
            "Some of the listeners complained that this advice was impractical.",
            "The lectures were published as a book in 1889, which went through three editions. "
            "The book may have influenced the debating clubs that formed in the following decade. "
            "Whitfield did not lecture again after 1890.",
        ],
    },
    {
        "id": "demo-nora-castell",
        "title": "Castell, School Gardens and Attendance",
        "scholar": "nora-castell",
        "passages": [
            "Nora Castell examined whether school gardens improved attendance in 36 rural schools. "
            "Schools with a garden recorded 6 more days of attendance per pupil each year. "
            "The analysis suggests that the effect was strongest in the spring term.",
            "Castell noted that schools with gardens also tended to have more teachers. "
            "The difference in attendance may therefore reflect staffing rather than the gardens themselves. "
            "The study was funded by a small grant from a teachers' association.",
            "The paper was published in 2019 and has been cited by several later studies. "
            "Castell recommended that future work should compare schools with similar staffing. "
            "She did not report any results for urban schools.",
        ],
    },
]


def chunk_rows() -> list[dict[str, Any]]:
    """The corpus as backend `Chunk` rows (see backend/src/footnote/domain.py)."""
    rows = []
    for doc in DOCUMENTS:
        offset = 0
        for ordinal, text in enumerate(doc["passages"]):
            rows.append(
                {
                    "id": f"{doc['id']}:{ordinal:04d}",
                    "document_id": doc["id"],
                    "ordinal": ordinal,
                    "title": doc["title"],
                    "section": None,
                    "text": text,
                    "char_start": offset,
                    "char_end": offset + len(text),
                    "scholar_ids": [doc["scholar"]],
                    "license_id": "CC0-1.0",
                    "source_url": f"https://example.org/demo/{doc['id']}",
                }
            )
            offset += len(text) + 2
    return rows


if __name__ == "__main__":
    from .io import write_jsonl

    target = sys.argv[1] if len(sys.argv) > 1 else "data/smoke/chunks.jsonl"
    count = write_jsonl(target, chunk_rows())
    print(f"{count} chunks -> {target}")
