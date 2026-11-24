// Fixtures for the mock API (scripts/mock-api.mjs).
//
// INVENTED PLACEHOLDER CONTENT FOR UI DEVELOPMENT. Not the real corpus and not real output.
// - Passages from public-domain books (The Souls of Black Folk, 1903; Narrative of the Life of
//   Frederick Douglass, 1845; Up from Slavery, 1901) are short excerpts of public-domain text.
// - Every other passage, summary, answer, ruling, draft and eval number below was written for this
//   mock and is marked "[Mock text]" where a reader could mistake it for a source.
// - No living person's data. The one "author" is fictional.

export const MOCK_NOTE = "[Mock text written for UI development.]";

const LICENSES = {
  "CC-BY-SA-4.0": {
    id: "CC-BY-SA-4.0",
    name: "Creative Commons Attribution-ShareAlike 4.0",
    url: "https://creativecommons.org/licenses/by-sa/4.0/",
  },
  "CC-BY-4.0": {
    id: "CC-BY-4.0",
    name: "Creative Commons Attribution 4.0",
    url: "https://creativecommons.org/licenses/by/4.0/",
  },
  "PD-US": {
    id: "PD-US",
    name: "Public domain in the United States",
    url: "https://www.gutenberg.org/policy/permission.html",
  },
};

function license(id, attribution) {
  return { ...LICENSES[id], attribution };
}

export const scholars = [
  {
    id: "w-e-b-du-bois",
    name: "W. E. B. Du Bois",
    born: 1868,
    died: 1963,
    summary: `Sociologist, historian and editor, and a founder of the NAACP. ${MOCK_NOTE}`,
    kind: "historical",
    document_count: 2,
    wikipedia_url: "https://en.wikipedia.org/wiki/W._E._B._Du_Bois",
  },
  {
    id: "frederick-douglass",
    name: "Frederick Douglass",
    born: 1818,
    died: 1895,
    summary: `Abolitionist, orator and writer who escaped slavery in 1838 and wrote three autobiographies. ${MOCK_NOTE}`,
    kind: "historical",
    document_count: 2,
    wikipedia_url: "https://en.wikipedia.org/wiki/Frederick_Douglass",
  },
  {
    id: "booker-t-washington",
    name: "Booker T. Washington",
    born: 1856,
    died: 1915,
    summary: `Educator and author; the first leader of the Tuskegee Institute. ${MOCK_NOTE}`,
    kind: "historical",
    document_count: 2,
    wikipedia_url: "https://en.wikipedia.org/wiki/Booker_T._Washington",
  },
  {
    id: "ida-b-wells",
    name: "Ida B. Wells",
    born: 1862,
    died: 1931,
    summary: `Investigative journalist and civil rights campaigner. ${MOCK_NOTE}`,
    kind: "historical",
    document_count: 1,
    wikipedia_url: "https://en.wikipedia.org/wiki/Ida_B._Wells",
  },
  {
    id: "placeholder-author",
    name: "A. Placeholder",
    born: null,
    died: null,
    summary: `A fictional author of a fictional open-access paper, standing in for a real CC BY paper's author. ${MOCK_NOTE}`,
    kind: "author",
    document_count: 1,
    wikipedia_url: null,
  },
];

export const documents = [
  {
    id: "wikipedia-w-e-b-du-bois",
    title: "W. E. B. Du Bois (Wikipedia)",
    kind: "biography",
    year: null,
    scholar_ids: ["w-e-b-du-bois"],
    source_url: "https://en.wikipedia.org/wiki/W._E._B._Du_Bois",
    license: license("CC-BY-SA-4.0", "From the Wikipedia article “W. E. B. Du Bois”, CC BY-SA 4.0."),
    chunk_count: 48,
    sections: ["Early life", "Education", "Career", "Legacy"],
  },
  {
    id: "gutenberg-408",
    title: "The Souls of Black Folk",
    kind: "work",
    year: 1903,
    scholar_ids: ["w-e-b-du-bois"],
    source_url: "https://www.gutenberg.org/ebooks/408",
    license: license("PD-US", "The Souls of Black Folk (1903), public domain in the United States."),
    chunk_count: 412,
    sections: ["The Forethought", "Of Our Spiritual Strivings", "Of Mr. Booker T. Washington and Others"],
  },
  {
    id: "wikipedia-frederick-douglass",
    title: "Frederick Douglass (Wikipedia)",
    kind: "biography",
    year: null,
    scholar_ids: ["frederick-douglass"],
    source_url: "https://en.wikipedia.org/wiki/Frederick_Douglass",
    license: license("CC-BY-SA-4.0", "From the Wikipedia article “Frederick Douglass”, CC BY-SA 4.0."),
    chunk_count: 61,
    sections: ["Early life", "Escape", "Abolitionism", "Later life"],
  },
  {
    id: "gutenberg-23",
    title: "Narrative of the Life of Frederick Douglass, an American Slave",
    kind: "work",
    year: 1845,
    scholar_ids: ["frederick-douglass"],
    source_url: "https://www.gutenberg.org/ebooks/23",
    license: license("PD-US", "Narrative of the Life of Frederick Douglass (1845), public domain in the United States."),
    chunk_count: 188,
    sections: ["Chapter VI", "Chapter VII", "Chapter X"],
  },
  {
    id: "wikipedia-booker-t-washington",
    title: "Booker T. Washington (Wikipedia)",
    kind: "biography",
    year: null,
    scholar_ids: ["booker-t-washington"],
    source_url: "https://en.wikipedia.org/wiki/Booker_T._Washington",
    license: license("CC-BY-SA-4.0", "From the Wikipedia article “Booker T. Washington”, CC BY-SA 4.0."),
    chunk_count: 44,
    sections: ["Early life", "Tuskegee", "Legacy"],
  },
  {
    id: "gutenberg-2376",
    title: "Up from Slavery: An Autobiography",
    kind: "work",
    year: 1901,
    scholar_ids: ["booker-t-washington"],
    source_url: "https://www.gutenberg.org/ebooks/2376",
    license: license("PD-US", "Up from Slavery (1901), public domain in the United States."),
    chunk_count: 301,
    sections: ["Chapter I", "Chapter II", "Chapter III"],
  },
  {
    id: "wikipedia-ida-b-wells",
    title: "Ida B. Wells (Wikipedia)",
    kind: "biography",
    year: null,
    scholar_ids: ["ida-b-wells"],
    source_url: "https://en.wikipedia.org/wiki/Ida_B._Wells",
    license: license("CC-BY-SA-4.0", "From the Wikipedia article “Ida B. Wells”, CC BY-SA 4.0."),
    chunk_count: 39,
    sections: ["Early life", "Journalism", "Legacy"],
  },
  {
    id: "pmc-PMC0000000",
    title: "Placeholder: reading practice in primary classrooms (fictional paper)",
    kind: "paper",
    year: 2024,
    scholar_ids: ["placeholder-author"],
    source_url: "https://europepmc.org/",
    license: license("CC-BY-4.0", "Fictional placeholder paper standing in for a CC BY 4.0 article."),
    chunk_count: 27,
    sections: ["Introduction", "Methods", "Results", "Discussion"],
  },
];

function passage(id, section, text, score = null) {
  const document_id = id.split(":")[0];
  const doc = documents.find((d) => d.id === document_id);
  return {
    id,
    document_id,
    title: doc.title,
    section,
    text,
    scholar_ids: doc.scholar_ids,
    license_id: doc.license.id,
    source_url: doc.source_url,
    attribution: doc.license.attribution,
    score,
  };
}

export const passages = [
  passage(
    "gutenberg-408:0002",
    "The Forethought",
    "Herein lie buried many things which if read with patience may show the strange meaning of being black here at the dawning of the Twentieth Century. This meaning is not without interest to you, Gentle Reader; for the problem of the Twentieth Century is the problem of the color-line.",
  ),
  passage(
    "gutenberg-408:0010",
    "Of Our Spiritual Strivings",
    "One ever feels his twoness,—an American, a Negro; two souls, two thoughts, two unreconciled strivings; two warring ideals in one dark body, whose dogged strength alone keeps it from being torn asunder.",
  ),
  passage(
    "wikipedia-w-e-b-du-bois:0001",
    "Early life",
    `Du Bois was born in Great Barrington, Massachusetts, in 1868. He studied at Fisk University and at Harvard, where he earned a doctorate in 1895. ${MOCK_NOTE}`,
  ),
  passage("gutenberg-23:0040", "Chapter VI", "From that moment, I understood the pathway from slavery to freedom."),
  passage("gutenberg-23:0055", "Chapter X", "You have seen how a man was made a slave; you shall see how a slave was made a man."),
  passage(
    "wikipedia-frederick-douglass:0001",
    "Early life",
    `Frederick Douglass was born into slavery in Maryland around 1818. He escaped in 1838 and became one of the best-known abolitionist speakers and writers of his time. ${MOCK_NOTE}`,
  ),
  passage(
    "gutenberg-2376:0012",
    "Chapter II",
    "I have learned that success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome while trying to succeed.",
  ),
  passage(
    "wikipedia-booker-t-washington:0003",
    "Tuskegee",
    `In 1881 Washington became the first leader of the new Tuskegee Normal School in Alabama. ${MOCK_NOTE}`,
  ),
  passage(
    "wikipedia-ida-b-wells:0002",
    "Journalism",
    `Wells co-owned and wrote for a Memphis newspaper, and in the 1890s published pamphlets documenting lynchings. ${MOCK_NOTE}`,
  ),
  passage(
    "pmc-PMC0000000:0004",
    "Results",
    `Fictional result: pupils who read aloud daily for ten weeks improved on a fluency measure. ${MOCK_NOTE}`,
  ),
];

export function findPassage(id) {
  return passages.find((p) => p.id === id) ?? null;
}

export function withScore(id, score) {
  return { ...findPassage(id), score };
}

// ── Judge rulings (invented) ────────────────────────────────────────────────

const J = "claude-sonnet-5-5 (mock)";
const supported = (evidence, rationale = "The passage states this directly.") => ({ label: "supported", evidence, rationale, judge: J });
const partial = (evidence, rationale) => ({ label: "partial", evidence, rationale, judge: J });
const unsupported = (rationale) => ({ label: "unsupported", evidence: "", rationale, judge: J });
const cite = (passage_id, cited_text) => ({ passage_id, cited_text });

// ── Ask answers ─────────────────────────────────────────────────────────────

export const answers = {
  dubois: {
    queries: ["Du Bois color-line problem of the twentieth century", "double consciousness twoness Souls of Black Folk"],
    passageIds: [
      ["gutenberg-408:0002", 0.912],
      ["gutenberg-408:0010", 0.874],
      ["wikipedia-w-e-b-du-bois:0001", 0.611],
    ],
    // First attempt: the judge fails two claims, the revision fixes one, the other is withheld.
    firstDraft:
      "Here is what the sources say. Du Bois names the color-line as the problem of the twentieth century in the opening of The Souls of Black Folk. He proves that double consciousness makes people stronger. He describes a “twoness”: feeling both American and Black, with two warring ideals in one body. He wrote the book while teaching at Atlanta University.",
    revisedDraft:
      "Here is what the sources say. Du Bois names the color-line as the problem of the twentieth century in the opening of The Souls of Black Folk. He describes a “twoness”: feeling both American and Black, with two warring ideals in one body. He presents that doubleness as a strain that only dogged strength withstands. He wrote the book while teaching at Atlanta University.",
    passes: [
      { checked: 4, failing: 2 },
      { checked: 4, failing: 1 },
    ],
    segments: [
      { text: "Here is what the sources say.", citations: [], verdict: null },
      {
        text: "Du Bois names the color-line as the problem of the twentieth century in the opening of The Souls of Black Folk.",
        citations: [cite("gutenberg-408:0002", "for the problem of the Twentieth Century is the problem of the color-line.")],
        verdict: supported("the problem of the Twentieth Century is the problem of the color-line"),
      },
      {
        text: "He describes a “twoness”: feeling both American and Black, with two warring ideals in one body.",
        citations: [cite("gutenberg-408:0010", "One ever feels his twoness,—an American, a Negro; two souls, two thoughts, two unreconciled strivings; two warring ideals in one dark body, whose dogged strength alone keeps it from being torn asunder.")],
        verdict: supported("two warring ideals in one dark body", "The passage describes this twoness in these terms."),
      },
      {
        text: "He presents that doubleness as a strain that only dogged strength withstands.",
        citations: [cite("gutenberg-408:0010", "One ever feels his twoness,—an American, a Negro; two souls, two thoughts, two unreconciled strivings; two warring ideals in one dark body, whose dogged strength alone keeps it from being torn asunder.")],
        verdict: partial(
          "whose dogged strength alone keeps it from being torn asunder",
          "The passage gives the strain and the strength, but “only” is a stronger claim than the wording supports.",
        ),
      },
    ],
    removed: [
      {
        text: "He wrote the book while teaching at Atlanta University.",
        citations: [cite("wikipedia-w-e-b-du-bois:0001", "He studied at Fisk University and at Harvard")],
        verdict: unsupported("The cited passage covers his education, not where he worked when he wrote the book."),
      },
    ],
    revised: true,
  },
  douglass: {
    queries: ["Frederick Douglass learning to read", "Douglass pathway from slavery to freedom"],
    passageIds: [
      ["gutenberg-23:0040", 0.903],
      ["wikipedia-frederick-douglass:0001", 0.722],
      ["gutenberg-23:0055", 0.688],
    ],
    firstDraft:
      "Douglass was born into slavery in Maryland and escaped in 1838. In his Narrative he marks the moment he understood the pathway from slavery to freedom. Later in the book he tells the reader they will see how a slave was made a man.",
    passes: [{ checked: 3, failing: 0 }],
    segments: [
      {
        text: "Douglass was born into slavery in Maryland and escaped in 1838.",
        citations: [cite("wikipedia-frederick-douglass:0001", "Frederick Douglass was born into slavery in Maryland around 1818. He escaped in 1838")],
        verdict: supported("He escaped in 1838", "The passage gives both the birth into slavery in Maryland and the escape in 1838."),
      },
      {
        text: "In his Narrative he marks the moment he understood the pathway from slavery to freedom.",
        citations: [cite("gutenberg-23:0040", "From that moment, I understood the pathway from slavery to freedom.")],
        verdict: supported("From that moment, I understood the pathway from slavery to freedom."),
      },
      {
        text: "Later in the book he tells the reader they will see how a slave was made a man.",
        citations: [cite("gutenberg-23:0055", "You have seen how a man was made a slave; you shall see how a slave was made a man.")],
        verdict: supported("you shall see how a slave was made a man"),
      },
    ],
    removed: [],
    revised: false,
  },
  washington: {
    queries: ["Booker T. Washington how success is measured"],
    passageIds: [
      ["gutenberg-2376:0012", 0.941],
      ["wikipedia-booker-t-washington:0003", 0.533],
    ],
    firstDraft:
      "In Up from Slavery, Washington says success is measured less by the position a person reaches than by the obstacles they overcome while trying to succeed. He believed this was the only fair measure.",
    passes: [{ checked: 2, failing: 0 }],
    segments: [
      {
        text: "In Up from Slavery, Washington says success is measured less by the position a person reaches than by the obstacles they overcome while trying to succeed.",
        citations: [cite("gutenberg-2376:0012", "I have learned that success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome while trying to succeed.")],
        verdict: supported("success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome"),
      },
      {
        text: "He believed this was the only fair measure.",
        citations: [cite("gutenberg-2376:0012", "I have learned that success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome while trying to succeed.")],
        verdict: partial(
          "not so much by the position",
          "“Not so much” weighs the two measures; the passage does not call obstacles the only fair one.",
        ),
      },
    ],
    removed: [],
    revised: false,
  },
  abstain: {
    queries: ["weather on Mars", "Mars climate"],
    passageIds: [
      ["pmc-PMC0000000:0004", 0.081],
      ["wikipedia-ida-b-wells:0002", 0.064],
    ],
    firstDraft: "None of the retrieved passages discusses this, so there is nothing to cite.",
    passes: [{ checked: 0, failing: 0 }],
    segments: [{ text: "None of the retrieved passages discusses this, so there is nothing to cite.", citations: [], verdict: null }],
    removed: [],
    revised: false,
  },
};

export function pickAnswer(question) {
  const q = question.toLowerCase();
  if (/(weather|mars|stock|football|recipe|bitcoin)/.test(q)) return "abstain";
  if (/(douglass|read)/.test(q)) return "douglass";
  if (/(washington|success|tuskegee)/.test(q)) return "washington";
  return "dubois";
}

// ── Drafts ──────────────────────────────────────────────────────────────────

export const draftStory = {
  title: "The Day Frederick Douglass Saw the Way to Freedom",
  sources: ["wikipedia-frederick-douglass:0001", "gutenberg-23:0040", "gutenberg-23:0055"],
  outline: [
    { heading: "Growing up enslaved", goal: "Where and when Douglass was born, and what slavery meant for him.", passage_ids: ["wikipedia-frederick-douglass:0001"] },
    { heading: "The pathway to freedom", goal: "The moment reading showed him a way out.", passage_ids: ["gutenberg-23:0040"] },
    { heading: "A promise to the reader", goal: "How he describes his own change in the Narrative.", passage_ids: ["gutenberg-23:0055"] },
  ],
  sections: [
    {
      first: [
        {
          text: "Frederick Douglass was born into slavery in Maryland, around the year 1818.",
          citations: [cite("wikipedia-frederick-douglass:0001", "Frederick Douglass was born into slavery in Maryland around 1818.")],
          verdict: supported("born into slavery in Maryland around 1818"),
        },
        {
          text: "When he was a young man, he escaped.",
          citations: [cite("wikipedia-frederick-douglass:0001", "He escaped in 1838")],
          verdict: supported("He escaped in 1838", "The passage gives the escape; 1838 fits “a young man” for someone born around 1818."),
        },
      ],
    },
    {
      // Attempt 1 has a claim the judge fails; attempt 2 names it and drops it.
      first: [
        {
          text: "One day, a kind teacher showed him the letters of the alphabet.",
          citations: [cite("gutenberg-23:0040", "From that moment, I understood the pathway from slavery to freedom.")],
          verdict: unsupported("The passage does not mention a teacher or the alphabet."),
        },
        {
          text: "Douglass later wrote that from that moment he understood the pathway from slavery to freedom.",
          citations: [cite("gutenberg-23:0040", "From that moment, I understood the pathway from slavery to freedom.")],
          verdict: supported("From that moment, I understood the pathway from slavery to freedom."),
        },
      ],
      second: [
        { text: "Something changed for Douglass when he was still a boy.", citations: [], verdict: null },
        {
          text: "He later wrote that from that moment he understood the pathway from slavery to freedom.",
          citations: [cite("gutenberg-23:0040", "From that moment, I understood the pathway from slavery to freedom.")],
          verdict: supported("From that moment, I understood the pathway from slavery to freedom."),
        },
      ],
      warning: "One sentence was dropped after redrafting: the judge found no passage about a teacher or the alphabet.",
    },
    {
      first: [
        {
          text: "In his book he made the reader a promise: “you shall see how a slave was made a man.”",
          citations: [cite("gutenberg-23:0055", "You have seen how a man was made a slave; you shall see how a slave was made a man.")],
          verdict: supported("you shall see how a slave was made a man"),
        },
      ],
    },
  ],
  readability: { grade_level: 4.6, target: "grade4", editor_note: `Fits a grade 4 reader; sentences average 12 words. ${MOCK_NOTE}` },
};

export const seededDraft = {
  id: "d-seed-washington",
  title: "How Booker T. Washington Measured Success",
  audience: "grade8",
  status: "published",
  created_at: "2026-11-10T15:04:00Z",
  updated_at: "2026-11-10T15:09:00Z",
  brief: "What Booker T. Washington said about measuring success, for eighth graders.",
  sections: [
    {
      heading: "Obstacles, not position",
      segments: answers.washington.segments.slice(0, 1),
      warnings: [],
    },
  ],
  readability: { grade_level: 8.2, target: "grade8", editor_note: `On target for grade 8. ${MOCK_NOTE}` },
  sources: [findPassage("gutenberg-2376:0012")],
};

// ── Evals (invented numbers) ────────────────────────────────────────────────

export const evalReport = {
  generated_at: "2026-11-18T12:00:00Z",
  retrieval: [
    { strategy: "bm25", n: 150, recall_at_5: 0.613, recall_at_10: 0.704, mrr: 0.512, ndcg_at_10: 0.561 },
    { strategy: "dense", n: 150, recall_at_5: 0.672, recall_at_10: 0.781, mrr: 0.548, ndcg_at_10: 0.602 },
    { strategy: "hybrid", n: 150, recall_at_5: 0.733, recall_at_10: 0.836, mrr: 0.601, ndcg_at_10: 0.655 },
    { strategy: "hybrid_rerank", n: 150, recall_at_5: 0.801, recall_at_10: 0.862, mrr: 0.694, ndcg_at_10: 0.731 },
  ],
  answers: [
    { config: "hybrid_rerank, no judge (mock)", n: 150, faithfulness: 0.861, citation_precision: 0.842, answer_relevance: 0.902, abstention_accuracy: 0.78, p50_latency_ms: 4100, cost_per_answer_usd: 0.021 },
    { config: "hybrid_rerank + judge + revise (mock)", n: 150, faithfulness: 0.957, citation_precision: 0.931, answer_relevance: 0.894, abstention_accuracy: 0.9, p50_latency_ms: 6900, cost_per_answer_usd: 0.034 },
  ],
  judge: [
    { judge: "qwen3-1.7b base (mock)", n: 200, macro_f1: 0.512, unsupported_recall: 0.47, json_valid_rate: 0.82, p50_latency_ms: 950, cost_per_1k_usd: 0 },
    { judge: "local:footnote-judge (mock)", n: 200, macro_f1: 0.844, unsupported_recall: 0.9, json_valid_rate: 0.995, p50_latency_ms: 980, cost_per_1k_usd: 0 },
    { judge: "claude-sonnet-5-5 (mock)", n: 200, macro_f1: 0.871, unsupported_recall: 0.92, json_valid_rate: 1, p50_latency_ms: 1900, cost_per_1k_usd: 2.4 },
  ],
};

export const emptyEvalReport = { generated_at: "2026-11-18T12:00:00Z", retrieval: [], answers: [], judge: [] };
