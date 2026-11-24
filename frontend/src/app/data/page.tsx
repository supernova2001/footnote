import type { Metadata } from "next";
import { CorpusCounts } from "@/components/health-banner";
import { LicenseBadge } from "@/components/license-badge";
import { Container, PageHeader } from "@/components/ui";
import type { LicenseId } from "@/lib/types";

export const metadata: Metadata = { title: "Data card" };

const SOURCES: { name: string; what: string; licenses: LicenseId[]; how: string }[] = [
  {
    name: "Wikipedia",
    what: "A biography of each historical scholar: the lead section and the article’s sections.",
    licenses: ["CC-BY-SA-4.0"],
    how: "MediaWiki API with an identified client; the revision fetched is recorded with the text.",
  },
  {
    name: "Project Gutenberg",
    what: "Books by those scholars that were published in the United States in 1930 or earlier.",
    licenses: ["PD-US"],
    how: "Plain-text files from a Gutenberg mirror, not the website, which asks automated clients to use mirrors.",
  },
  {
    name: "Europe PMC",
    what: "Open-access research papers, for the writer agent to turn into stories.",
    licenses: ["CC-BY-4.0", "CC0-1.0"],
    how: "Europe PMC REST API. The licence field is checked before anything is stored; NC and ND licences are refused.",
  },
];

export default function DataPage() {
  return (
    <Container>
      <PageHeader eyebrow="Data card" title="What Footnote reads, and on what terms">
        A small corpus, chosen so that every passage can be shown, cited and adapted lawfully. The licence travels with
        each passage from ingestion to the footnote a reader clicks.
      </PageHeader>

      <div className="space-y-12">
        <section aria-labelledby="counts">
          <h2 id="counts" className="mb-3 font-serif text-2xl font-semibold">
            Right now
          </h2>
          <CorpusCounts />
        </section>

        <section aria-labelledby="sources">
          <h2 id="sources" className="mb-3 font-serif text-2xl font-semibold">
            Sources
          </h2>
          <ul className="grid gap-4 md:grid-cols-3">
            {SOURCES.map((s) => (
              <li key={s.name} className="flex flex-col gap-2 rounded-lg border border-rule bg-surface p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-serif text-lg font-semibold">{s.name}</h3>
                  <span className="flex flex-wrap gap-1">
                    {s.licenses.map((l) => (
                      <LicenseBadge key={l} id={l} />
                    ))}
                  </span>
                </div>
                <p className="text-sm">{s.what}</p>
                <p className="text-xs text-muted">{s.how}</p>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="kept" className="grid gap-8 md:grid-cols-2">
          <div>
            <h2 id="kept" className="mb-3 font-serif text-2xl font-semibold">
              What is kept
            </h2>
            <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed marker:text-accent">
              <li>The text of each source, with its URL, licence, retrieval date and a content hash.</li>
              <li>Passages cut from that text, each carrying its document, section, character offsets and licence.</li>
              <li>An attribution line beside every passage a reader sees, and ShareAlike terms on text derived from Wikipedia.</li>
              <li>Drafts and the judge’s rulings, in the project’s own MongoDB, so they can be reviewed and evaluated.</li>
            </ul>
          </div>
          <div>
            <h2 className="mb-3 font-serif text-2xl font-semibold">What is not</h2>
            <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed marker:text-accent">
              <li>No private data about living people: no emails, addresses, photos, ratings or social media.</li>
              <li>No scraped websites, and nothing fetched by getting around robots.txt or a site’s terms.</li>
              <li>No paywalled text and no non-commercial or no-derivatives licences.</li>
              <li>Paper authors appear only as their papers name them.</li>
            </ul>
          </div>
        </section>

        <section aria-labelledby="judge-data">
          <h2 id="judge-data" className="mb-3 font-serif text-2xl font-semibold">
            Training data for the judge
          </h2>
          <p className="max-w-3xl text-sm leading-relaxed">
            The fine-tuned judge learns from claims built out of this corpus: true statements about a passage, and copies
            changed in known ways (a wrong number, a swapped name, a claim stated more certainly than the source does). The
            label is known from how each copy was made. The test set is labelled by hand, and the model is scored on it with
            the same prompt a frontier-model judge receives.
          </p>
        </section>
      </div>
    </Container>
  );
}
