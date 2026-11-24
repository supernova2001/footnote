import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-rule">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          Footnote: a portfolio project in retrieval, agents and fine-tuning. Sources are public domain or openly
          licensed; see{" "}
          <Link href="/data" className="text-ink underline decoration-rule hover:decoration-accent">
            the data card
          </Link>
          .
        </p>
        <p className="font-mono text-xs">Wikipedia text is CC BY-SA 4.0 and keeps its licence here.</p>
      </div>
    </footer>
  );
}
