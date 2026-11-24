import type { Metadata } from "next";
import { ScholarList } from "@/components/scholar-list";
import { Container, PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Scholars" };

export default function ScholarsPage() {
  return (
    <Container>
      <PageHeader eyebrow="Corpus" title="Scholars">
        Historical scholars whose books are in the public domain, and authors of openly licensed papers. Each has a
        Wikipedia biography or a paper of their own as a source.
      </PageHeader>
      <ScholarList />
    </Container>
  );
}
