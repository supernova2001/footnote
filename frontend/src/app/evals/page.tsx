import type { Metadata } from "next";
import { EvalTables } from "@/components/eval-tables";
import { Container, PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Evals" };

export default function EvalsPage() {
  return (
    <Container>
      <PageHeader eyebrow="Measurement" title="Evaluations">
        Every design choice is measured: retrieval strategies against labelled questions, answers for faithfulness and
        citations, and the fine-tuned judge against a frontier model.
      </PageHeader>
      <EvalTables />
    </Container>
  );
}
