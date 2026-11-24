import Link from "next/link";
import { Container } from "@/components/ui";

export default function NotFound() {
  return (
    <Container className="py-16">
      <p className="font-mono text-xs uppercase tracking-[0.14em] text-muted">404</p>
      <h1 className="mt-1 font-serif text-3xl font-semibold">No such page</h1>
      <p className="mt-2 text-muted">
        This page has no source.{" "}
        <Link href="/" className="text-ink underline decoration-rule hover:decoration-accent">
          Back to the front page
        </Link>
        .
      </p>
    </Container>
  );
}
