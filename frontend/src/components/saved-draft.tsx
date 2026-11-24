"use client";

import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Button, EmptyState, Notice, Spinner } from "./ui";
import { WriterWorkspace } from "./writer-workspace";

/** A saved draft, opened in the writer's workspace: awaiting review, published, or failed. */
export function SavedDraft({ id }: { id: string }) {
  const loaded = useApi(`draft:${id}`, async (s) => {
    try {
      return await api.draft(id, s);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  });

  if (loaded.status === "loading") {
    return (
      <div className="px-6 py-10">
        <Spinner label="Loading draft" />
      </div>
    );
  }
  if (loaded.status === "error") {
    return (
      <div className="mx-auto max-w-xl space-y-3 px-6 py-10">
        <Notice tone="bad">{loaded.error}</Notice>
        <Button variant="secondary" onClick={loaded.reload}>
          Try again
        </Button>
      </div>
    );
  }
  if (!loaded.data) {
    return (
      <div className="mx-auto max-w-xl px-6 py-10">
        <EmptyState title="No such draft">
          <Link href="/write" className="underline">
            Start a new story
          </Link>
        </EmptyState>
      </div>
    );
  }
  return (
    <>
      <title>{`${loaded.data.title ?? "Draft"} · Footnote`}</title>
      <WriterWorkspace key={loaded.data.id} initialDraft={loaded.data} />
    </>
  );
}
