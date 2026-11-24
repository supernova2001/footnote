import { Suspense } from "react";
import { SavedDraft } from "@/components/saved-draft";
import { Spinner } from "@/components/ui";

export default function DraftPage({ params }: PageProps<"/write/[id]">) {
  return (
    <Suspense
      fallback={
        <div className="px-6 py-10">
          <Spinner label="Loading draft" />
        </div>
      }
    >
      {params.then(({ id }) => (
        <SavedDraft id={decodeURIComponent(id)} />
      ))}
    </Suspense>
  );
}
