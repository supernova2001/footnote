import type { Metadata } from "next";
import { Suspense } from "react";
import { Spinner } from "@/components/ui";
import { WriterWorkspace } from "@/components/writer-workspace";

export const metadata: Metadata = { title: "Write" };

export default function WritePage() {
  return (
    // The workspace reads ?scholar= and ?document=, which are only known in the browser.
    <Suspense
      fallback={
        <div className="px-6 py-10">
          <Spinner label="Opening the writer" />
        </div>
      }
    >
      <WriterWorkspace />
    </Suspense>
  );
}
