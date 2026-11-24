import { Suspense } from "react";
import { ScholarView } from "@/components/scholar-view";
import { Container, Spinner } from "@/components/ui";

// The id is only known at request time (no generateStaticParams: the corpus lives in the API),
// so with Cache Components the param is read inside a Suspense boundary.
export default function ScholarPage({ params }: PageProps<"/scholars/[id]">) {
  return (
    <Container>
      <Suspense
        fallback={
          <div className="py-10">
            <Spinner label="Loading scholar" />
          </div>
        }
      >
        {params.then(({ id }) => (
          <ScholarView id={decodeURIComponent(id)} />
        ))}
      </Suspense>
    </Container>
  );
}
