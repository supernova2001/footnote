import { permanentRedirect } from "next/navigation";

// The chat moved to the front page; old /ask links land there.
export default function AskPage() {
  permanentRedirect("/");
}
