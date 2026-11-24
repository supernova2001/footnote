import { AskChat } from "@/components/ask-chat";

// The front page is the chat: ask a question, get an answer whose every sentence is cited and checked.
const EXAMPLES = [
  "What did Ida B. Wells write about the press?",
  "How did Carter G. Woodson describe schooling for enslaved people before 1861?",
  "Why did Ida B. Wells publish The Red Record?",
  "What does the corpus say about the weather on Mars?",
];

export default function HomePage() {
  return <AskChat examples={EXAMPLES} />;
}
