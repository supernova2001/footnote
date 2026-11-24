import type { Metadata } from "next";
import { ChatView } from "@/components/chat-view";
import { Container, PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Chat" };

export default function ChatPage() {
  return (
    <Container>
      <PageHeader eyebrow="Supervisor agent · MCP" title="Chat">
        A tool-using agent that calls the other agents as tools and reaches the corpus through the Footnote MCP server.
      </PageHeader>
      <ChatView />
    </Container>
  );
}
