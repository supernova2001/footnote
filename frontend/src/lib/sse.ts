// Server-sent events over a POST. EventSource only does GET, so the streaming
// routes (ask, drafts, review, chat) are read with fetch + a ReadableStream.

export type RawEvent = { event: string; data: unknown };

/** Parse one SSE frame ("event: x\ndata: {...}"). Returns null for comments/keep-alives. */
export function parseFrame(frame: string): RawEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  const text = data.join("\n");
  try {
    return { event, data: JSON.parse(text) };
  } catch {
    return { event, data: text };
  }
}

/** Yield every event from a text/event-stream response body until it ends. */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<RawEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line; accept \n\n and \r\n\r\n.
      let match: RegExpExecArray | null;
      const separator = /\r?\n\r?\n/g;
      let last = 0;
      while ((match = separator.exec(buffer)) !== null) {
        const frame = buffer.slice(last, match.index);
        last = match.index + match[0].length;
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
      buffer = buffer.slice(last);
    }
    buffer += decoder.decode();
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}
