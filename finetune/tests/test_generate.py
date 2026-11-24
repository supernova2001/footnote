"""Model-written claims: the checks, and the OpenAI-compatible client against a local stub server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from footnote_ft.dataset import BuildConfig, build_examples
from footnote_ft.generate import OpenAICompatibleGenerator, parse_generation, validate_generated

ANCHOR = "She was elected secretary of the Harlow Reading Society in 1911, which then had about 40 members."
PASSAGE = ANCHOR + " Her letters suggest that she often paid for new books herself."


def good() -> dict:
    return {
        "paraphrase": "In 1911 the Harlow Reading Society, then about 40 members strong, elected her its secretary.",
        "partial": {
            "claim": "She was elected secretary of the Harlow Reading Society in 1911 by a single vote.",
            "added": "by a single vote",
        },
        "contradiction": {
            "claim": "She lost the election for secretary of the Harlow Reading Society in 1911.",
            "changed": "elected -> lost",
        },
    }


def test_valid_variants_are_kept_with_their_labels():
    kept = validate_generated(good(), ANCHOR, PASSAGE)
    assert [(k.method, k.label) for k in kept] == [
        ("model_paraphrase", "supported"),
        ("model_partial", "partial"),
        ("model_contradiction", "unsupported"),
    ]


def test_a_paraphrase_that_adds_a_number_or_name_is_rejected():
    for bad in (
        "In 1911 the Harlow Reading Society, with 75 members, elected her its secretary.",
        "In 1911 Marcus Bell saw the Harlow Reading Society elect her its secretary.",
    ):
        kept = validate_generated({**good(), "paraphrase": bad}, ANCHOR, PASSAGE)
        assert "model_paraphrase" not in [k.method for k in kept]


def test_an_added_detail_the_passage_contains_is_rejected():
    data = {
        **good(),
        "partial": {
            "claim": "She was elected secretary in 1911 and often paid for new books.",
            "added": "often paid for new books",
        },
    }
    assert "model_partial" not in [k.method for k in validate_generated(data, ANCHOR, PASSAGE)]


def test_parse_generation_handles_fenced_replies():
    assert parse_generation("```json\n" + json.dumps(good()) + "\n```")["paraphrase"]
    assert parse_generation("no json here") is None


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system"
        sentence = body["messages"][1]["content"].split("<sentence>\n")[1].split("\n</sentence>")[0]
        data = {
            "paraphrase": sentence,  # identical: must be rejected
            "partial": {"claim": sentence.rstrip(".") + " by a narrow margin.", "added": "by a narrow margin"},
            "contradiction": {"claim": "It is false that " + sentence[0].lower() + sentence[1:], "changed": "negated"},
        }
        payload = json.dumps({"choices": [{"message": {"content": json.dumps(data)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_openai_compatible_generator_end_to_end(chunks, doc_split):
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        gen = OpenAICompatibleGenerator(f"http://127.0.0.1:{server.server_port}/v1", "stub-model", workers=4)
        examples, rejected = build_examples(
            chunks[:6], {k: "train" for k in doc_split}, BuildConfig(seed=1), generator=gen, rules=False
        )
    finally:
        server.shutdown()
    methods = {e.method for e in examples}
    assert methods == {"model_partial", "model_contradiction"}
    assert all(e.source == "model:stub-model" for e in examples)
    assert all(e.extra.get("generation_version") for e in examples)
