"""Footnote: a research assistant whose every sentence carries its source.

Agents (LangGraph + Claude): a RAG agent that answers with native citations and a grounding judge
on every claim; a writer agent that drafts cited articles with a human-in-the-loop review; a
research agent and a supervisor that reach their tools over MCP. A small open model, fine-tuned
with QLoRA, can stand in for the judge (finetune/).
"""

__version__ = "0.1.0"
