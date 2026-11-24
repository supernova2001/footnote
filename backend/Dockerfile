# Footnote API: FastAPI + LangGraph agents + MCP server (Streamable HTTP at /mcp).
FROM python:3.12-slim AS app

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY data/snapshot.json ./data/snapshot.json
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 footnote && chown -R footnote /app
USER footnote
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["footnote-api"]
