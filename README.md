# investment-thesis-lens-rag

Naive RAG study track over investment-domain YouTube transcripts.

## Setup

1. Ensure main project (`iterate-to-wealth`) docker postgres is running:
   ```bash
   cd ../iterate-to-wealth && docker compose up -d postgres
   ```
2. Clone and install:
   ```bash
   gh repo clone soomsoomee/investment-thesis-lens-rag
   cd investment-thesis-lens-rag
   cp .env.example .env  # fill OPENAI_API_KEY
   uv sync
   ```
3. Ingest transcripts into pgvector:
   ```bash
   uv run python -m naive_rag.vectorstore ingest
   ```
4. Open the walkthrough notebook:
   ```bash
   uv run jupyter lab notebooks/01_naive_rag_walkthrough.ipynb
   ```

## Architecture

PostgreSQL transcripts -> SQLAlchemy -> LangChain Documents -> RecursiveCharacterTextSplitter (chunk 600 / overlap 100) -> `BAAI/bge-m3` embeddings -> pgvector (`collection_name='study_naive_rag'`) -> retriever (k=4) -> `gpt-4o-mini`.

See `docs/design.md` for the full spec.
