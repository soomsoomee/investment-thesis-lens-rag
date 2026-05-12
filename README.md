# investment-thesis-lens-rag

투자 인플루언서들의 YouTube 트랜스크립트를 기반으로, **인플루언서 페르소나별 RAG**를 만드는 스터디 트랙. 같은 질문이라도 페르소나마다 그 사람의 관점·근거·톤으로 답하게 하는 게 최종 목표.

## 단계별 로드맵

| Phase | 목표 | 상태 |
|---|---|---|
| Phase 1 | **Naive RAG** — 전체 트랜스크립트 통합 검색 → gpt-4o-mini 답변 | 완료 (2026-05-12) |
| Phase 2 | **Ragas 정량 평가** — 4 metric으로 Phase 1 품질 측정 | 진행 중 (reference 작성 단계) |
| Phase 3 | **페르소나 분기 RAG** — 페르소나별 retrieval 필터 + 페르소나 페르소나 system prompt로 톤/관점 분리 | 예정 |
| Phase 4+ | 페르소나간 의견 대비, 인용 추적, 시점 가중 등 | 추후 |

본 repo의 현재 코드는 Phase 1+2 까지를 다룬다. Phase 3부터는 retriever와 chain을 페르소나 분기 가능하게 확장한다.

## 셋업

1. 메인 프로젝트(`iterate-to-wealth`)의 docker postgres가 떠 있어야 한다 (트랜스크립트 데이터 + pgvector):
   ```bash
   cd ../iterate-to-wealth && docker compose up -d postgres
   ```

2. Clone + 의존성 설치:
   ```bash
   gh repo clone soomsoomee/investment-thesis-lens-rag
   cd investment-thesis-lens-rag
   cp .env.example .env   # OPENAI_API_KEY 채우기
   uv sync
   ```

3. pgvector에 트랜스크립트 ingest:
   ```bash
   uv run python -m naive_rag.vectorstore ingest
   ```
   첫 실행 시 `BAAI/bge-m3` 임베딩 모델(~2.3 GB)을 `~/.cache/huggingface/`로 다운받는다. 몇 분 걸린다. 다음 실행부턴 캐시 사용.

4. 노트북 열기:
   ```bash
   uv run jupyter lab notebooks/01_naive_rag_walkthrough.ipynb
   ```

## 현재 (Phase 1) 아키텍처

```
PostgreSQL contents.transcript
        │  (SQLAlchemy)
        ▼
LangChain Documents (메타: content_id, title, channel, published_at)
        │  RecursiveCharacterTextSplitter (chunk 600 / overlap 100)
        ▼
청크 Documents
        │  BAAI/bge-m3 임베딩 (다국어, 한국어 대응)
        ▼
pgvector (collection_name='study_naive_rag')
        │  similarity_search(k=4)
        ▼
retrieved chunks
        │  LCEL chain: 컨텍스트 포맷 → 프롬프트 → ChatOpenAI(gpt-4o-mini, temp=0)
        ▼
한국어 답변
```

## 폴더 구조

```
src/naive_rag/
├── config.py       # env 로드 + 모든 상수 (모델명, chunk size, k 등)
├── loader.py       # PostgreSQL → LangChain Document
├── splitter.py     # RecursiveCharacterTextSplitter wrapper
├── embeddings.py   # bge-m3 HuggingFaceEmbeddings wrapper
├── vectorstore.py  # PGVector setup + `ingest` CLI
├── retriever.py    # vs.as_retriever(k=4)
└── chain.py        # retriever → prompt → LLM 체인

notebooks/
└── 01_naive_rag_walkthrough.ipynb   # 단계별 실행 + ragas 평가

docs/
├── design.md             # 본 repo의 설계 spec
└── eval_questions.md     # 10개 평가 질문 세트

tests/                    # 각 모듈별 pytest (integration 마커 = 실제 DB/모델/OpenAI 호출)
```

## 평가 (Phase 2 — Ragas)

노트북 마지막 셀에서 4개 reference-free + reference-required metric으로 평가:

| Metric | 측정 |
|---|---|
| Faithfulness | 답변이 retrieved context에서 유도되는가 |
| AnswerRelevancy | 답변이 질문과 관련 있는가 |
| LLMContextPrecisionWithoutReference | retrieved 순서가 관련도순인가 |
| ContextEntityRecall | reference의 핵심 entity가 contexts에 들어있는가 |

`eval_data` 리스트의 `reference` 슬롯 10개를 채운 뒤 셀 실행. Judge LLM은 `gpt-4o-mini`, judge embeddings는 `bge-m3`.

## Phase 3 예정 사항 (페르소나 분기 RAG)

- `personas` 테이블 기반 필터 retrieval (collection metadata에 `channel` 이미 포함 → `PGVector.as_retriever(search_kwargs={"filter": {"channel": "소수몽키"}})` 식으로 분기)
- 페르소나별 system prompt — 그 사람의 평소 화법/근거 스타일을 few-shot으로 주입
- "이 질문에 6명 페르소나가 어떻게 답할까" 한 번에 보여주는 비교 뷰
- 페르소나 간 의견 차이 자동 추출

코드 변경 포인트가 모듈로 잘 분리돼 있어서 (retriever / chain) Phase 3은 두 모듈 확장 정도로 끝날 예정.

## 메인 프로젝트와의 관계

본 repo는 스터디 트랙이라 메인 프로젝트(`iterate-to-wealth`, 비공개)와 **코드 의존성 없다**:
- 메인 패키지(`invest_engine`) import 안 함
- 메인 docker postgres + pgvector는 인프라로만 공유
- `collection_name='study_naive_rag'`로 namespace 격리

스터디 종료 시 본 repo 폴더 통째로 삭제 가능, 메인 프로젝트엔 흔적 안 남는다.
