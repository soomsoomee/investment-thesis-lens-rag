# investment-thesis-lens-rag (Naive RAG 스터디 트랙) — Spec

Date: 2026-05-11
Track: Study (Track B)
Status: Implemented (2026-05-12)

> 본 문서는 본 repo의 설계 문서이다. 메인 프로젝트와 코드 의존성 없음. 인프라(Docker PostgreSQL+pgvector)는 공유한다.

---

## 1. 목적

스터디 과제(Naive RAG 구현 + 평가)를 수행하면서 결과물을 별도 GitHub repo로 공유한다. 메인 프로젝트(`iterate-to-wealth`)의 트랙 A에서 수집한 트랜스크립트를 도메인 데이터로 재활용하되, 코드 의존성은 두지 않는다. 스터디 종료 시 흔적 없이 제거 가능해야 한다.

산출물:
- 별도 GitHub repo `soom/investment-thesis-lens-rag` (Public)
- 로컬 경로: `~/Projects/investment-thesis-lens-rag/` (메인 repo와 sibling)
- 동작하는 Naive RAG 파이프라인 + 10개 평가 질문
- 스터디 참여자가 `git clone + uv sync + .env`만으로 재현 가능

비목적:
- 고급 RAG 기법(rerank, hybrid, multi-query 등) — 이번 phase에서는 의도적으로 미포함
- ragas 정량 평가 — 자리만 만들어두고 다음 phase
- 메인 프로젝트 통합 — 영구히 분리

---

## 2. 위치 결정

- **물리 위치**: `~/Projects/investment-thesis-lens-rag/` (sibling)
  - 이유: 메인 repo 안 nested git repo는 IDE/tooling에서 헷갈림. Sibling이 깔끔.
  - 메인 repo `.gitignore`의 `investment-thesis-lens-rag/` 엔트리는 그대로 둬도 무방 (방어적).
- **GitHub repo**: `soomsoomee/investment-thesis-lens-rag` Public.
  - 이유: 스터디 참여자가 link 하나로 접근. 트랜스크립트 raw 데이터는 repo에 커밋되지 않으므로 노출 위험 낮음 (.env로 DB만 가리킴).

---

## 3. Architecture & Data Flow

```
PostgreSQL (main repo's docker-compose, shared)
  └─ contents.transcript  ── SQLAlchemy ─→  LangChain Document objects
                                                   │
                                          RecursiveCharacterTextSplitter
                                          (chunk_size=600, chunk_overlap=100)
                                                   │
                                          sentence-transformers
                                          (BAAI/bge-m3 — 다국어 지원)
                                                   │
                                          pgvector (langchain-postgres PGVector)
                                          collection_name='study_naive_rag'
                                                   │
                                          PGVector.as_retriever(search_kwargs={'k': 4})
                                                   │
                                          ChatOpenAI(model='gpt-4o-mini')
                                                   │
                                                답변
```

핵심 결정:
- **데이터 소스**: 메인 DB의 `contents.transcript` 직접 조회. 메인 패키지(`src/invest_engine/`)는 import 하지 않는다. DB 연결 문자열은 `.env`의 `MAIN_DATABASE_URL`로 따로 둔다.
- **Vector store namespace 격리**: 메인 docker pgvector를 공유하되 LangChain PGVector의 `collection_name='study_naive_rag'`로 메인 프로젝트가 쓸 수 있는 다른 collection과 충돌 방지.
- **Retriever**: similarity search, `k=4`. Naive RAG라 rerank/hybrid 일체 없음.
- **LLM**: `gpt-4o-mini`. `OPENAI_API_KEY` 사용.

---

## 4. 폴더 구조

```
investment-thesis-lens-rag/
├── .gitignore
├── .env.example                  # 환경변수 템플릿 (커밋)
├── .env                          # 실제 키 (gitignored)
├── README.md                     # 스터디 참여자용 setup 가이드
├── pyproject.toml                # uv 관리, Python 3.11
├── data/
│   └── .gitkeep                  # 비어있음. PDF 추가 시 여기로.
├── docs/
│   ├── design.md                 # 본 spec의 사본 (스터디 참여자용)
│   └── eval_questions.md         # 평가 질문 10개
├── notebooks/
│   └── 01_naive_rag_walkthrough.ipynb   # 단계별 실행 노트북
└── src/
    └── naive_rag/
        ├── __init__.py
        ├── config.py             # env 로드, 모델/collection 이름 상수
        ├── loader.py             # PostgreSQL contents → LangChain Document
        ├── splitter.py           # RecursiveCharacterTextSplitter wrapper
        ├── embeddings.py         # HuggingFaceEmbeddings(bge-m3) wrapper
        ├── vectorstore.py        # PGVector setup, ingest CLI
        ├── retriever.py          # retriever build
        ├── chain.py              # LCEL chain (retriever → prompt → llm)
        └── eval.py               # ragas 평가 (Phase 2, 자리만)
```

설계 원칙:
- 각 단계를 파일 단위로 분리. 스터디 동료가 "어느 단계를 보는 중"을 한눈에 파악.
- `notebooks/01_*.ipynb`는 위 모듈을 import해서 단계별로 실행/시각화. 학습용 흐름.
- `data/`는 placeholder. 본 spec 범위에서는 비어있다.

---

## 5. Tech Stack

| 단계 | 라이브러리 | 비고 |
|---|---|---|
| DB 조회 | `sqlalchemy`, `psycopg[binary]` | 메인 docker postgres, `contents.transcript` SELECT |
| Splitter | `langchain-text-splitters` | RecursiveCharacterTextSplitter, chunk 600 / overlap 100 |
| Embedding | `sentence-transformers`, `langchain-huggingface` | `BAAI/bge-m3` |
| Vector store | `langchain-postgres` (PGVector v2) | collection `study_naive_rag` |
| LLM | `langchain-openai` | `gpt-4o-mini` |
| Eval (자리만) | `ragas` | 다음 phase |
| 패키지 매니저 | `uv` | Python 3.11, 메인과 일관성 |

---

## 6. Setup Commands

```bash
# 1. GitHub repo 생성
cd ~/Projects
gh repo create investment-thesis-lens-rag --public \
  --description "Naive RAG study track — investment domain transcripts"

# 2. 로컬 폴더 + uv init
gh repo clone soomsoomee/investment-thesis-lens-rag
cd investment-thesis-lens-rag
uv init --python 3.11
uv add langchain langchain-community langchain-openai langchain-huggingface \
       langchain-postgres langchain-text-splitters \
       sentence-transformers sqlalchemy "psycopg[binary]" \
       python-dotenv ragas
uv add --dev jupyter ipykernel

# 3. 폴더 스캐폴딩
mkdir -p data docs notebooks src/naive_rag
touch data/.gitkeep src/naive_rag/__init__.py

# 4. README, .env.example, .gitignore 작성 (writing-plans 단계)

# 5. 첫 커밋 + push
git add .
git commit -m "scaffold naive RAG study track"
git push -u origin main
```

스터디 참여자 재현 명령:
```bash
gh repo clone soomsoomee/investment-thesis-lens-rag
cd investment-thesis-lens-rag
cp .env.example .env  # 값 채움
uv sync
uv run python -m naive_rag.vectorstore ingest  # 최초 1회 임베딩 ingest
uv run jupyter lab notebooks/
```

---

## 7. 평가 질문 (10개)

도메인: 투자 인플루언서 트랜스크립트 (메인 DB `contents.transcript`).

```
Q1.  최근 트랜스크립트에서 언급된 주요 매수 근거는 무엇인가?
Q2.  자주 인용되는 매크로 지표는 무엇인가?
Q3.  반증 조건(thesis가 깨지는 조건)으로 제시된 내용은?
Q4.  특정 종목에 대한 방향성(상승/하락)과 시간 지평은?
Q5.  최근 콘텐츠에서 위험 요소로 언급된 것은?
Q6.  특정 섹터(반도체/2차전지/AI 등)에 대한 시각은?
Q7.  두 인플루언서의 의견이 갈리는 지점은?
Q8.  동일 종목에 대해 서로 다른 근거를 제시한 사례는?
Q9.  미국 금리/연준 정책에 대한 견해는?
Q10. 환율(원/달러)에 대한 언급과 그 영향은?
```

평가 방식:
- Phase 1 (본 spec 범위): 정성 평가. 노트북에서 질문별 답변/근거 chunk 출력 확인.
- Phase 2 (다음): ragas 정량 평가 (faithfulness, context precision/recall, answer relevancy).

---

## 8. Deviation log (과제 → 본 spec)

| 과제 | 본 spec | 이유 |
|---|---|---|
| `pip install langchain-chroma` | 미설치, `langchain-postgres` 사용 | 메인 docker pgvector 공유 결정 |
| 데이터: PDF/문서/논문 등 | PostgreSQL transcript | 트랙 A 결과 재활용 (선행 spec 계승) |
| Embedding: `sentence-transformers` (generic) | `BAAI/bge-m3` 구체화 | 한국어 트랜스크립트 대응 |
| LLM: 미지정 | `gpt-4o-mini` | 사용자 지정, 비용 효율 |
| 구조: `data/ docs/ notebooks/ src/` (flat src) | `src/naive_rag/` 패키지 + 단계별 모듈 분리 | 각 RAG 단계 가시화, 노트북에서 import 깔끔 |

---

## 9. 메인 프로젝트와의 격리 원칙

| 항목 | 격리 정책 |
|---|---|
| Python import | 메인 패키지(`invest_engine`) import 금지. SQLAlchemy로 raw SQL만. |
| DB | 같은 docker postgres 공유 OK. 단 collection_name namespace 분리. |
| 환경변수 | 별도 `.env`. 메인 `.env` 재사용 X. `MAIN_DATABASE_URL`로 명시. |
| Docker | 메인의 docker-compose 그대로 사용 (별도 docker-compose 생성 X). |
| Spec/Plan 위치 | `docs/superpowers/study/`(메인 repo, gitignored). |
| 종료 시 | 새 repo 폴더 + study/ spec 모두 삭제 가능. 메인 repo에 흔적 없음. |

---

## 10. Out of scope (다음 phase)

- ragas 정량 평가 구현
- 고급 RAG (rerank, hybrid search, multi-query)
- 다중 LLM 비교 (Claude/Gemini)
- PDF/논문 데이터 추가
- Streamlit/Gradio UI

---

## 11. Open items

- 메인 DB `contents.transcript`에 데이터가 실제로 있는지 ingest 시점에 확인 필요 (1건 이상).
- bge-m3 모델 첫 다운로드 시 디스크/RAM 사용량 확인 (≈2GB).
- PGVector collection `study_naive_rag`가 메인 프로젝트에서 사용되지 않는지 grep으로 확인.

---

## 12. Next step

본 spec 사용자 리뷰 → writing-plans 스킬로 task 단위 구현 plan 작성.
