# Week 5 — Retrieval 전략 Ablation 회고

> 산출 노트북: [week5_retrieval_ablation.ipynb](../notebooks/week5_retrieval_ablation.ipynb)
> 선행 회고: [week4_retrospective.md](week4_retrospective.md) (청킹 우승 = B-large 1500/300)

---

## 1. 통제 변수 (무엇을 고정했나)

4주차에서 청킹을 끝냈으므로, 5주차는 **retrieval 방식만** 바꿔 비교한다. 나머지는 전부 고정:

| 항목 | 고정 값 |
|---|---|
| 데이터 | 소수몽키 5건 (`WEEK4_CONTENT_IDS`), 68 chunk |
| 청킹 | RecursiveCharacterTextSplitter, **1500/300, baseline separators** (= 4주차 우승 B-large) |
| collection | `study_advanced_retrieval` (B-large 청크 적재) |
| 임베딩 | `BAAI/bge-m3`, `normalize_embeddings=True` |
| LLM | `gpt-4o-mini`, temperature=0 |
| 평가셋 | `WEEK4_EVAL_DATA` 10문항 + reference |
| RAGAS | Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, ContextRecall |
| 최종 k | 4 (4 strategy 공통) |

**바뀐 것은 retriever 하나뿐.** 청크 수가 4주차 B-large(68)와 정확히 일치해 동일 청크 위 비교임을 확인했다.

---

## 2. 4 Retriever 비교 표

| strategy | retriever | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | 평균 latency(s) |
|---|---|---|---|---|---|---|
| **Dense** | PGVector (bge-m3) | 0.9917 | **0.6858** | 0.9444 | **0.9250** | 0.42 |
| BM25 | rank-bm25 + kiwipiepy | 0.9097 | 0.6203 | 0.9639 | 0.8500 | **0.04** |
| Hybrid | Ensemble(dense+bm25, 0.5/0.5) | 0.9909 | 0.6288 | 0.9444 | 0.9167 | 0.08 |
| Hybrid+Rerank | Ensemble→bge-reranker-v2-m3 | 0.9784 | 0.6282 | **0.9917** | 0.8917 | 3.77 |

**RAGAS 4지표 평균 랭킹:** Dense 0.8867 > Hybrid+Rerank 0.8725 > Hybrid 0.8702 > BM25 0.8360

> latency는 질문당 `retriever.invoke` 단일 측정이라 거칠다(첫 측정인 Dense는 cold-start 포함). 견고한 신호는 **Hybrid+Rerank ≈ 3.8s** — CPU cross-encoder가 12개 후보를 재정렬하는 비용이다. dense/bm25/hybrid는 모두 sub-second.

---

## 3. 선택한 retrieval 전략 + 데이터 근거

**최종 선택: Dense (PGVector + bge-m3, k=4).** 6주차 Agentic RAG의 baseline retriever로 사용한다.

근거:
1. **RAGAS 4지표 평균 1위(0.887)** 이면서 **AnswerRelevancy(0.686)·ContextRecall(0.925) 단독 최고.** 답에 필요한 정보를 가장 잘 끌어오고(recall), 답변 적합성도 가장 높다.
2. **BM25 단독은 최하위(0.836).** 한국어 형태소 토크나이즈(kiwipiepy)를 넣었는데도 lexical 매치만으로는 부족했다. 이 코퍼스는 같은 종목을 여러 표현("삼총사", "CPU 수혜주", "인텔·AMD·ARM")으로 말해 **의미 매치가 lexical 매치보다 유리**하다. 다만 ContextPrecision은 BM25가 0.964로 dense보다 높다 — 정확 키워드가 들어간 chunk만 좁게 잡기 때문.
3. **Hybrid는 dense와 사실상 동급(0.870 vs 0.887).** BM25를 0.5 섞었더니 AnswerRelevancy·Recall이 오히려 살짝 내려갔다. 약한 retriever(BM25)를 절반 가중으로 섞는 게 강한 retriever(dense)를 끌어내린 셈. **가중치 튜닝 없이는 hybrid가 dense를 못 이긴다.**
4. **Hybrid+Rerank는 ContextPrecision만 우승(0.992).** cross-encoder가 노이즈 chunk를 잘 걷어내 "검색된 chunk가 질문에 직접 도움 되는 비율"은 최고다. 그러나 Recall(0.892)·Relevancy(0.628)는 dense보다 낮고(재정렬이 답에 필요한 chunk를 4개 밖으로 밀어내기도 함), 무엇보다 **latency가 ~9배(0.42→3.77s).**

**핵심 발견:** 작고(68 chunk) 깨끗하며 의미가 풍부한 한국어 코퍼스 + 강력한 다국어 임베딩(bge-m3) 조합에서는 **dense가 이미 품질 상한에 근접**해 있다. BM25·Hybrid·Rerank 같은 추가 장치는 (이 데이터에서는) 정밀도를 조금 사거나 latency를 크게 더할 뿐, 종합 품질을 끌어올리지 못했다. "점수가 안 올라도 괜찮다, 왜 그런지가 핵심"이라는 과제 취지에 정확히 해당하는 결과.

---

## 4. accuracy ↔ latency trade-off 관찰

- **ContextPrecision만 보면 Hybrid+Rerank가 정답**(0.944→0.992, +0.05). 노이즈 chunk를 걷어내는 능력은 분명하다.
- 그러나 그 +0.05를 **질문당 ~3.3초 추가 비용**으로 산다. 10문항이면 ~33초, 운영에서는 사용자 체감 지연으로 직결된다.
- dense는 precision이 0.944로 rerank보다 낮지만 recall·relevancy가 더 높고 sub-second다. **"정밀도 한 끗을 위해 10배 느려질 가치가 있는가?"** 가 6주차의 출발 질문이 된다.
- 6주차 가설: rerank를 **항상 켜는** 대신, **grade_documents가 노이즈를 감지한 질문에서만** 재검색/정밀화를 트리거하면, rerank의 precision 이득을 **선택적으로(=평균 latency를 덜 내고)** 회수할 수 있지 않을까?

---

## 5. 고도화 이후에도 여전히 실패하는 3 케이스 (→ 6주차 §1)

우승 retriever(Dense)로 10문항을 다시 retrieve해 top-4 chunk를 덤프한 결과, **retrieval 단으로는 못 잡는** 3가지 실패 패턴이 남았다. (노트북 cell "실패 케이스 발굴" 참조)

### 케이스 A — Q10 "빅테크 중 AI 투자에 가장 적극적인 곳은?" : 도입부 노이즈가 top-k 잠식
- 검색된 top-4 중 [0]"자, 이번 주 가장 중요한 이슈… 슈퍼위크입니다", [1]"인센티브 주고 혜택을…", [2]"엔트로픽 추격팀 특별편성…" — **3개가 영상 도입부/곁가지 멘트**. 정작 답(아마존·구글의 엔트로픽 투자 규모)이 든 chunk는 [3] 하나뿐.
- **왜 retrieval로 못 잡나:** 도입부 멘트가 "빅테크·실적·투자" 키워드를 그대로 포함해 임베딩 상 질문과 가깝다. dense는 "키워드는 맞지만 답은 없는" chunk를 구분하지 못한다. (4주차 §1에서 진단한 도입부 노이즈가 청킹·retrieval 고도화 후에도 잔존)
- **구조가 필요한 이유:** chunk가 질문에 *실제로 답하는지*는 임베딩 거리로는 판별 불가 → **LLM이 chunk를 읽고 관련성을 판정(grade)** 하고, 노이즈면 **재검색**하는 단계가 있어야 한다.

### 케이스 B — Q2 "AI 에이전트가 반도체 주도주 판도를 어떻게 바꾸나?" : 다른 영상으로 의미 표류
- top chunk [0]이 **전혀 다른 영상(2025-10-04 '2026 버블 시나리오')의 "인간의 생활 양식을 바꾸는 주도주"** 부분. "주도주"라는 단어가 겹쳐 의미적으로 끌려왔지만, 질문이 묻는 "AI 에이전트 → CPU(인텔·AMD·ARM) 재평가"와는 다른 맥락.
- **왜 retrieval로 못 잡나:** "주도주"가 두 영상에 모두 등장하고, broad한 추상 명사라 임베딩이 시간적·주제적으로 먼 영상을 구분하지 못한다. Hybrid+Rerank도 이 chunk를 완전히 배제하진 못했다.
- **구조가 필요한 이유:** 질문을 **도메인 한정 쿼리로 재작성**(예: "AI 에이전트 CPU 수혜주 인텔 AMD ARM")하면 표류를 줄일 수 있다 → **rewrite_query** 동기.

### 케이스 C — Q1 "최근 AI 투자 전쟁에서 주목한 수혜주는?" : broad query의 발산
- "AI 투자 전쟁 수혜주"가 지나치게 광범위해, top-4가 도입부 멘트 [0]·인프라 경쟁 [1]·CPU 삼총사 [2]·양극화 해지 [3]로 **여러 영상에 흩뿌려진다.** 답은 나오지만(구글·아마존) precision이 낮고, 한 질문에 여러 하위 주제(빅테크/반도체/데이터센터/전력)가 섞인다.
- **왜 retrieval로 못 잡나:** 단일 임베딩 벡터로는 "여러 하위 주제를 포괄하는 broad query"를 한 번에 만족시키기 어렵다. top-k를 늘려도 노이즈가 함께 는다.
- **구조가 필요한 이유:** broad query를 **구체적 sub-query로 쪼개거나 재작성**해 단계적으로 검색하면 발산을 줄일 수 있다 → **rewrite_query / 반복 검색** 동기.

### 왜 단일 RAG 파이프라인이 아니라 Agentic 구조인가 (한 단락)
세 케이스의 공통점은 **"임베딩 거리상으로는 가깝지만 질문에 실제로 답하지 못하는 chunk"** 가 top-k에 들어온다는 것이다. 이는 retriever를 dense→hybrid→rerank로 바꿔도 근본적으로 해결되지 않는다 — 어떤 단일 retriever도 *한 번의 검색*으로 "이 chunk가 답에 쓸모 있는가"를 스스로 판단하거나, 실패했을 때 *질문을 바꿔 다시 시도*할 수 없기 때문이다. 단일 파이프라인은 검색 결과가 노이즈여도 그대로 LLM에 넘겨 그럴듯한 답을 생성(케이스 A·C)하거나, 표류한 맥락으로 잘못된 근거를 댈 위험(케이스 B)을 안는다. 필요한 것은 **검색 → 관련성 판정(grade) → 부족하면 질문 재작성(rewrite) → 재검색 → 그래도 없으면 답변 불가** 라는 *피드백 루프*, 즉 검색을 한 번의 함수 호출이 아니라 **상태를 가진 라우팅 구조**로 다루는 것이다. 6주차의 출발점은 새 기능을 많이 붙이는 게 아니라, 바로 이 3개 실패 케이스를 시스템 구조로 어떻게 다룰지를 고민하는 데 있다.

---

## 6. 6주차로 넘기는 인터페이스

- **Baseline retriever:** `advanced_retrieval.retrievers.build_dense()` — 6주차 `retrieve` 노드가 그대로 import.
- **비교 대상으로 남겨둔 것:** `build_hybrid_rerank()` (precision 우승, latency 비쌈) — 6주차에서 "agentic routing으로 rerank의 precision 이득을 선택적으로 회수 가능한가"를 검증하는 reference.
- **실패 케이스 3개 (A: 도입부 노이즈 / B: 의미 표류 / C: broad query 발산):** 6주차 grade·rewrite·retry 구조가 실제로 개선하는지 평가하는 대상.
