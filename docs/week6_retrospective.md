# Week 6 — Agentic RAG (LangGraph) 회고

> 산출 노트북: [week6_agentic_rag.ipynb](../notebooks/week6_agentic_rag.ipynb)
> 워크플로우: [week6_workflow_diagram.md](week6_workflow_diagram.md) · ADR: [adr/week6_agentic_rag.md](adr/week6_agentic_rag.md)
> 선행: [week5_retrieval_ablation.md](week5_retrieval_ablation.md) (우승 retriever = Dense, 실패 3케이스 A/B/C)

---

## 1. 5주차에서 이어받은 것

- **Baseline retriever = Dense** (5주차 RAGAS 4지표 평균 우승).
- **여전히 실패하던 3 케이스:** A) 도입부 노이즈가 top-k 잠식(Q10) / B) 다른 영상으로 의미 표류(Q2) / C) broad query 발산(Q1).
- 공통 원인: "임베딩상 가깝지만 답에 쓸모없는 chunk가 top-k에 끼고, 부족해도 LLM이 답을 만든다". → 6주차는 이를 `retrieve→grade→rewrite→generate`(+retry, +reflect) 구조로 다룬다.

---

## 2. 정량 비교 — Baseline / V0 / V1

동일 10문항, 동일 임베딩/LLM/DB, **clean collection 공통**. 바뀐 것은 routing 구조뿐.

| 구성 | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | 평균 Latency(s) | 평균 LLM 호출 |
|---|---|---|---|---|---|---|
| Baseline (Dense + plain chain) | 0.9378 | 0.6253 | 0.9556 | 0.9250 | 4.18 | 1.0 |
| **V0** (grade+meta / rewrite / retry + neighbor) | 1.0000 | 0.6265 | 0.9932 | 0.9667 | 8.49 | 2.0 |
| V1 (V0 + reflect) | 0.8532 | **0.6305** | 0.9932 | 0.9667 | 8.78 | 3.0 |
| **V2** (adaptive router: analyze→decompose) | **1.0000** | 0.6074 | **1.0000** | **1.0000** | 11.78 | 3.8 |

해석:
- **V0가 baseline 대비 분명히 개선.** Faithfulness 0.938→1.0, ContextPrecision 0.956→0.993, ContextRecall 0.925→0.967. AnswerRelevancy만 거의 평평(0.625→0.627).
  - **왜:** `grade_documents`가 검색된 chunk를 관련 있는 것만 남겨 **ContextPrecision↑**, neighbor expansion(±1)이 답 경계 chunk를 보강해 **ContextRecall↑**, 걸러진 chunk만 근거로 쓰니 **Faithfulness 완벽**.
  - **비용:** latency ~2배(4.18→8.49s), LLM 호출 1→2(grade 1콜 추가).
- **V1(reflect)은 오히려 Faithfulness를 깎았다**(1.0→0.853) — precision/recall은 동일, AnswerRel 미미하게↑, LLM 호출 +1(3.0). reflect가 일부 질문에서 ungrounded로 판정→rewrite→재검색→재생성을 돌렸고, 재생성된 답이 오히려 덜 grounded해지는 역효과. **"agentic을 더 얹는다고 좋아지지 않는다"의 실증.**

### V2 — adaptive router (후속 실험, branch `week6-adaptive-router`)

V0의 한계는 "rewrite/retry 루프가 in-corpus 질문에서 한 번도 발화하지 않는다"였다(grade가 늘 통과 → §4 참조). 원인은 분해가 **reactive**(grade 실패 후)였다는 점. V2는 분해를 **proactive**로 옮긴다: 진입부에 `analyze_query`(단답 vs 복합 분류)를 두고, **복합이면 `decompose`로 multi-query를 먼저 만든 뒤 retrieve**한다.

- **결과: ContextPrecision·ContextRecall 둘 다 1.0(완벽).** V0(0.993/0.967) 대비 더 올랐다. **router가 10문항 중 8개를 complex로 분류해 decompose가 실제로 8번 발화**(V0의 rewrite는 0번) → sub-query별 검색이 관련 context를 빠짐없이 끌어와 recall/precision이 만점.
- **router 분류는 신뢰할 만함:** 별도 broad/simple 8문항 프로토타입에서 8/8 정확. 평가셋 10문항에서도 단답 2개(Q5 매수후보, Q8 GTC)는 simple로 분류해 decompose를 건너뜀.
- **대가:** latency 최고(8.49→11.78s, baseline 대비 ~2.8배), LLM 호출 3.8(analyze 항상 +1, complex면 decompose +1). **AnswerRelevancy는 오히려 최저(0.607)** — 분해로 답이 넓어지면서 원질문 정조준에서 살짝 벗어나는 부작용.
- **정성(broad 5문항):** Q1(수혜주 분야별)에 반도체·엔트로픽 분야까지, Q2(트럼프 정책)에 조선·방산 수혜까지 붙어 **breadth가 결정적으로 향상**. 단 Q4(2025-26 핵심테마 종합)는 분해해도 거부 — 코퍼스에 종합 자료가 없는 한계(방법 문제 아님).

**해석:** V2는 **context 완전성(precision·recall 만점)을 위해 비용(latency·LLM 호출)을 가장 많이 쓰는** 구성. "분해가 실제로 발화하게 만든" 것이 V0와의 결정적 차이다. 정확도 우선이면 V2, 비용 민감하면 V0가 sweet spot(아래 §4 결론).

---

## 3. Preprocessing 효과 분리 (원본 vs clean baseline)

| baseline | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall |
|---|---|---|---|---|
| 원본 collection (study_advanced_retrieval) | 0.9689 | 0.6280 | 0.9556 | 0.9250 |
| clean collection (study_agentic_clean) | 0.9378 | 0.6253 | 0.9556 | 0.9250 |

**ContextPrecision·ContextRecall 완전히 동일**(0.9556 / 0.9250). Faithfulness 차이(0.969 vs 0.938)는 검색 컨텍스트가 동일하므로 LLM judge 노이즈일 뿐.

→ **Preprocessing은 효과 0.** 5주차에서 의도한 "LLM utility로 도입부/꼬리 노이즈 제거"가 **1500자 chunk-level에서는 한 개도 drop되지 않았다**(raw 68 = clean 68). 큰 chunk 안에 노이즈가 투자 본문과 섞여 있어 LLM이 전부 keep으로 판정하기 때문. **노이즈는 chunk가 아니라 문장 단위**라 chunk-level 분류로는 못 잡는다는 negative finding. (노이즈 처리의 부담은 자연히 query-time `grade_documents`로 넘어갔고, 그게 V0의 ContextPrecision 개선으로 나타났다.)

---

## 4. 운영 관점 회고

### 재검색(분해)이 도움이 된 사례
**V0의 reactive rewrite/retry는 10문항 전부에서 한 번도 트리거되지 않았다.** route_history 모두 `retrieve > grade > generate`, retry 0, LLM 2콜. `grade_documents`가 항상 ≥1개 관련 chunk를 찾아 "enough"로 판정한 탓 — Dense가 broad·표류 질문에서도 주제적으로 인접한 chunk를 최소 하나는 끌어오고 grade가 관대해서다. **즉 "grade 실패 시 재검색"이라는 reactive 설계로는 in-corpus 질문에서 분해가 영영 발화하지 않는다.**

**→ 이걸 V2(adaptive router)가 해결했다.** 분해를 grade 실패에 의존하지 않고 **질문 분석으로 proactive하게** 트리거하니, 10문항 중 **8개에서 decompose가 실제 발화**(`analyze(complex) > decompose > retrieve > grade > generate`)했고 그 결과 ContextRecall 0.967→**1.0**, ContextPrecision 0.993→**1.0**으로 올랐다. **"재검색이 도움이 되려면, grade 실패를 기다리지 말고 질문을 보고 먼저 분해해야 한다"**가 핵심 교훈.

### 재검색이 불필요/악영향이었던 사례
**V1의 reflect.** reflect가 일부 질문에서 답을 ungrounded로 보고 재루프를 돌렸는데, 결과적으로 **Faithfulness가 1.0→0.853으로 하락**하고 LLM 호출만 +50%(2→3) 늘었다. 이미 충분히 근거 있는 답을 reflect가 "다시 의심"해 재생성하면서 오히려 표류시킨, 명백한 over-engineering 사례.

### Baseline 대비 평균 latency / 비용 변화
- Latency: **4.18s → V0 8.49s(~2.0배) → V1 8.78s(~2.1배) → V2 11.78s(~2.8배).**
- LLM 호출: **1.0 → V0 2.0 → V1 3.0 → V2 3.8** (질문당). grade +1, reflect +1, V2는 analyze 항상 +1·decompose(complex) +1.
- 즉 V0는 2배 비용으로 품질↑, V1은 3배 비용에 품질↓, **V2는 3.8배 비용으로 context 만점(precision·recall 1.0)**.

### 본인 도메인에 Agentic RAG가 필요한가
**선택적으로 그렇다 — 우선순위에 따라 V0/V2 갈림.** 이 작고 깨끗한 한국어 코퍼스(68 chunk)에서:
- **공통으로 값어치 있는 부분:** `grade_documents`(노이즈 chunk 제거 → precision↑) + neighbor expansion(recall↑). LLM 1콜로 Faithfulness/Precision/Recall을 끌어올렸다.
- **값어치 없는 부분:** V1의 reflect(품질 하락, 비용↑). 전형적인 "구조 복잡도↑, 효과↓".
- **분해(multi-query)는 reactive(V0)면 죽고 proactive(V2)면 산다.** V2는 분해를 실제 발화시켜 context를 만점으로 끌어올렸으나, latency 2.8배·LLM 3.8콜에 AnswerRelevancy는 소폭 하락.
- **결론(sweet spot):**
  - **비용 민감 / 단답 위주** → **V0(grade+neighbor, reflect·router 끔).** 2배 비용으로 precision 0.993·recall 0.967.
  - **정확도·context 완전성 우선 / 복합 질문 많음** → **V2(adaptive router).** 2.8배 비용으로 precision·recall 1.0. router가 단답은 자동으로 건너뛰어 불필요 비용을 일부 줄임.
  - **공통: reflect는 끈다.** 7주차에서 grade 기준·decompose 품질을 튜닝하면 V2의 AnswerRelevancy 하락과 비용을 더 다듬을 수 있다.

---

## 5. 실패 3케이스 재검토 (구조로 해결됐나)

노트북 cell 5에서 V0/V1로 다시 돌린 결과:

| 케이스 | 결과 | 판정 |
|---|---|---|
| **C (Q1 broad)** "AI 투자 전쟁 수혜주" | V0: `retrieve>grade>generate`, retry 0. 답=구글·아마존(+엔비디아·ARM). grade가 관련 chunk만 남겨 답 정확. | **부분 해결** — precision은 개선됐으나, 정작 broad query용으로 만든 rewrite 분해는 발화 안 함(grade가 바로 enough). |
| **B (Q2 표류)** "AI 에이전트 반도체 판도" | V0: 답=GPU→CPU 재평가(인텔·AMD). 표류 영상 chunk 없이 on-topic. | **해결** — grade(+제목/날짜 메타)가 다른 영상 chunk를 걸러낸 것으로 보임. |
| **A (Q10 도입부)** "빅테크 중 AI 투자 최적극" | V0: 답=아마존·구글 엔트로픽 투자. 도입부 노이즈 없이 핵심. | **해결** — grade가 "슈퍼위크" 류 도입부 chunk를 관련 없음으로 제외(ContextPrecision 0.956→0.993이 이를 뒷받침). |

→ 3 케이스 모두 **`grade_documents`의 chunk 필터링으로 개선**됐다. 흥미롭게도 broad query(C)를 겨냥해 만든 **rewrite 분해는 한 번도 안 쓰였다** — grade가 먼저 충분한 관련 chunk를 찾았기 때문. 즉 실패 케이스를 푼 주역은 비싼 루프가 아니라 **싼 grade 필터링**이었다.

---

## 6. 왜 단일 RAG가 아니라 Agentic 구조인가 (한 단락)

5주차 3 케이스의 공통 병폐는 "임베딩 거리상 가깝지만 질문에 답하지 못하는 chunk가 top-k에 섞여 들어와, 단일 파이프라인이 그걸 그대로 LLM에 넘긴다"였다. 6주차에서 그 사이에 **검색 결과를 LLM이 읽고 관련성을 판정(grade)하고 노이즈를 떨어내는 한 단계**를 넣은 것만으로 ContextPrecision(0.956→0.993)·Faithfulness(0.938→1.0)가 올랐다 — 단일 retriever를 dense→hybrid→rerank로 바꿔도 못 얻던 개선이다. 이 점에서 "검색을 한 번의 함수 호출이 아니라, 판정·재시도를 포함한 상태 기반 구조로 다룬다"는 Agentic 전제는 이 도메인에서 **부분적으로 옳았다.** 다만 같은 실험이 **반례도 함께** 보여줬다: 재검색 루프는 (관대한 grade + 주제적으로 풍부한 코퍼스 탓에) 한 번도 발화하지 않았고, reflect는 비용만 늘리며 품질을 깎았다. 결국 Agentic의 가치는 "모든 노드를 다 켜는 것"이 아니라 **어느 노드가 실제로 실패 케이스를 푸는지 식별하고 그것만 남기는 것**(여기선 grade+neighbor)에 있었다. 7주차 평가의 출발점은 이 선택적 적용 — 어떤 질문 유형에서 어떤 노드가 비용 대비 효과가 있는가 — 를 정량화하는 것이다.
