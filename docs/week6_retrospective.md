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
| **V0** (grade+meta / rewrite / retry + neighbor) | **1.0000** | 0.6265 | **0.9932** | **0.9667** | 8.49 | 2.0 |
| V1 (V0 + reflect) | 0.8532 | **0.6305** | 0.9932 | 0.9667 | 8.78 | 3.0 |

해석:
- **V0가 baseline 대비 분명히 개선.** Faithfulness 0.938→1.0, ContextPrecision 0.956→0.993, ContextRecall 0.925→0.967. AnswerRelevancy만 거의 평평(0.625→0.627).
  - **왜:** `grade_documents`가 검색된 chunk를 관련 있는 것만 남겨 **ContextPrecision↑**, neighbor expansion(±1)이 답 경계 chunk를 보강해 **ContextRecall↑**, 걸러진 chunk만 근거로 쓰니 **Faithfulness 완벽**.
  - **비용:** latency ~2배(4.18→8.49s), LLM 호출 1→2(grade 1콜 추가).
- **V1(reflect)은 오히려 Faithfulness를 깎았다**(1.0→0.853) — precision/recall은 동일, AnswerRel 미미하게↑, LLM 호출 +1(3.0). reflect가 일부 질문에서 ungrounded로 판정→rewrite→재검색→재생성을 돌렸고, 재생성된 답이 오히려 덜 grounded해지는 역효과. **"agentic을 더 얹는다고 좋아지지 않는다"의 실증.**

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

### 재검색이 도움이 된 사례
**해당 없음 — 10문항 전부에서 rewrite/retry가 한 번도 트리거되지 않았다.** 운영지표(노트북 cell 6)에서 10문항 모두 route_history = `retrieve > grade > generate`, retry_count = 0, LLM 호출 = 2(grade 1 + generate 1, rewrite 0). 즉 **`grade_documents`가 항상 ≥1개 관련 chunk를 찾아 "enough"로 판정**해서 재검색 루프가 발화하지 않았다. Dense가 (broad·표류 질문에서도) 주제적으로 인접한 chunk를 최소 하나는 끌어오고, grade 프롬프트가 관대한 탓. → **재검색 루프는 이 코퍼스/이 grade 기준에서는 사실상 dead code.** 발화시키려면 grade 기준을 빡세게(예: 관련 chunk N개 이상 요구) 하거나, 코퍼스 밖 질문 + 엄격한 judge가 필요하다.

### 재검색이 불필요/악영향이었던 사례
**V1의 reflect.** reflect가 일부 질문에서 답을 ungrounded로 보고 재루프를 돌렸는데, 결과적으로 **Faithfulness가 1.0→0.853으로 하락**하고 LLM 호출만 +50%(2→3) 늘었다. 이미 충분히 근거 있는 답을 reflect가 "다시 의심"해 재생성하면서 오히려 표류시킨, 명백한 over-engineering 사례.

### Baseline 대비 평균 latency / 비용 변화
- Latency: **4.18s → V0 8.49s(~2.0배) → V1 8.78s(~2.1배).**
- LLM 호출: **1.0 → V0 2.0 → V1 3.0** (질문당). grade가 +1, reflect가 추가로 +1.
- 즉 V0는 품질을 위해 **2배 비용**, V1은 **3배 비용에 품질은 더 나쁨**.

### 본인 도메인에 Agentic RAG가 필요한가
**선택적으로 그렇다.** 이 작고 깨끗한 한국어 코퍼스(68 chunk)에서:
- **값어치 있는 부분:** `grade_documents`(노이즈 chunk 제거 → precision↑) + neighbor expansion(recall↑). 추가 비용 LLM 1콜로 Faithfulness/Precision/Recall을 모두 끌어올렸다.
- **값어치 없는 부분:** rewrite/retry 루프(발화 안 함)와 reflect(품질 하락). 전형적인 "구조 복잡도 ↑, 효과 ↓".
- **결론:** 모든 질문에 full agentic을 돌릴 이유는 없다. **grade+neighbor만 켜고, retry는 정말 어려운/모호한 질문에서만, reflect는 끈다**가 이 도메인의 sweet spot. → **최종 채택 = V0(reflect 제외)**.

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
