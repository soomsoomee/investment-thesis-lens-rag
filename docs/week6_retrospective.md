# Week 6 — Agentic RAG (LangGraph) 회고


## 1. 추가한 처리 단계 (Agentic 파이프라인 순서)

5주차 단일 RAG(`retrieve → generate`) 위에, 6주차에서 아래 단계를 순서대로 끼워넣었다. **V2(adaptive)가 전체, V0는 1·2단계 제외, V1은 V0 + 9단계.**

| 순서 | 단계 | 무엇을 하나 | 변형 | 겨냥한 실패케이스 |
|---|---|---|---|---|
| 1 | `analyze_query` | 질문을 **단답 vs 복합**으로 분류(라우팅) | V2 | C(broad) |
| 2 | `decompose` | 복합이면 **2~4개 sub-query로 분해**(proactive query rewriting) | V2 | C(broad) |
| 3 | `retrieve` | clean collection에 Dense 검색(복합이면 sub-query별 union) | 공통 | — |
| 4 | **neighbor expansion** | 검색된 chunk의 ±1 이웃(같은 영상)을 동봉 → recall 보강 | V0+ | 경계 잘림 |
| 5 | `grade_documents` | LLM judge가 **본문 + 제목/날짜 메타데이터**를 함께 보고 관련성 판정 → 노이즈·표류 chunk 제거 | V0+ | A(도입부)·B(표류) |
| 6 | **조건부 라우팅** | 관련 chunk 충분→generate / 부족→rewrite 재검색(`retry≤2`) | V0+ | OOD 거부 |
| 7 | `rewrite_query` | (reactive) 부족 시 질문 재작성 후 재검색 | V0+ | — |
| 8 | `generate` | 관련 chunk 근거로만 답변; 부족하면 "확인할 수 없습니다" 거부 | 공통 | — |
| 9 | `reflect` | 생성 답이 컨텍스트에 근거하는지 자가검증 → 아니면 재시도 | V1 | hallucination |


---

## 2. 정량 비교


| 구성 | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | 평균 Latency(s) | 평균 LLM 호출 |
|---|---|---|---|---|---|---|
| Baseline (Dense + plain chain) | 0.9378 | 0.6253 | 0.9556 | 0.9250 | 4.18 | 1.0 |
| V0 (grade+meta / rewrite / retry + neighbor) | 1.0000 | 0.6265 | 0.9932 | 0.9667 | 8.49 | 2.0 |
| V1 (V0 + reflect) | 0.8532 | **0.6305** | 0.9932 | 0.9667 | 8.78 | 3.0 |
| **V2** (adaptive router: analyze→decompose) | **1.0000** | 0.6074 | **1.0000** | **1.0000** | 11.78 | 3.8 |

해석: **Baseline → V0(grade+neighbor) → V1(+reflect) → V2(adaptive router)** 를 순서대로 본다.

- **Baseline → V0: 분명한 개선.** Faithfulness 0.938→1.0, ContextPrecision 0.956→0.993, ContextRecall 0.925→0.967 (AnswerRelevancy만 평평, 0.625→0.627).
  - **왜:** `grade_documents`가 검색된 chunk를 관련 있는 것만 남겨 **precision↑** — 실측으로 검색+neighbor로 부풀린 88개 중 **47%(41개)를 실제 drop**했고, Q2의 표류 chunk("인간의 생활양식 주도주", 버블 영상)도 여기서 걸러졌다. neighbor expansion(±1)이 답 경계 chunk를 보강해 **recall↑**, 걸러진 chunk만 근거로 써서 **Faithfulness 완벽**.
  - **비용:** latency ~2배(8.49s), LLM 1→2(grade +1).
- **V0 → V1(+reflect): 오히려 악화 → 기각.** Faithfulness 1.0→0.853(precision/recall 동일, LLM +1=3.0). reflect가 일부 질문에서 ungrounded로 보고 재루프→재생성하면서 답이 덜 grounded해지는 역효과. **"agentic을 더 얹는다고 좋아지지 않는다"의 실증.**
- **V0 → V2(adaptive router): context 만점.** ContextPrecision·ContextRecall 둘 다 **1.0**(V0 0.993/0.967 대비↑). 결정적 차이는 **분해를 reactive(grade 실패 후 — V0에선 한 번도 안 떰)에서 proactive(질문 분석 후)로 옮긴 것**: router가 10문항 중 8개를 complex로 분류해 `decompose`가 8번 발화(V0의 rewrite는 0번) → sub-query별 검색이 관련 context를 빠짐없이 끌어와 만점. router 분류도 신뢰할 만함(프로토타입 8/8, 평가셋에선 단답 Q5·Q8을 simple로 걸러 decompose 건너뜀). **대가:** latency 최고 11.78s(~2.8배)·LLM 3.8, **AnswerRelevancy는 최저(0.607)** — 분해로 답이 넓어지며 원질문 정조준에서 살짝 벗어남. 정성(broad 5문항)에선 Q1에 데이터센터·반도체, Q2에 조선·방산 수혜까지 붙어 **breadth가 결정적으로 향상**, 단 Q4(종합 테마)는 코퍼스 한계로 거부.
- **종합:** 싼 비용(LLM 1콜)으로 품질을 끌어올린 건 **V0**, reflect(V1)는 비용↑·품질↓, **V2**는 최고 비용(3.8콜)으로 context 만점. → **정확도·복합질문 우선 = V2, 비용민감·단답위주 = V0, reflect는 끔**(상세 §5 결론).


---

## 3. 운영 관점 회고

### 재검색(분해)이 도움이 된 사례
**V0의 reactive rewrite/retry는 10문항 전부에서 한 번도 트리거되지 않았다.** route_history 모두 `retrieve > grade > generate`, retry 0, LLM 2콜. `grade_documents`가 항상 ≥1개 관련 chunk를 찾아 "enough"로 판정한 탓 — Dense가 broad·표류 질문에서도 주제적으로 인접한 chunk를 최소 하나는 끌어오고 grade가 관대해서다. **즉 "grade 실패 시 재검색"이라는 reactive 설계로는 in-corpus 질문에서 분해가 영영 발화하지 않는다.**

**→ 이걸 V2(adaptive router)가 해결했다.** 분해를 grade 실패에 의존하지 않고 **질문 분석으로 proactive하게** 트리거하니, 10문항 중 **8개에서 decompose가 실제 발화**(`analyze(complex) > decompose > retrieve > grade > generate`)했고 그 결과 ContextRecall 0.967→**1.0**, ContextPrecision 0.993→**1.0**으로 올랐다. **"재검색이 도움이 되려면, grade 실패를 기다리지 말고 질문을 보고 먼저 분해해야 한다"**가 핵심 교훈.

### 재검색이 불필요/악영향이었던 사례
**V1의 reflect.** reflect가 일부 질문에서 답을 ungrounded로 보고 재루프를 돌렸는데, 결과적으로 **Faithfulness가 1.0→0.853으로 하락**하고 LLM 호출만 +50%(2→3) 늘었다. 이미 충분히 근거 있는 답을 reflect가 "다시 의심"해 재생성하면서 오히려 표류시킨, 명백한 over-engineering 사례.

### Baseline 대비 평균 latency / 비용 변화
- Latency: **4.18s → V0 8.49s(~2.0배) → V1 8.78s(~2.1배) → V2 11.78s(~2.8배).**
- LLM 호출: **1.0 → V0 2.0 → V1 3.0 → V2 3.8** (질문당). grade +1, reflect +1, V2는 analyze 항상 +1·decompose(complex) +1.
- 즉 V0는 2배 비용으로 품질↑, V1은 3배 비용에 품질↓, **V2는 3.8배 비용으로 context 만점(precision·recall 1.0)**.

### 이 프로젝트에서 Agentic RAG가 필요한가
**선택적으로 그렇다 — 우선순위에 따라 V0/V2 갈림.** 이 작고 깨끗한 한국어 코퍼스(68 chunk)에서:
- **공통으로 값어치 있는 부분:** `grade_documents`(노이즈 chunk 제거 → precision↑) + neighbor expansion(recall↑). LLM 1콜로 Faithfulness/Precision/Recall을 끌어올렸다.
- **값어치 없는 부분:** V1의 reflect(품질 하락, 비용↑). 전형적인 "구조 복잡도↑, 효과↓".
- **분해(multi-query)는 reactive(V0)면 죽고 proactive(V2)면 산다.** V2는 분해를 실제 발화시켜 context를 만점으로 끌어올렸으나, latency 2.8배·LLM 3.8콜에 AnswerRelevancy는 소폭 하락.
- **결론(sweet spot):**
  - **비용 민감 / 단답 위주** → **V0(grade+neighbor, reflect·router 끔).** 2배 비용으로 precision 0.993·recall 0.967.
  - **정확도·context 완전성 우선 / 복합 질문 많음** → **V2(adaptive router).** 2.8배 비용으로 precision·recall 1.0. router가 단답은 자동으로 건너뛰어 불필요 비용을 일부 줄임.
  - **공통: reflect는 끈다.** 7주차에서 grade 기준·decompose 품질을 튜닝하면 V2의 AnswerRelevancy 하락과 비용을 더 다듬을 수 있다.
