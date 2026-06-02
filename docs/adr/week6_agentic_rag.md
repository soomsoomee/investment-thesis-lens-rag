# ADR — Week 6: Agentic RAG (LangGraph)

Date: 2026-06-02
Status: Accepted
Track: Study — Week 6

## 1. Decision

5주차 최종 Dense retriever를 baseline으로 두고, 6주차에 **LangGraph 기반 `retrieve → grade_documents → rewrite_query → generate` + 조건부 retry(≤2)** 구조를 도입했다. `grade_documents`는 검색된 chunk의 본문 + 제목/날짜 메타데이터를 함께 보고 관련성을 판정하고, `retrieve`는 multi-query union + neighbor expansion(±1)을 수행한다. `reflect`(생성 후 groundedness 자가검증)는 변형(V1)으로 구현해 비교했다.

**정량 비교 결과 최종 채택 = V0(reflect 제외).** V0가 baseline 대비 Faithfulness 0.938→1.0, ContextPrecision 0.956→0.993, ContextRecall 0.925→0.967로 개선됐고(LLM 호출 1→2, latency ~2배), V1(reflect)은 Faithfulness를 0.853으로 오히려 떨어뜨려 기각했다.

## 2. Context

5주차 retrieval 고도화(Dense/BM25/Hybrid/Hybrid+Rerank ablation) 이후에도 특정 질문에서 ① 도입부/곁가지 chunk가 top-k에 끼고(케이스 A), ② 다른 영상의 chunk로 의미가 표류하고(B), ③ broad query가 여러 주제로 발산(C)하는 문제가 남았다. 어떤 단일 retriever도 "검색된 chunk가 답에 쓸모 있는지"를 스스로 판단하거나 실패 시 질문을 바꿔 재시도할 수 없다는 점이 한계였고, 이를 상태 기반 라우팅으로 다루기 위해 Agentic 구조를 도입했다.

## 3. Alternatives (검토했으나 채택하지 않음)

- **단순 Dense/Hybrid 유지:** 케이스 A·B의 노이즈 chunk를 못 거른다(5주차에서 확인).
- **Reranker threshold 조정:** 5주차에서 Hybrid+Rerank가 ContextPrecision은 최고(0.992)였으나 latency ~9배에 recall 하락. threshold만으로는 "답 유무" 판정 불가.
- **Preprocessing(LLM utility 노이즈 제거):** 시도했으나 1500자 chunk-level에선 0개 drop(negative finding). 노이즈가 문장 단위라 chunk 분류로 못 잡음.
- **Self-RAG / CRAG:** 더 정교한 self-reflection 구조지만, 이미 reflect(간이판) 실험에서 품질 하락을 확인 — 이 코퍼스에 과하다고 판단.
- **Web Search fallback (Tavily):** 선택 과제. 코퍼스 폐쇄형 도메인이라 범위 제외.

## 4. Trade-off (도입 비용)

- **Latency 증가:** 4.18s → 8.49s(V0, ~2배). reflect 포함 시 8.78s.
- **LLM 호출/비용 증가:** 질문당 1 → 2(V0) → 3(V1). grade가 +1, reflect가 +1.
- **디버깅 복잡도:** route_history·retry_count·조건부 엣지 도입으로 추적 포인트 증가.
- **Routing 실패 가능성:** 본 실험에선 rewrite/retry 루프가 한 번도 발화하지 않음(grade가 관대) — 정반대로 "비싼 스캐폴딩이 dead code가 되는" 실패도 가능.
- **평가 설계 복잡도:** RAGAS 외에 latency·LLM 호출수·route_history를 함께 측정해야 trade-off가 보인다.

## 5. Consequence (7주차 Evaluation으로 연결)

이번 구조는 "어떤 노드가 실제로 실패 케이스를 푸는가"를 정량화할 토대를 만든다. 7주차에는:
- Agentic이 도움 되는 질문 유형 vs 그렇지 않은 유형을 나눠 평가(여기선 grade+neighbor는 유효, rewrite/retry·reflect는 무용/유해).
- 단순 RAGAS 외에 **refusal accuracy**(코퍼스 밖 질문 거부), **routing accuracy**(grade가 노이즈를 옳게 거르는 비율), **citation accuracy**(제목/날짜 출처 정확도)를 추가 검토.
- grade 기준을 빡세게 했을 때 retry 루프가 발화/개선되는지 재실험.
