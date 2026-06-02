# Week 6 — Agentic RAG Workflow Diagram

LangGraph 기반 `retrieve → grade_documents → rewrite_query → generate` + 조건부 라우팅/retry(≤2).
V1은 generate 뒤 `reflect`(groundedness 자가검증)를 추가해 ungrounded면 재시도.

## V0 — base (retrieve→grade→rewrite→generate + retry)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrieve(retrieve)
	grade_documents(grade_documents)
	rewrite_query(rewrite_query)
	generate(generate)
	__end__([<p>__end__</p>]):::last
	__start__ --> retrieve;
	grade_documents -.-> generate;
	grade_documents -.-> rewrite_query;
	retrieve --> grade_documents;
	rewrite_query --> retrieve;
	generate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## V1 — +reflect

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrieve(retrieve)
	grade_documents(grade_documents)
	rewrite_query(rewrite_query)
	generate(generate)
	reflect(reflect)
	__end__([<p>__end__</p>]):::last
	__start__ --> retrieve;
	generate --> reflect;
	grade_documents -.-> generate;
	grade_documents -.-> rewrite_query;
	reflect -.-> __end__;
	reflect -.-> rewrite_query;
	retrieve --> grade_documents;
	rewrite_query --> retrieve;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
