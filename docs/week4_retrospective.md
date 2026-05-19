# Week 4 — Chunking 전략 비교 회고

> 산출 노트북: [week4_data_analysis.ipynb](../notebooks/week4_data_analysis.ipynb), [week4_chunking_experiments.ipynb](../notebooks/week4_chunking_experiments.ipynb)


---

## 1. 3주차 Baseline 회고

`study_naive_rag` collection에서 5건 `WEEK4_CONTENT_IDS`로 metadata filter한 후 10개 질문에 대해 k=4로 retrieve한 결과를 훑어봤다 (`week4_chunking_experiments.ipynb` cell 2). 대부분 retrieve는 정확했지만 명백히 잘못된/노이즈인 사례 3건:

| Q# | 질문 | 잘못 검색된 chunk 요약 | 왜 잘못됐는가 (가설) |
|---|---|---|---|
| Q1 | 최근 AI 투자 전쟁에서 주목한 수혜주는 무엇인가? | [2] 2025-10-04 "2026 버블 시나리오" 영상에서 "네트워크 장비 쪽이 있어요. 아스테라랩스, 아이스탄 네트워크..." 부분 | 질문이 "AI 투자 전쟁"이라는 광범위 query라 retrieve가 데이터센터 인프라 종목까지 폭넓게 잡음. 다른 영상(2025-10) chunk가 더 직접적인 4월 영상보다 위로 올라옴 — 의미적으로 인접하지만 시간적으로 거리 있음. |
| Q4 | 2026 버블 시나리오에서 본격 버블은 시작됐다고 보나? | [1] "그래서 오늘 리포트 정말 좋거든요. 제가 밑줄 깜빡 긋고 포스트 표시가 너무 많아서... 우리 구독자분들도 보실 수 있게 링크 댓글에다가 남겨놓을 테니까" | **명백한 광고/CTA 노이즈**. 영상 꼬리에 일관되게 등장하는 "특강/카페/댓글 안내" 멘트가 별도 chunk로 잘리지 않고 본문에 섞여 retrieve됨. |
| Q10 | 빅테크 중 AI 투자에 가장 적극적인 곳은? | [0] "자, 이번 주 가장 중요한 이슈 있죠? 바로 슈퍼위크입니다. 빅테크 5개가 실적 발표합니다" | 영상 도입부의 일반적인 인사/예고 멘트. 키워드(빅테크, 실적)가 질문과 매치되지만 정작 답에 필요한 정보(어떤 기업이 적극적인지) 0. **도입부 노이즈**. |

### 패턴

- **노이즈 chunk의 정체**: 영상 도입부(슈퍼위크 같은 주간 인트로)와 꼬리(특강 안내, 댓글 링크, 카페 공지) — 모든 영상에 일관되게 등장하는 광고/CTA 멘트가 본문과 섞여 chunk로 retrieve됨.
- **광범위 query의 약점**: 질문이 broad일수록(Q1 같은 "수혜주") 의미적으로 인접한 다른 영상까지 retrieve되어 직접 답이 있는 영상이 밀려남.

### 데이터 전처리 관점 개선 가설 (1-2개)

1. **꼬리 광고/CTA 패턴 제거 (preprocessing)** — 영상 꼬리 ~10% 자르거나, keyword detection으로 "특강 고정댓글", "카페에 올려둘게요", "댓글에 남겨놓을게요", "구독", "좋아요" 같은 패턴을 포함한 문장들을 ingest 전에 제거. Q4 같은 명백한 노이즈가 직접 해결됨.

---

## 2. 데이터 품질 진단 요약 

(상세는 [week4_data_analysis.ipynb](../notebooks/week4_data_analysis.ipynb) cell 7 참고)

5건 transcript는 Whisper 기반 plain text라 헤더/표/이미지/명시적 섹션이 없고, **`\n\n`과 `\n`이 한 번도 등장하지 않는다** — baseline `RecursiveCharacterTextSplitter`의 `separators` 중 처음 두 개가 무용지물이다. 영어식 `". "`가 평균 212회/문서로 잡히지만 대부분 외래어 약자·잘못 인식된 종결어미 잔재이고, 한국어 자연 종결어미 `다.`/`요.`/`습니다`(합 평균 230회/문서)는 baseline이 의미 단위로 잡지 못한다. 결과적으로 baseline 600/100은 사실상 `". "` → 공백 → 빈 문자열 fallback 위주의 character-level 분할에 가깝다. 또한 모든 영상 꼬리에 일관된 광고/CTA 패턴이 있어 노이즈 chunk로 retrieve될 가능성이 높다. 길이는 5건 합 ~77K tokens, 평균 15.4K, 최대 27.3K(2026 버블 시나리오 풀버전)로 한 영상당 약 25 chunk(chunk_size=600 기준), 가장 긴 건 ~45 chunk.

---

## 3. 청킹 전략 선택 이유

** `RecursiveCharacterTextSplitter` + 한국어 종결어미 우선 separators 보강** (`["다. ", "요. ", "습니다.", "다.", "요.", ". ", " ", ""]`). C는 600/100, D는 1500/300.

데이터 진단에서 `\n\n`/`\n`가 0개이고 한국어 종결어미가 풍부하다는 결과를 토대로, baseline의 "한국어 marker 부재" 약점을 직접 해결하는 가장 가벼운 방법을 우선 채택. 

---

## 4. Chunking 비교 실험 결과 

| strategy | splitter | chunk_size | overlap | chunk_count | Faithfulness | AnswerRelevancy | LLMContextPrecisionWithReference | LLMContextRecall |
|---|---|---|---|---|---|---|---|---|
| A | RecursiveCharacter | 600 | 100 | 160 | 0.9800 | 0.4671 | 0.9833 | 0.7250 |
| B-small | RecursiveCharacter | 500 | 100 | 196 | 0.8875 | 0.6223 | **1.0000** | 0.7167 |
| **B-large** | RecursiveCharacter | 1500 | 300 | 68 | **1.0000** | 0.6285 | 0.9444 | **0.9750** |
| C | RecursiveCharacter + 한국어 separators | 600 | 100 | 176 | 0.9250 | 0.6228 | 0.9917 | 0.8417 |
| **D** | RecursiveCharacter + 한국어 separators | 1500 | 300 | 72 | 0.9667 | 0.6178 | 0.9722 | 0.9500 |

(`week4_chunking_experiments.ipynb` cell 11)

**2x2 design 분석 (size × separators)**:

| | baseline separators | 한국어 separators | Δ (separators 효과) |
|---|---|---|---|
| **600 size** | A: Recall 0.725 | C: Recall 0.842 | **+0.117** ✓ |
| **1500 size** | B-large: Recall 0.975 | D: Recall 0.950 | -0.025 (무차이) |
| **Δ (size 효과)** | +0.250 ⬆️ | +0.108 ⬆️ | |

---

## 5. 메타데이터 활용 아이디어 (과제 #5, 5주차 활용 예정)

4주차에 추가한 메타데이터: `chunk_index`, `chunk_total`, `chunk_token_count` (+ 기존 `content_id`, `title`, `channel`, `published_at` = 총 7개). vector store 검증은 `week4_chunking_experiments.ipynb` cell 7에서 chunk 1개를 출력해 7 필드 모두 확인.

활용 아이디어:
- **`chunk_index` / `chunk_total`**: retrieve된 chunk의 ±1 chunk를 함께 LLM에 넘기는 "neighbor expansion". 한 영상에서 같은 화제가 연속해서 펼쳐지는 transcript 특성상 유효할 가능성이 큼. 회고 §1의 도입부 노이즈 chunk를 본문 쪽으로 펼쳐 의미를 보강하는 효과도 기대.
- **`chunk_token_count`**: retrieval 결과 합산 토큰 모니터링 → LLM context window 초과 방지 및 비용 예측. `tiktoken` 기반이라 OpenAI 모델 토큰과 직결.
- **`published_at`**: 시점 가중치(최근 발화 우선)로 query에 따라 retrieval rank 보정. 이미 chain에서 답변 출처 표시로는 활용 중 (`[chain.py](../src/naive_rag/chain.py)` `_format_docs`).

---

## 6. RAGAS 점수 해석 회고 (과제 #6)


### 3주차 baseline 대비 변화

대부분의 지표가 B-large/D에서 A 대비 크게 개선:
- **AnswerRelevancy**: A 0.467 → B-large 0.629 (+35%) / D 0.618 (+32%) — A가 압도적 worst.
- **ContextRecall**: A 0.725 → B-large 0.975 (+34%) / D 0.950 (+31%).
- **Faithfulness**: A 0.980 → B-large 1.0 / D 0.967. 거의 보존.
- **ContextPrecision**: A 0.983 → B-large 0.944 (-4%) / D 0.972 (-1%). 큰 chunk라 noise도 같이 들어와 약간 떨어짐.

### 변화가 왜 일어났는가 (가설)

1. **Size가 dominant 변수 (ContextRecall 기준)**: 600→1500으로 키울 때 Recall +0.25(baseline separators) / +0.108(한국어 separators). 한국어 transcript는 한 화제가 길게 펼쳐지는 발화 특성이라, 답에 필요한 정보가 작은 chunk(600)에는 잘려서 들어가는 반면 큰 chunk(1500)에는 한 chunk 안에 다 들어간다. 앞뒤로 유튜브 멘트 있는 것도 청크 자체를 크게 줘버리면 문제 완화됨. 
2. **한국어 separators는 small chunk에서만 유의미**: 600에서 A→C로 Recall +0.117 명확한 효과. 1500에서 B-large→D는 -0.025로 사실상 무차이. **큰 chunk가 한국어 marker 효과를 흡수**한다(어차피 종결어미 여러 번 포함하는 큰 단위라 separators 보강이 결정적 차이 못 만듦).
3. **B-small이 Faithfulness worst (0.887)**: chunk가 너무 작으면 의미 단위가 깨져 LLM이 부분 정보로 추측 → 답변의 사실성 떨어짐. "small chunk + 한국어 marker 부재"가 가장 위험.
4. **ContextPrecision은 큰 chunk에서 약간 떨어짐**(B-large 0.944, D 0.972): 큰 chunk는 답에 필요한 정보 외 다른 화제도 함께 포함 → "검색된 chunk가 질문에 직접 도움 되는 비율"이 살짝 낮아짐. 단 차이 미미.

### 본인 도메인에 가장 잘 맞는 chunking 전략 + 이유

**B-large (RecursiveCharacterTextSplitter, 1500/300, baseline separators)** — 종합 best.
- Faithfulness 1.0 (완벽), AnswerRelevancy 0.629 (top), ContextRecall 0.975 (top).
- 한국어 YouTube transcript의 발화 특성(한 화제가 길게 펼쳐짐)과 size 1500이 자연스럽게 맞아떨어짐.
- chunk 개수가 68로 가장 적어 ingest·retrieve 비용도 가장 낮음.
- 단점: ContextPrecision 0.944로 5 strategy 중 최저 — 큰 chunk에 다른 화제 섞임. 정밀도 보강이 5주차 과제.

차선 D (1500/300 + 한국어 separators)는 B-large 대비 거의 무차이 — 큰 chunk에선 한국어 marker 보강 효과가 사라지므로 의존성 없는 baseline separators(B-large)로도 충분.

**의외의 발견**: 같은 size(600)에선 한국어 separators(C)가 분명한 개선을 주는데, 큰 size(1500)에선 그 효과가 사라진다. → "한국어 separators 보강"이라는 처방은 **small chunk 한정 유효**.

### 5주차 retrieval 고도화에서 시도하고 싶은 것

§1 가설 2개를 5주차 작업으로 옮긴다:

1. **Preprocessing: 꼬리 광고/CTA 제거** (§1 가설 1 구현). ingest 전 단계에서 keyword detection ("특강 고정댓글", "카페에 올려둘게요", "댓글에 남겨놓을게요", "구독", "좋아요" 등)으로 문장 제거하거나 영상 꼬리 ~10% cut. Q4 같은 명백한 노이즈 직접 차단. 가장 가벼운 처방이라 5주차 1순위.
2. **`chunk_utility_score` 메타데이터** (§1 가설 2 + §5 후보): LLM classifier 또는 keyword/embedding 기반으로 chunk를 high/low utility로 분류해 메타데이터로 저장. retrieve 시 score 가중치 또는 threshold filter. Q10 도입부 멘트 같은 케이스 처리.
3. **Neighbor expansion** (`chunk_index` 활용, §5 첫 아이디어): retrieve된 chunk의 ±1 chunk를 같이 LLM에 넘김. 큰 chunk(1500) 위에서는 효과가 작을 수 있어 small chunk 전략과 함께 시험.
4. **Hybrid retrieval (BM25 + dense)**: 키워드 정확 매치 + 의미 매치 가중 결합. 종목/숫자/고유명사(엔비디아, GTC, 30GW 등) 정확 매치 강화.
5. **Multi-query rewriting**: 광범위 query를 sub-query로 쪼개 retrieve (Q1 같은 broad query 케이스). LLM으로 쿼리 분해 후 결과 union/rerank.
6. **Cross-encoder rerank**: 1차 dense retrieve(top-k 큰 값) 후 cross-encoder로 정렬. 회고 §1 misretrieve의 일반적 해결책.