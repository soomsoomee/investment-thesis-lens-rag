from naive_rag.constants import WEEK4_CONTENT_IDS, WEEK4_EVAL_DATA


def test_week4_content_ids_has_five_unique_ids():
    assert len(WEEK4_CONTENT_IDS) == 5
    assert len(set(WEEK4_CONTENT_IDS)) == 5
    assert all(isinstance(x, str) and x for x in WEEK4_CONTENT_IDS)


def test_week4_eval_data_has_ten_complete_rows():
    assert len(WEEK4_EVAL_DATA) == 10
    for row in WEEK4_EVAL_DATA:
        assert row["question"].strip()
        assert row["reference"].strip()
