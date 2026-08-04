"""
RAG Evaluation Pipeline.

Framework đã chọn: RAGAS 0.1.21 (đã pin trong requirements.txt).

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Wiring LLM/embeddings cho RAGAS:
    RAGAS 0.1.21 mặc định gọi thẳng OpenAI (cần OPENAI_API_KEY) cho cả LLM giám khảo
    lẫn embeddings. Repo này chỉ có OPENROUTER_API_KEY, nên:
      - LLM giám khảo: bọc ChatOpenAI trỏ base_url OpenRouter (giống Task 10) qua
        ragas.llms.LangchainLLMWrapper.
      - Embeddings (chỉ answer_relevancy cần): dùng lại BAAI/bge-m3 LOCAL (đã cache
        sẵn từ Task 4/5) qua ragas.embeddings.LangchainEmbeddingsWrapper — không cần
        thêm API key, không tốn thêm quota OpenRouter.

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval (không dùng — xem lựa chọn RAGAS ở trên)
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS — framework đã chọn
# =============================================================================

def _build_ragas_llm():
    """LLM giám khảo RAGAS, trỏ qua OpenRouter (cùng model với Task 10)."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from src.task10_generation import LLM_MODEL

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) trong .env")

    chat = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings():
    """Embeddings cho RAGAS (answer_relevancy) — dùng lại bge-m3 local, không cần API key."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.task4_chunking_indexing import EMBEDDING_MODEL

    hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf)


def evaluate_with_ragas(
    golden_dataset: list[dict],
    use_reranking: bool = True,
    top_k: int = 5,
):
    """
    Evaluate RAG pipeline sử dụng RAGAS trên 1 config cụ thể.

    Args:
        golden_dataset: List {'question', 'expected_answer', 'expected_context'}
        use_reranking: Truyền xuống generate_with_citation() -> retrieve() (Task 9).
            True = hybrid search + RRF rerank, False = hybrid không rerank.
            Đây là trục A/B chính dùng ở compare_configs().
        top_k: Số chunks retrieve cho mỗi câu hỏi.

    Returns:
        pandas.DataFrame — mỗi hàng là 1 câu hỏi, cột là 4 metric RAGAS.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from src.task10_generation import generate_with_citation

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = generate_with_citation(
            item["question"], top_k=top_k, use_reranking=use_reranking
        )
        contexts = [c["content"] for c in result["sources"]]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        # RAGAS yêu cầu contexts non-empty; nếu retrieval/fallback đều rỗng,
        # nhét 1 chuỗi rỗng để không crash Dataset.from_dict (faithfulness sẽ
        # tự chấm 0 vì answer không có ground trong context rỗng).
        eval_data["contexts"].append(contexts if contexts else [""])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=_build_ragas_llm(),
        embeddings=_build_ragas_embeddings(),
        raise_exceptions=False,
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens (không dùng — xem lựa chọn RAGAS ở trên)
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 config retrieval:
        - hybrid_rerank:    Semantic + BM25 -> RRF merge -> RRF rerank (Task 7)
        - hybrid_no_rerank: Semantic + BM25 -> RRF merge, KHÔNG rerank thêm

    (Không so PageIndex riêng ở đây vì PageIndex chỉ kích hoạt khi cosine < 0.48 —
    với 17 câu hỏi trong-domain của golden dataset, hybrid gần như luôn đủ điểm,
    nên A/B có/không PageIndex sẽ không tạo khác biệt quan sát được.)

    Returns:
        dict[str, pandas.DataFrame] — key là tên config, value là kết quả RAGAS.
    """
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "hybrid_no_rerank": {"use_reranking": False},
    }

    results = {}
    for name, params in configs.items():
        print(f"\n>>> Đang chạy RAGAS cho config: {name} ...")
        results[name] = evaluate_with_ragas(golden_dataset, **params)
    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict):
    """Export evaluation results (bảng điểm A/B + worst performers) ra results.md"""
    config_names = list(comparison.keys())
    avgs = {name: df[METRIC_COLUMNS].mean() for name, df in comparison.items()}

    lines = ["# RAG Evaluation Results", ""]
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append(
        "RAGAS 0.1.21 — LLM giám khảo qua OpenRouter (cùng model Task 10), "
        "embeddings BAAI/bge-m3 local (chỉ dùng cho answer_relevancy)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")

    header = "| Metric | " + " | ".join(config_names) + " | Δ |"
    sep = "|" + "---|" * (len(config_names) + 2)
    lines.append(header)
    lines.append(sep)
    for m in METRIC_COLUMNS:
        vals = [avgs[c][m] for c in config_names]
        delta = vals[0] - vals[1] if len(vals) == 2 else float("nan")
        row = f"| {m} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {delta:+.3f} |"
        lines.append(row)
    avg_vals = [avgs[c][METRIC_COLUMNS].mean() for c in config_names]
    avg_delta = avg_vals[0] - avg_vals[1] if len(avg_vals) == 2 else float("nan")
    lines.append(
        "| **Average** | "
        + " | ".join(f"**{v:.3f}**" for v in avg_vals)
        + f" | {avg_delta:+.3f} |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## A/B Comparison Analysis")
    lines.append("")
    for name in config_names:
        lines.append(f"**{name}:**")
        lines.append(f"> Average score: {avgs[name][METRIC_COLUMNS].mean():.3f}")
        lines.append("")
    lines.append("**Kết luận:**")
    lines.append(
        "> _(Điền sau khi xem số liệu thật ở trên — config nào tốt hơn, chênh lệch "
        "có đáng kể không, có nhất quán giữa các câu hỏi hay chỉ lệch ở vài câu)._"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Worst Performers (Bottom 3)")
    lines.append("")
    main_df = comparison[config_names[0]].copy()
    main_df["avg_score"] = main_df[METRIC_COLUMNS].mean(axis=1)
    worst = main_df.nsmallest(3, "avg_score")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        q_text = row.get("question") or row.get("user_input") or row.get("query") or "Unknown"
        q = str(q_text)[:70].replace("|", "/")
        lines.append(
            f"| {i} | {q} | {row['faithfulness']:.2f} | {row['answer_relevancy']:.2f} "
            f"| {row['context_recall']:.2f} | {row['context_precision']:.2f} | | |"
        )
    lines.append("")
    lines.append(
        "> _Failure Stage / Root Cause: điền tay sau khi đọc lại answer + context thật "
        "của 3 câu này (retrieval sai chunk? LLM bịa? context đúng nhưng answer diễn giải sai?)._"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1")
    lines.append("**Action:**")
    lines.append("**Expected impact:**")
    lines.append("")
    lines.append("### Cải tiến 2")
    lines.append("**Action:**")
    lines.append("**Expected impact:**")
    lines.append("")
    lines.append("### Cải tiến 3")
    lines.append("**Action:**")
    lines.append("**Expected impact:**")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Đã ghi kết quả vào {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset)
    export_results(comparison)

    for name, df in comparison.items():
        print(f"\n=== {name} ===")
        print(df[METRIC_COLUMNS].mean())
