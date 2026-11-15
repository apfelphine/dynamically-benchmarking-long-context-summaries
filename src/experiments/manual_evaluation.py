import asyncio
import os
import random
from typing import List

import multiprocessing
import tkinter as tk
from tkinter import ttk

import pandas as pd
import tqdm
from colorama import Fore
import numpy as np

from core.document_alignment.annotation_interface import AnnotationTool
from experiments.base_experiment_context import ExperimentContext
from experiments.configuration import base_experiment_dir
from experiments.generic_summaries.context import GenericSummaryContext
from experiments.query_based_summaries.context import QueryBasedSummaryContext

random.seed(42)

# How many summaries should be selected per LLM
K_generic = 2
K_query_focused = 3


def randomly_select_summaries_per_llm_summarizer(context: ExperimentContext, k: int):
    total_selected_summaries = []
    for llm in context.llms.keys():
        available_summaries = []
        for summarizer in context.summarizers.keys():
            available_summaries.extend([(s, llm, summarizer) for s in context.summaries])

        selected_summaries = random.sample(available_summaries, k)

        total_selected_summaries.extend([
                dict(
                    summary_type=context.name,
                    topic_name=context.corpus[s.bin_idx].topic,
                    bin_idx=s.bin_idx,
                    summary_path=context.get_save_path(s),
                    llm_name=llm_name,
                    summarizer_name=summarizer_name,
                )
                for s, llm_name, summarizer_name in selected_summaries
            ]
        )

    print(f"Selected {len(total_selected_summaries)} summaries")
    return total_selected_summaries


def select_summaries(selected_summary_path: str):
    if os.path.exists(selected_summary_path):
        return pd.read_csv(selected_summary_path, index_col=None)

    total_selected_summaries: List[dict] = []

    print(f"{Fore.BLUE}>> GENERIC SUMMARIES <<")
    exp1_context = GenericSummaryContext()
    total_selected_summaries.extend(
        randomly_select_summaries_per_llm_summarizer(exp1_context, K_generic)
    )

    print(f"{Fore.BLUE}>> QUERY-BASED SUMMARIES <<")
    exp2_context = QueryBasedSummaryContext()
    total_selected_summaries.extend(
        randomly_select_summaries_per_llm_summarizer(exp2_context, K_query_focused)
    )

    df = pd.DataFrame([s for s in total_selected_summaries])

    print(f"{Fore.GREEN}>> RESULTS SAVED <<")
    print(df.head(20))

    df.to_csv(
        selected_summary_path,
        index=False,
    )
    return df


def open_annotations_gui(summary_path, summarizer_name, llm_name):
    base_dir = os.path.join(summary_path, summarizer_name, llm_name)
    app = AnnotationTool()
    app._load_sim_matrix(
        os.path.join(summary_path, "documents.json"),
        os.path.join(summary_path, "document_chunks.json"),
        os.path.join(base_dir, "summary.md"),
        os.path.join(base_dir, "summary_chunks.json"),
        os.path.join(base_dir, "sim_matrix.npy"),
        threshold=0
    )
    app.mainloop()


def open_error_mode_gui(row_idx, shared_counts, error_modes: List[str]):
    root = tk.Tk()
    root.title(f"Errors (Row: {row_idx})")

    for i, mode in enumerate(error_modes):
        tk.Label(root, text=mode.capitalize()).grid(row=i, column=0, padx=10, pady=5)
        var = tk.IntVar(value=0)
        var.trace_add("write", lambda *_, m=mode, v=var: shared_counts.update({m: v.get()}))
        ttk.Spinbox(root, from_=0, to=100, textvariable=var, width=5).grid(row=i, column=1, padx=10)

    ttk.Button(root, text="Save & Next", command=root.destroy).grid(row=len(error_modes), columnspan=2, pady=10)
    root.mainloop()

async def manually_evaluate_summary(row, error_modes, results_path):
    with multiprocessing.Manager() as manager:
        counts = manager.dict({m: 0 for m in error_modes})

        p_anno = multiprocessing.Process(
            target=open_annotations_gui, args=(row.summary_path, row.summarizer_name, row.llm_name)
        )
        p_log = multiprocessing.Process(target=open_error_mode_gui, args=(row.Index, counts, error_modes))

        p_anno.start()
        p_log.start()

        # Wait until EITHER window is closed
        while p_anno.is_alive() and p_log.is_alive():
            await asyncio.sleep(0.1)

        # Append the new counts directly to the CSV
        new_data = pd.DataFrame([{'row_index': row.Index, **counts}])
        new_data.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

        # Terminate whichever window was left open
        for p in (p_anno, p_log):
            if p.is_alive():
                p.terminate()
            p.join()


def analyse_aligner_performance(df: pd.DataFrame, error_modes: List[str]) -> None:
    """Calculates alignment metrics based on the sim_matrix.npy files."""

    grouped_errors = df[error_modes].sum()
    print(">> Error types found:")
    print(grouped_errors.to_string())

    # Dictionary to keep track of overall metrics
    metrics = {
        "total_pairs": 0,
        "made_alignments": 0,
        "false_negatives": 0,
        "false_positives": 0
    }

    for row in df.itertuples():
        sim_matrix_path = os.path.join(row.summary_path, row.summarizer_name, row.llm_name, "sim_matrix.npy")

        if not os.path.exists(sim_matrix_path):
            continue

        sim_matrix = np.load(sim_matrix_path)

        metrics["total_pairs"] += sim_matrix.size
        metrics["made_alignments"] += np.count_nonzero(sim_matrix)

        # False Negatives = missing sources
        metrics["false_negatives"] += row.source_missing

        # False Positives = wrong source in correct article + wrong source in wrong article
        metrics["false_positives"] += row.wrong_source_in_correct_article + row.wrong_source_in_wrong_article + row.related_source_in_wrong_article

    if metrics["total_pairs"] == 0:
        print("No valid similarity matrices found.")
        return

    # Derive Confusion Matrix Values
    total_pairs = metrics["total_pairs"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]

    # True Positives: Total alignments made minus the false positive alignments
    tp = max(0, metrics["made_alignments"] - fp)

    # True Negatives: Total possible pairs minus everything else (TP, FP, FN)
    tn = total_pairs - (tp + fp + fn)

    # Actual ground truth values (What SHOULD have been aligned vs NOT aligned)
    actual_positives = tp + fn
    actual_negatives = tn + fp

    # Calculate requested percentages and metrics
    pct_should_align = (actual_positives / total_pairs) * 100
    pct_should_not_align = (actual_negatives / total_pairs) * 100

    accuracy = (tp + tn) / total_pairs if total_pairs > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n>> Total pairs (possible alignments): {total_pairs:,}")
    print(f"\n>> Ground Truth Distribution:")
    print(f" - Percentage of pairs that SHOULD be aligned:     {pct_should_align:.2f}% ({actual_positives:,} pairs)")
    print(
        f" - Percentage of pairs that SHOULD NOT be aligned: {pct_should_not_align:.2f}% ({actual_negatives:,} pairs)"
    )

    print(f"\n>> Aligner Performance:")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f" - F1 Score:  {f1_score:.4f}")
    print(f" - Accuracy:  {accuracy:.4f}")

def analyse_results(selected_summaries: pd.DataFrame, analysis_results: pd.DataFrame, error_modes: List[str]) -> None:
    df_combined = pd.concat([
        selected_summaries.reset_index(drop=True),
        analysis_results.reset_index(drop=True)
    ], axis=1)

    generic = df_combined[df_combined["summary_type"] == "generic_summaries"]
    print("\n---------- GENERIC SUMMARIES ----------")
    analyse_aligner_performance(generic, error_modes)

    query_based = df_combined[df_combined["summary_type"] == "query_based_summaries"]
    print("\n---------- QUERY BASED SUMMARIES ----------")
    analyse_aligner_performance(query_based, error_modes)

    print("\n---------- OVERALL ----------")
    analyse_aligner_performance(df_combined, error_modes)


async def main():
    base_path = os.path.join(base_experiment_dir, "manual_evaluation")
    os.makedirs(base_path, exist_ok=True)

    print("[1/4] Selecting summaries for manual evaluation...")
    selected_summary_path = os.path.join(base_path, "summaries.csv")
    df = select_summaries(selected_summary_path)

    print("[2/4] Preparing evaluating error modes in the alignment...")
    error_modes = [
        # False Negative
        "source_missing",
        # False Positive
        "wrong_source_in_correct_article",
        "wrong_source_in_wrong_article",
        # True Positive, but would classify as FP in query-based analysis
        "related_source_in_wrong_article",
    ]
    manual_results_path = os.path.join(base_path, "results.csv")
    processed = set(pd.read_csv(manual_results_path)['row_index']) if os.path.exists(manual_results_path) else set()


    pbar = tqdm.tqdm(df.itertuples(), total=len(df), desc="[3/4] Manually evaluating alignment...")

    for row in pbar:
        if row.Index in processed:
            continue

        await manually_evaluate_summary(row, error_modes, manual_results_path)

    print("[4/4] Analysing the results...")
    analyse_results(df, pd.read_csv(manual_results_path), error_modes)

    print("\n\nDone.")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())