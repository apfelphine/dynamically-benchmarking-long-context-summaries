import os
from abc import ABC
from typing import Dict, Any, List

import numpy as np
import tiktoken

from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.query_based_summaries.evaluation.base_evaluator import QueryBasedSummaryBaseEvaluator
from models.corpus import Bin, QueryBasedSummary
from models.document_splitting import Chunk
from utils import vis
from utils.async_json import json_load
from utils.partitioning import partition_by_cumulative_sum

from utils.pretty_table import pretty_table


class CorrectDocUsageEvaluator(QueryBasedSummaryBaseEvaluator, ABC):
    """
    Evaluates how the model utilizes document content.

   - Valid Content = Focus Document only.
   - Any usage of other documents is a False Positive.
   - Partition analysis runs on the Focus Document.
    """

    N_PARTITIONS = 25
    ENC = tiktoken.get_encoding("cl100k_base")

    def __init__(
        self, context: QueryBasedSummaryContext, dir_name: str
    ):
        dir_name = os.path.join(
            "", dir_name
        )
        super().__init__(context, dir_name)

        self.context: QueryBasedSummaryContext = context  # Reassign for type hints

        self.metrics = {
            "f1_raw": r"$F1$" if vis.USE_LATEX else "F1",
            "precision_raw": r"$P$" if vis.USE_LATEX else "P",
            "recall_raw": r"$R$" if vis.USE_LATEX else "R",
        }

    def print_metrics(self, df, aggregate_by: List[str], title: str):
        pretty_table(
            df,
            aggregate_by,
            [
                {"col": m, "header": m, "high_is_good": True}
                for m in self.metrics.keys()
            ],
            title,
        )

    def skip_summary(self, summary: QueryBasedSummary):
        return False


    async def evaluate_single_summary(
        self,
        b: Bin,
        summary_idx: str,
        summary: QueryBasedSummary,
        data: Dict,
        summarizer_name: str,
        llm_name: str,
    ):
        if self.skip_summary(summary):
            return None

        relevant_mask, chunk_tokens, document_tokens = data[summary_idx]

        used_indices = await self.get_used_chunk_indices(
            summary, summarizer_name, llm_name
        )
        y_pred_all = np.zeros(len(relevant_mask), dtype=bool)
        y_pred_all[used_indices] = True

        target_mask = relevant_mask
        recall_basis_mask = relevant_mask

        global_metrics = self.calculate_precision_recall_f1(
            y_pred_all, target_mask, recall_basis_mask
        )

        data_out: Dict[str, Any] = {
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "bin_idx": summary.bin_idx,
            "bin_name": b.name,
            **global_metrics,
            "partition_data": [],
            "chunk_order_data": None,
            "is_distributed": False,
        }

        partition_indices = partition_by_cumulative_sum(chunk_tokens, self.N_PARTITIONS)
        total_valid_global = np.sum(recall_basis_mask)

        for p_id in range(self.N_PARTITIONS):
            p_mask = partition_indices == p_id
            p_pred = y_pred_all[p_mask]
            p_target = target_mask[p_mask]
            p_basis = recall_basis_mask[p_mask]

            # Calculate metrics for this specific partition
            p_metrics = self.calculate_precision_recall_f1(
                p_pred, p_target, p_basis
            )

            # Weight: How much of the valid content is in this partition?
            total_valid_p = np.sum(p_basis)
            weight = total_valid_p / total_valid_global if total_valid_global > 0 else 0

            data_out["partition_data"].append({"weight": weight, **p_metrics})

        if summary.chunk_order:
            data_out["is_distributed"] = True

            chunk_data = []

            focus_indices = self.context.get_focus_indices(
                len(summary.document_order),
                len(summary.chunk_order),
            )
            for idx, article_section_idx in enumerate(summary.chunk_order):
                doc_idx = focus_indices[idx]

                # Mask for the specific document
                doc_scope_mask = np.array(
                    [c.document_idx == doc_idx for c in document_tokens]
                )

                # Slice arrays for this document
                d_pred = y_pred_all[doc_scope_mask]
                d_target = target_mask[doc_scope_mask]
                d_basis = recall_basis_mask[doc_scope_mask]

                # Calculate metrics for this specific document part
                d_metrics = self.calculate_precision_recall_f1(
                    d_pred, d_target, d_basis
                )

                chunk_data.append(
                    {"article_section_idx": article_section_idx, **d_metrics}
                )
            data_out["chunk_order_data"] = chunk_data

        return data_out

    async def evaluate_bin(
        self,
        bin_idx: int,
        b: Bin,
        summarizer_name: str,
        llm_name: str,
        bin_results: List[Dict],
        prepared_data,
    ):
        # filter out empty results
        return [r for r in bin_results if r]

    async def prepare_data(self):
        """Loads chunks and determines relevant/related masks."""
        # Preload graph structure only if we need related masks
        summary_data = {}
        for summary_idx, summary in enumerate(self.context.summaries):
            path = self.context.get_save_path(summary)
            chunks_data = await json_load(os.path.join(path, "document_chunks.json"))
            document_chunks = [Chunk(**c) for c in chunks_data]
            chunk_tokens = [len(self.ENC.encode(c.text)) for c in document_chunks]

            doc_order = self.context.get_document_order(summary)

            relevant_mask = np.array(
                [
                    doc_order[c.document_idx] == summary.focus_doc_idx
                    for c in document_chunks
                ]
            )

            summary_data[summary_idx] = (
                relevant_mask,
                chunk_tokens,
                document_chunks,
            )

        return summary_data

    def calculate_precision_recall_f1(
        self, y_pred, target_mask, recall_basis_mask
    ):
        # Relevant = Predicted AND Valid (Intersection)
        pred_relevant = y_pred[recall_basis_mask]

        # Irrelevant = Predicted AND Invalid (Predicted but NOT in Target)
        pred_irrelevant = y_pred[~target_mask]

        # Total Valid = Count of all valid items in this scope
        total_valid = np.sum(recall_basis_mask)

        # RAW COUNTS
        tp = np.sum(pred_relevant)
        fp = np.sum(pred_irrelevant)
        p, r, f1 = self.calculate_prf(tp, fp, total_valid)

        return {
            "precision_raw": p,
            "recall_raw": r,
            "f1_raw": f1,
        }

    @staticmethod
    def calculate_prf(tp, fp, total_positives):
        """
        Helper to calculate Precision, Recall, and F1.
        """
        total_pred = tp + fp

        precision = tp / total_pred if total_pred > 1e-9 else 0.0
        recall = tp / total_positives if total_positives > 0 else 0.0

        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1
