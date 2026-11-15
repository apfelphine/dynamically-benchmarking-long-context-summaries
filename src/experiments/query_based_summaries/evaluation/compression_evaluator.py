import asyncio
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import tiktoken
from black.trans import defaultdict

from experiments.configuration import base_data_dir
from experiments.query_based_summaries.context import QueryBasedSummaryContext
from experiments.experiment_evaluator import ExperimentEvaluator
from models.corpus import Bin, Summary, Document
from models.document_splitting import Chunk
from utils.async_json import json_load
from utils.pretty_table import pretty_table


class Experiment2CompressionEvaluator(ExperimentEvaluator):
    ENC = tiktoken.get_encoding("cl100k_base")

    def __init__(self, context: QueryBasedSummaryContext, only_found: bool = False):
        super().__init__(context, "summary_compression")
        self.only_found = only_found

    async def prepare_data(self):
        focus_doc_data = defaultdict(dict)
        relevant_mask_data = {}

        for summary_idx, summary in enumerate(self.context.summaries):
            path = self.context.get_save_path(summary)
            chunks_data = await json_load(os.path.join(path, "document_chunks.json"))
            document_chunks = [Chunk(**c) for c in chunks_data]

            doc_order = self.context.get_document_order(summary)

            relevant_mask = np.array(
                [
                    doc_order[c.document_idx] == summary.focus_doc_idx
                    for c in document_chunks
                ]
            )

            relevant_mask_data[summary_idx] = relevant_mask

            focus_doc_idx = summary.focus_doc_idx
            if (
                summary.bin_idx in focus_doc_data
                and focus_doc_idx in focus_doc_data[summary.bin_idx]
            ):
                continue
            document_path = self.context.corpus[summary.bin_idx].document_paths[
                focus_doc_idx
            ]
            doc = Document(
                **await json_load(
                    os.path.join(base_data_dir, document_path, "metadata.json")
                )
            )
            focus_doc_data[summary.bin_idx][focus_doc_idx] = doc

        return focus_doc_data, relevant_mask_data


    async def evaluate_single_summary(
        self,
        b: Bin,
        summary_idx: str,
        summary: Summary,
        data: Dict,
        summarizer_name: str,
        llm_name: str,
    ):
        summary_text, _, binary_sim_matrix, _ = await self._load_llm_data(
            summary, summarizer_name, llm_name
        )
        focus_doc_data, relevant_mask_data = data

        relevant_mask = relevant_mask_data[summary_idx]
        chunk_usage = binary_sim_matrix.any(axis=1).astype(int)
        used_indices = np.where(chunk_usage == 1)[0].tolist()
        y_pred_all = np.zeros(len(relevant_mask), dtype=bool)
        y_pred_all[used_indices] = True
        found = bool(np.any(y_pred_all[relevant_mask]))

        doc: Document = focus_doc_data[summary.bin_idx][summary.focus_doc_idx]
        summary_tokens = len(self.ENC.encode(summary_text))

        return {
            "bin_idx": summary.bin_idx,
            "bin_name": b.name,
            "summary_idx": summary_idx,
            "summarizer_name": summarizer_name,
            "llm_name": llm_name,
            "summary_tokens": summary_tokens,
            "document_tokens": doc.document.tokens,
            # Compression Rate
            # 1.0: Summary has same length as document
            # 0.5: Summary is half the length of the document
            "compression_rate": summary_tokens / doc.document.tokens,
            "found": found
        }

    async def evaluate_overall(self, overall_results, prepared_data):
        df = pd.DataFrame(overall_results)
        if self.only_found:
            df = df[df["found"] == True]

        df.to_csv(
            os.path.join(self.overall_analysis_dir, "compression_rate_raw.csv"),
            index=False,
        )

        self.print_metrics(df, ["summarizer_name", "llm_name"], "Übersicht")
        self.print_metrics(
            df, ["summarizer_name"], "Analyse pro Zusammenfassungsstrategie"
        )
        self.print_metrics(df, ["llm_name"], "Analyse pro Sprachmodell")
        self.print_metrics(df, ["bin_name"], "Analyse pro Kollektion")

        grouped_metrics = self.group_df_and_calculate_correlation(
            df, ["summarizer_name", "llm_name"]
        )
        grouped_metrics.to_csv(
            os.path.join(
                self.overall_analysis_dir, "compression_rate_by_llm_strategy.csv"
            ),
            index=False,
        )

        grouped_metrics = self.group_df_and_calculate_correlation(
            df, ["bin_idx", "bin_name"]
        )
        grouped_metrics.to_csv(
            os.path.join(self.bin_analysis_dir, "compression_rate_by_bin.csv"),
            index=False,
        )

    @staticmethod
    def print_metrics(df, aggregate_by: List[str], title: str):
        df_print = df.copy()
        df_print["cr_std"] = df_print["compression_rate"]
        df_print["cr_min"] = df_print["compression_rate"]
        df_print["cr_p10"] = df_print["compression_rate"]
        df_print["cr_p25"] = df_print["compression_rate"]
        df_print["cr_p50"] = df_print["compression_rate"]
        df_print["cr_p75"] = df_print["compression_rate"]
        df_print["cr_p90"] = df_print["compression_rate"]
        df_print["cr_max"] = df_print["compression_rate"]

        my_metrics_config = [
            # Main Mean
            {
                "col": "compression_rate",
                "header": "Mean %",
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            # Consistency
            {
                "col": "cr_std",
                "header": "Std Dev",
                "agg": "std",
                "mult": 100,
                "unit": "%",
                "decimals": 1,
                "high_is_good": False,
            },
            # Percentiles & Range (using alias columns to prevent key collision)
            {
                "col": "cr_min",
                "header": "Min %",
                "agg": "min",
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_p10",
                "header": "P10 %",
                "agg": lambda x: x.quantile(0.10),
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_p25",
                "header": "P25 %",
                "agg": lambda x: x.quantile(0.25),
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_p50",
                "header": "Median %",
                "agg": lambda x: x.quantile(0.50),
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_p75",
                "header": "P75 %",
                "agg": lambda x: x.quantile(0.75),
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_p90",
                "header": "P90 %",
                "agg": lambda x: x.quantile(0.90),
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
            {
                "col": "cr_max",
                "header": "Max %",
                "agg": "max",
                "mult": 100,
                "unit": "%",
                "high_is_good": False,
            },
        ]

        pretty_table(
            df=df_print,
            group_cols=aggregate_by,
            metrics_config=my_metrics_config,
            title=title,
        )

    @staticmethod
    def group_df_and_calculate_correlation(
        df: pd.DataFrame, aggregate_columns: List[str]
    ):
        grouped_means = (
            df.groupby(aggregate_columns)[
                ["summary_tokens", "document_tokens", "compression_rate"]
            ]
            .mean()
            .reset_index()
        )
        grouped_corr = (
            df.groupby(aggregate_columns)
            .apply(
                lambda x: x["document_tokens"].corr(x["summary_tokens"]),
                include_groups=False,
            )
            .reset_index(name="doc_summary_correlation")
        )
        grouped_metrics = pd.merge(grouped_means, grouped_corr, on=aggregate_columns)
        return grouped_metrics


async def main():
    evaluator = Experiment2CompressionEvaluator(QueryBasedSummaryContext(), only_found=True)
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())