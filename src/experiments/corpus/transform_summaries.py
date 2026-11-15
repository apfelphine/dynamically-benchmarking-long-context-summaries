import asyncio
import os
from typing import List


from utils.latin_square import balanced_latin_square
from experiments.configuration import base_experiment_dir, base_data_dir
from models.corpus import Document, Bin, QueryBasedSummary, GenericSummary
from utils.async_json import pretty_json_dump, json_load

os.makedirs(base_experiment_dir, exist_ok=True)


async def main():
    corpus: List[Bin] = [
        Bin(**s)
        for s in await json_load(os.path.join(base_experiment_dir, "corpus.json"))
    ]

    generic_summaries: List[GenericSummary] = []
    query_based_summaries: List[QueryBasedSummary] = []

    for bin_idx, document_bin in enumerate(corpus):
        loaded_documents = [
            Document(**await json_load(os.path.join(base_data_dir, d)))
            for d in document_bin.document_paths
        ]
        doc_indices = list(range(len(loaded_documents)))
        latin_square = balanced_latin_square(doc_indices)

        if len(doc_indices) < 3:
            focus_document_indices = doc_indices
        else:
            sorted_documents = sorted(
                [
                    {"idx": idx, "len": d.document.tokens}
                    for idx, d in enumerate(loaded_documents)
                ],
                key=lambda x: x["len"],
            )
            middle_index = (len(sorted_documents) - 1) // 2
            focus_document_indices = [
                sorted_documents[0]["idx"],
                sorted_documents[middle_index]["idx"],
                sorted_documents[-1]["idx"],
            ]

        for focus_idx in focus_document_indices:
            for row in latin_square:
                query_based_summaries.append(
                    QueryBasedSummary(
                        bin_idx=bin_idx,
                        focus_doc_idx=focus_idx,
                        document_order=row,
                        chunk_order=None,
                    )
                )

            # Target documents are segmented in 4 pieces to avoid high complexity
            segment_count = 4
            for chunk_row in balanced_latin_square(list(range(segment_count))):
                query_based_summaries.append(
                    QueryBasedSummary(
                        bin_idx=bin_idx,
                        focus_doc_idx=focus_idx,
                        document_order=[
                            doc_idx
                            for doc_idx in doc_indices
                            if not doc_idx == focus_idx
                        ],
                        chunk_order=chunk_row,
                    )
                )

        for row in latin_square:
            generic_summaries.append(GenericSummary(bin_idx=bin_idx, document_order=row))

    await pretty_json_dump(
        os.path.join(base_experiment_dir, "generic_summaries.json"),
        [d.model_dump() for d in generic_summaries],
    )
    await pretty_json_dump(
        os.path.join(base_experiment_dir, "query_based_summaries.json"),
        [d.model_dump() for d in query_based_summaries],
    )


if __name__ == "__main__":
    asyncio.run(main())
