import asyncio
import os
from typing import List

from colorama import Fore, Style

from experiments.configuration import base_experiment_dir, base_data_dir
from models.corpus import Document, Bin
from utils.async_json import pretty_json_dump, json_load
from utils.bin_packing import bin_packing

os.makedirs(base_experiment_dir, exist_ok=True)


async def main():
    created_bins: List[Bin] = []

    for data_topic in os.listdir(base_data_dir):
        dataset_path = os.path.join(base_data_dir, data_topic)
        if not os.path.isdir(dataset_path) or data_topic == "__pycache__":
            continue

        print(
            f"{Fore.CYAN}{'=' * 15} {Style.BRIGHT}Topic: {data_topic}{Style.RESET_ALL}{Fore.CYAN} {'=' * 15}{Style.RESET_ALL}\n"
        )

        data = []
        for doc in os.listdir(dataset_path):
            doc_path = os.path.join(dataset_path, doc)
            if not os.path.isdir(doc_path):
                continue

            metadata = await json_load(os.path.join(doc_path, "metadata.json"))
            data.append(Document(**metadata))

        document_bins, discarded_documents = bin_packing(data)
        sum_doc_tokens_discarded = sum(d.document.tokens for d in discarded_documents)
        print(
            Style.BRIGHT
            + Fore.GREEN
            + f"Number of docs: {Style.RESET_ALL}\t\t\t{len(data)}"
        )
        print(
            Style.BRIGHT
            + Fore.GREEN
            + f"Number of bins: {Style.RESET_ALL}\t\t\t{len(document_bins)}"
        )
        print(
            Style.BRIGHT
            + Fore.GREEN
            + f"Number of discarded docs: {Style.RESET_ALL}\t{len(discarded_documents)} ({sum_doc_tokens_discarded} tokens)"
        )
        documents_in_bins = sum(len(b) for b in document_bins)
        print(
            Style.BRIGHT
            + Fore.GREEN
            + f"Number of docs in bins: {Style.RESET_ALL}\t{documents_in_bins}"
        )
        print(
            Style.BRIGHT
            + Fore.GREEN
            + f"% of docs used: {Style.RESET_ALL}\t\t\t{documents_in_bins/len(data)*100:.2f}%\n"
        )

        if not document_bins:
            print(f"{Fore.RED}All documents discarded. No bins created.\n")
            continue

        for bin_idx, document_bin in enumerate(document_bins):
            created_bins.append(
                Bin(
                    topic=data_topic,
                    name=f"{data_topic}"
                    + (f" (Bin: {bin_idx})" if len(document_bins) > 1 else ""),
                    doc_tokens=sum(d.document.tokens for d in document_bin),
                    sum_tokens=sum(d.summary.tokens for d in document_bin),
                    document_paths=[d.path for d in document_bin],
                )
            )

    await pretty_json_dump(
        os.path.join(base_experiment_dir, "corpus.json"),
        [d.model_dump() for d in created_bins],
    )


if __name__ == "__main__":
    asyncio.run(main())
