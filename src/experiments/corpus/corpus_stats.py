import datetime
import json
import os
import numpy as np
from typing import List, Dict, Optional
from colorama import Style, Fore

from core.wikipedia.wiki_crawler import WikipediaCrawler
from experiments.configuration import base_experiment_dir, base_data_dir
from models.corpus import Document, Bin

download_date = datetime.datetime(2025, 11, 24, 21, 21, 55)

def main():
    # Analyze raw data and get the counts per category
    raw_category_counts = analyse_extracted_articles()

    # Analyze the corpus and calculate discarded docs using the raw counts
    analyse_corpus(raw_category_counts)


def analyse_extracted_articles() -> Dict[str, int]:
    """
    Scans the raw data directory, prints statistics, and returns
    a dictionary of {category_name: total_document_count}.
    """
    print(f"{Fore.CYAN}Scanning raw articles in {os.path.abspath(base_data_dir)}...")

    # Structure: { 'category_name': {'doc_tokens': [], 'sum_tokens': []} }
    category_stats: Dict[str, Dict[str, List[int]]] = {}

    raw_counts: Dict[str, int] = {}

    for category_name in os.listdir(base_data_dir):
        category_path = os.path.join(base_data_dir, category_name)
        if not os.path.isdir(category_path):
            continue

        if category_name not in category_stats:
            category_stats[category_name] = {"doc_tokens": [], "sum_tokens": []}
            raw_counts[category_name] = 0

        for item_name in os.listdir(category_path):
            item_path = os.path.join(category_path, item_name)
            if not os.path.isdir(item_path):
                continue
            raw_counts[category_name] += 1

            # Read metadata for detailed token stats
            metadata_file = os.path.join(item_path, "metadata.json")
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = Document(**json.load(f))

            category_stats[category_name]["doc_tokens"].append(metadata.document.tokens)
            category_stats[category_name]["sum_tokens"].append(metadata.summary.tokens)

    print(
        f"\n{Fore.MAGENTA}{'>' * 5} {Style.BRIGHT} RAW ARTICLES PER CATEGORY {Style.RESET_ALL} {Fore.MAGENTA}{'<' * 5}{Style.RESET_ALL}\n"
    )

    total_docs_all = 0

    for category, stats in category_stats.items():
        n_docs = len(stats["doc_tokens"])
        total_docs_all += n_docs

        sum_doc_tokens = sum(stats["doc_tokens"])
        sum_sum_tokens = sum(stats["sum_tokens"])

        avg_doc_tokens = np.mean(stats["doc_tokens"]) if n_docs > 0 else 0
        avg_sum_tokens = np.mean(stats["sum_tokens"]) if n_docs > 0 else 0

        print(
            Fore.BLUE + Style.BRIGHT + f"Category: {category}\n{Style.RESET_ALL}"
            f"> Number of Docs: \t{n_docs}\n"
            f"> Total Doc Tokens: \t{sum_doc_tokens} "
            f"{Fore.WHITE}\t(Avg: {avg_doc_tokens:.1f}){Style.RESET_ALL}\n"
            f"> Total Sum Tokens: \t{sum_sum_tokens} "
            f"{Fore.WHITE}\t(Avg: {avg_sum_tokens:.1f}){Style.RESET_ALL}\n"
        )

    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total processed documents: {total_docs_all}{Style.RESET_ALL}\n"
    )

    return raw_counts


def analyse_corpus(raw_category_counts: Optional[Dict[str, int]] = None):
    """
    Analyzes the Bins/Summaries and calculates discarded documents
    by comparing against raw_category_counts.
    """
    print(f"{Fore.CYAN}Loading corpus and summaries...")
    with open(os.path.join(base_experiment_dir, "corpus.json"), "r") as f:
        corpus: List[Bin] = [Bin(**b) for b in json.load(f)]

    with open(os.path.join(base_experiment_dir, "generic_summaries.json"), "r") as f:
        generic_summaries = json.load(f)
    with open(os.path.join(base_experiment_dir, "query_based_summaries.json"), "r") as f:
        query_based_summaries = json.load(f)

    total_n_bins = len(corpus)
    total_n_docs = sum(len(b.document_paths) for b in corpus)

    print(
        f"\n{Fore.MAGENTA}{'>' * 5} {Style.BRIGHT} EXPERIMENT OVERVIEW {Style.RESET_ALL} {Fore.MAGENTA}{'<' * 5}{Style.RESET_ALL}\n"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total number of bins: \t{Style.RESET_ALL}{total_n_bins}"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total number of docs: \t{Style.RESET_ALL}{total_n_docs}"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total summaries: \t\t{Style.RESET_ALL}{len(query_based_summaries) + len(generic_summaries)}"
    )

    if raw_category_counts:
        print(
            f"\n{Fore.MAGENTA}{'>' * 5} {Style.BRIGHT} DISCARDED DOCUMENTS ANALYSIS {Style.RESET_ALL} {Fore.MAGENTA}{'<' * 5}{Style.RESET_ALL}\n"
        )

        # Calculate used docs per topic from the bins
        used_counts: Dict[str, int] = {}
        for b in corpus:
            topic = b.topic
            if topic not in used_counts:
                used_counts[topic] = 0
            used_counts[topic] += len(b.document_paths)

        # Merge keys from both raw and used to ensure we cover everything
        all_categories = set(raw_category_counts.keys()) | set(used_counts.keys())

        # Header
        print(
            f"{Style.BRIGHT}{'CATEGORY':<30} {'RAW':<10} {'USED':<10} {'DISCARDED':<12} {'% LOST':<10}{Style.RESET_ALL}"
        )
        print("-" * 70)

        for cat in sorted(all_categories):
            raw = raw_category_counts.get(cat, 0)
            used = used_counts.get(cat, 0)
            discarded = raw - used

            # Formatting color based on discard rate
            if raw > 0:
                pct_lost = (discarded / raw) * 100
            else:
                pct_lost = 0.0

            color = Fore.GREEN
            if pct_lost > 0:
                color = Fore.YELLOW
            if pct_lost > 50:
                color = Fore.RED

            print(
                f"{color}{cat:<30} {raw:<10} {used:<10} {discarded:<12} {pct_lost:>6.1f}%{Style.RESET_ALL}"
            )

    print(
        f"\n{Fore.MAGENTA}{'>' * 5} {Style.BRIGHT} BIN DETAILS {Style.RESET_ALL} {Fore.MAGENTA}{'<' * 5}{Style.RESET_ALL}\n"
    )
    crawler = WikipediaCrawler()
    all_revisions = []

    for idx, b in enumerate(corpus):
        documents = []
        last_revision_dates = []

        for doc_path in b.document_paths:
            try:
                with open(
                    os.path.join(base_data_dir, doc_path, "metadata.json"),
                    encoding="utf-8",
                ) as f:
                    d = Document(**json.load(f))

                last_revision_date = crawler.get_last_revision_before(d.title, download_date)
                documents.append(d)
                last_revision_dates.append(last_revision_date)
            except FileNotFoundError:
                print(
                    f"{Fore.RED}Warning: Could not find metadata for {doc_path}{Style.RESET_ALL}"
                )

        # Safely calculate stats only if documents exist
        if documents:
            doc_toks = [d.document.tokens for d in documents]
            sum_toks = [d.summary.tokens for d in documents]

            last_revision_dates = [r for r in last_revision_dates if r is not None]
            print(
                Fore.BLUE
                + Style.BRIGHT
                + f"Document Bin #{idx} - {b.name}: \n{Style.RESET_ALL}"
                f"> Number of Docs: \t{len(b.document_paths)}\n"
                f"> Bin Doc Tokens: \t{b.doc_tokens}\n"
                f"{Fore.WHITE}> Mean (Doc): \t\t{np.mean(doc_toks):.4f}\n"
                f"> Std (Doc): \t\t{np.std(doc_toks):.4f}\n"
                f"> Min (Doc): \t\t{np.min(doc_toks):.4f}\n"
                f"> Max (Doc): \t\t{np.max(doc_toks):.4f}\n{Style.RESET_ALL}"
                f"> Bin Sum Tokens: \t{b.sum_tokens}\n"
                f"{Fore.WHITE}> Mean (Sum): \t\t{np.mean(sum_toks):.4f}\n"
                f"> Std (Sum): \t\t{np.std(sum_toks):.4f}\n"
                f"> Min (Sum): \t\t{np.min(sum_toks):.4f}\n"
                f"> Max (Sum): \t\t{np.max(sum_toks):.4f}\n{Style.RESET_ALL}"
                f"> Oldest article: \t{min(last_revision_dates)}\n"
                f"> Newest article: \t{max(last_revision_dates)}\n"
            )
            #
            # for i, d in enumerate(documents):
            #     print(d.title, ": ", last_revision_dates[i])

            all_revisions.extend(last_revision_dates)

    print(
        f"\n{Fore.MAGENTA}{'>' * 5} {Style.BRIGHT} SUMMARIES {Style.RESET_ALL} {Fore.MAGENTA}{'<' * 5}{Style.RESET_ALL}\n"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total number of generic summaries: \t{Style.RESET_ALL}{len(generic_summaries)}"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total number of query-based summaries: \t{Style.RESET_ALL}{len(query_based_summaries)}"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Total number of summaries: \t\t{Style.RESET_ALL}{len(query_based_summaries) + len(generic_summaries)}"
    )
    all_revisions = [r for r in all_revisions if r is not None]
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Oldest / last edited article: \t\t{Style.RESET_ALL}{min(all_revisions)}"
    )
    print(
        f"{Style.BRIGHT}{Fore.GREEN}Newest / most recently edited article: \t{Style.RESET_ALL}{max(all_revisions)}"
    )

if __name__ == "__main__":
    main()
