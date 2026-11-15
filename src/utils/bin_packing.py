from typing import List, Dict

import numpy as np

from models.corpus import Document

MAX_TOKENS = 100_000
MIN_TOKENS = 90_000

STD_FACTOR = 1
MAX_DOCS_PER_BIN = 20
MIN_DOCS_PER_BIN = 4


def __filter_bins_by_min_tokens(
    bins: List[List[Document]],
    min_tokens: int = MIN_TOKENS,
) -> tuple[List[List[Document]], List[Document]]:
    discarded_documents = []
    good_bins = []
    for b in bins:
        bin_size = sum(x.document.tokens for x in b)
        if bin_size < min_tokens or len(b) < MIN_DOCS_PER_BIN:
            discarded_documents.extend(b)
        else:
            good_bins.append(b)

    return good_bins, discarded_documents


def first_fit_decreasing(
    documents: List[Document],
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    max_docs_per_bin: int = MAX_DOCS_PER_BIN,
) -> tuple[List[List[Document]], List[Document]]:
    # Sort groups in decreasing order
    documents = sorted(documents, reverse=True, key=lambda x: x.document.tokens)

    bins = []  # A list of lists

    # Place each item in the first bin that can fit it
    for doc in documents:
        placed = False
        for b in bins:
            if max_docs_per_bin and len(b) > max_docs_per_bin:
                continue
            current_bin_group = sum(x.document.tokens for x in b)
            if current_bin_group + doc.document.tokens <= max_tokens:
                b.append(doc)
                placed = True
                break
        if not placed:
            bins.append([doc])

    return __filter_bins_by_min_tokens(bins, min_tokens)


def custom_first_fit_decreasing(
    documents: List[Document],
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    max_docs_per_bin: int = MAX_DOCS_PER_BIN,
) -> tuple[List[List[Document]], List[Document]]:
    # Sort groups in decreasing order
    documents = sorted(documents, reverse=True, key=lambda x: x.document.tokens)

    # Initialize list of bins (each bin is a list of groups)
    bins: List[List[Document]] = []

    # Place each doc in the first bin that can fit it
    for doc in documents:
        doc_size = doc.document.tokens

        first_even_full_bin = None  # Priority 1: Bin that will be full AND even
        even_fit_bin = None  # Priority 2: Bin fits AND will be even
        odd_fit_bin = (
            None  # Priority 3: Bin that fits (and not be full yet) BUT will be odd
        )

        # Find the best bin
        for b in bins:
            if max_docs_per_bin and len(b) > max_docs_per_bin:
                continue
            bin_size = sum(x.document.tokens for x in b)

            if bin_size + doc_size <= max_tokens:

                if len(b) % 2 != 0:
                    if bin_size + doc_size >= min_tokens:
                        first_even_full_bin = b
                        break
                    elif not even_fit_bin:
                        even_fit_bin = b
                else:
                    if bin_size + doc_size < min_tokens and odd_fit_bin is None:
                        odd_fit_bin = b

        if first_even_full_bin:
            first_even_full_bin.append(doc)
        elif even_fit_bin:
            even_fit_bin.append(doc)
        elif odd_fit_bin:
            odd_fit_bin.append(doc)
        else:
            bins.append([doc])

    return __filter_bins_by_min_tokens(bins, min_tokens)


def best_fit_decreasing(
    documents: List[Document],
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    max_docs_per_bin: int = MAX_DOCS_PER_BIN,
) -> tuple[List[List[Document]], List[Document]]:
    """
    Packs documents using the Best Fit Decreasing (BFD) algorithm.
    """
    # Sort groups in decreasing order (the "Decreasing" part)
    documents = sorted(documents, reverse=True, key=lambda x: x.document.tokens)

    bins: List[List[Document]] = []

    for document in documents:
        doc_size = document.document.tokens

        best_bin_index = -1
        min_remaining_space = max_tokens + 1  # "worst" possible fit

        for i, b in enumerate(bins):
            bin_size = sum(x.document.tokens for x in b)
            if max_docs_per_bin and len(b) > max_docs_per_bin:
                continue

            if bin_size + doc_size <= max_tokens:
                remaining_space = max_tokens - (bin_size + doc_size)

                if remaining_space < min_remaining_space:
                    min_remaining_space = remaining_space
                    best_bin_index = i

        if best_bin_index != -1:
            bins[best_bin_index].append(document)
        else:
            bins.append([document])

    # Filter out bins that are too small
    return __filter_bins_by_min_tokens(bins, min_tokens)


def __calculate_placement_score(
    new_bin_size: int,
    new_doc_count: int,
    min_tokens: int,
    max_tokens: int,
    max_docs_per_bin: int,
) -> float:
    if new_bin_size > max_tokens:  # Too many tokens
        return float("inf")

    if new_doc_count > max_docs_per_bin:  # Too many docs
        return float("inf")

    is_even = (new_doc_count % 2) == 0
    is_full = new_bin_size >= min_tokens
    remaining_space = float(max_tokens - new_bin_size)

    if is_full and not is_even:
        return float("inf")

    return remaining_space


def custom_best_fit_decreasing(
    documents: List[Document],
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    max_docs_per_bin: int = MAX_DOCS_PER_BIN,
) -> tuple[List[List[Document]], List[Document]]:
    # Sort groups in decreasing order
    documents = sorted(documents, reverse=True, key=lambda x: x.document.tokens)

    bins: List[List[Document]] = []

    for doc in documents:
        doc_size = doc.document.tokens

        best_bin_index = -1
        best_score = float(
            "inf"
        )  # Looking for the lowest score, so any score is better than inf

        # Find the bin with the "best" score
        for i, b in enumerate(bins):
            bin_size = sum(x.document.tokens for x in b)
            new_size = bin_size + doc_size
            new_doc_count = len(bins[i]) + 1

            score = __calculate_placement_score(
                new_size, new_doc_count, min_tokens, max_tokens, max_docs_per_bin
            )

            if score < best_score:
                best_score = score
                best_bin_index = i

        if best_bin_index != -1:
            bins[best_bin_index].append(doc)
        else:
            bins.append([doc])

    # Filter out bins that are still too small
    return __filter_bins_by_min_tokens(bins, min_tokens)


def __cluster_documents(
    documents: List[Document], std_factor: float = STD_FACTOR
) -> List[List[Document]]:
    mean = np.mean([d.document.tokens for d in documents])
    std = np.std([d.document.tokens for d in documents])
    small_thresh = mean - (std_factor * std)
    large_thresh = mean + (std_factor * std)

    clusters: Dict[str, List] = {"small": [], "medium": [], "large": []}

    for doc in documents:
        doc_value = doc.document.tokens
        if doc_value < small_thresh:
            clusters["small"].append(doc)
        elif doc_value > large_thresh:
            clusters["large"].append(doc)
        else:
            clusters["medium"].append(doc)

    return list(clusters.values())


def clustered_bin_packing(
    documents: List[Document],
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    std_factor: float = STD_FACTOR,
    bin_packing_funct=custom_best_fit_decreasing,
) -> tuple[List[List[Document]], List[Document]]:
    # Cluster the documents
    clusters = __cluster_documents(documents, std_factor)

    # Pack each cluster independently
    packed_bins = []
    discarded_documents = []

    # Pack each cluster separately first
    for idx, c in enumerate(clusters):
        bins, discarded = bin_packing_funct(c, max_tokens, min_tokens)
        packed_bins.extend(bins)
        discarded_documents.extend(discarded)

    # Try to pack all discarded docs together
    if discarded_documents:
        bins, discarded = bin_packing_funct(discarded_documents, max_tokens, min_tokens)
        packed_bins.extend(bins)
        discarded_documents = discarded

    return packed_bins, discarded_documents


bin_packing = custom_best_fit_decreasing
