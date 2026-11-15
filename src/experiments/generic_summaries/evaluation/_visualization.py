from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from black.trans import defaultdict
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os


from models.corpus import Bin
from utils import vis
from utils.normalization import normalize_matrix
from utils.partitioning import partition_by_cumulative_sum
from utils.vis import bold, wrap_text


def plot_bin_heatmap(
    doc_usages: List[np.array],
    doc_orders: List[List[int]],
    b: Bin,
    sub_titles: List[str] = None,
    save_path: Optional[str] = None,
) -> None:
    sub_titles = sub_titles if sub_titles else []
    assert len(doc_usages) == len(doc_orders)
    n_documents = len(b.document_names)
    d_matrix = np.zeros((n_documents, n_documents))
    annot_matrix = np.full((n_documents, n_documents), "", dtype=object)

    for i, doc_usage in enumerate(doc_usages):
        for doc_pos, doc_idx in enumerate(doc_orders[i]):
            if annot_matrix[doc_idx, doc_pos] == "":
                annot_matrix[doc_idx, doc_pos] = str(i)
            d_matrix[doc_idx, doc_pos] += doc_usage[doc_idx]

    plt.figure(figsize=(30, 15))

    is_empty_mask = annot_matrix == ""

    ax = sns.heatmap(
        d_matrix,
        cbar=False,
        cmap="RdYlGn",
        linewidths=0.005,
        linecolor="white",
        annot=annot_matrix,
        fmt="",
        mask=is_empty_mask,  # noqa
        annot_kws={"size": 24, "weight": "bold"},
        vmax=1,
        vmin=0,
    )

    ax.set_facecolor("lightgrey")

    title = "Document Utilization by Position"
    for t in sub_titles:
        title += r"\\" + t

    ax.set_title(
        bold(title),
        pad=30,
        fontsize=35,
        fontweight="bold",
    )

    ax.set_yticks(np.arange(n_documents) + 0.5)
    ax.set_yticklabels(
        b.document_names,
        rotation=30,
        fontsize=20,
        va="center",
    )
    ax.set_xlabel("Position", labelpad=15, fontsize=30)
    ax.tick_params(axis="x", labelsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_partition_usage(
    partitions: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    heatmap_data = partitions.reshape(1, -1)
    plt.figure(figsize=(12, 3))

    ax = sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",
        fmt=".0f",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        vmax=100,
        vmin=0,
    )

    title = "\n" + title if title else ""
    plt.title(
        f"Utilization by position (aggregated to {len(partitions)} bin){title}"
    )
    plt.xlabel("Relative position in the document")

    num_ticks = 11  # 0%, 10%, .. 100%
    position_ticks = np.linspace(0, len(partitions), num_ticks)
    logical_ticks = [f"{int(i)}%" for i in np.linspace(0, 100, num_ticks)]

    ax.set_xticks(position_ticks)
    ax.set_xticklabels(logical_ticks, rotation=0, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_doc_usage_frequency(
    doc_usages: List[np.array],
    b: Bin,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    counts: List[int] = np.sum(doc_usages, axis=0).tolist()
    labels: List[str] = b.document_names

    # Sort documents
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=sorted_labels,
        y=sorted_counts,
        hue=sorted_labels,
        palette="Blues_r",
        legend=False,
    )

    title = "\n" + title if title else ""
    plt.title(f"Most used documents{title}")
    plt.xlabel("Doc")
    plt.ylabel("Usage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_avg_metric(
    grouped_metrics: pd.DataFrame,
    metric: str,
    save_path: Optional[str] = None,
    max_value: Optional[int] = 1.1,
    metric_type: str = "Percentage",
):
    plt.figure(figsize=(14, 8))

    # Setup Data & Colors
    unique_summarizers = grouped_metrics["summarizer_name"].unique()
    unique_llms = grouped_metrics["llm_name"].unique()
    n_hues = len(unique_llms)

    palette = vis.PALETTE_MODELS[:n_hues]
    color_map = dict(zip(unique_llms, palette))

    # Calculate Bar Positions manually
    group_width = 0.8
    bar_width_raw = group_width / n_hues
    indices = np.arange(len(unique_summarizers))

    for i, llm in enumerate(unique_llms):
        # Filter data for this LLM
        llm_data = grouped_metrics[grouped_metrics["llm_name"] == llm]

        # Align data to the x-axis indices
        llm_data = (
            llm_data.set_index("summarizer_name").reindex(unique_summarizers).fillna(0)
        )

        x_positions = (
            indices - (group_width / 2) + (bar_width_raw / 2) + (i * bar_width_raw)
        )

        plt.bar(
            x_positions,
            llm_data[f"{metric}"],
            width=bar_width_raw,
            color=color_map[llm],
            edgecolor=None,
            zorder=2,  # Set zorder to ensure grid is behind, but lines are on top
        )

    # Calculate mean per summarizer group
    group_means = (
        grouped_metrics.groupby("summarizer_name")[metric]
        .mean()
        .reindex(unique_summarizers)
    )

    for idx, mean_val in zip(indices, group_means):
        # Calculate start and end X coordinates for the line (spanning the whole group)
        x_start = idx - (group_width / 2)
        x_end = idx + (group_width / 2)

        plt.hlines(
            y=mean_val,
            xmin=x_start,
            xmax=x_end,
            linestyles="--",
            colors="red",
            linewidth=2.5,
            zorder=3,  # Higher zorder creates the "overlay" effect on top of bars
        )

    # Formatting
    plt.ylabel(metric_type, fontsize=16)
    plt.xlabel("Summarization strategy", fontsize=16)

    if max_value:
        plt.ylim(0, max_value)

    plt.yticks(fontsize=14)
    plt.xticks(indices, unique_summarizers, rotation=0, ha="center", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # Create handles for the LLMs
    legend_handles = [Patch(facecolor=color_map[llm], label=llm) for llm in unique_llms]

    # Create handle for the Average Line
    avg_line_handle = Line2D([0], [0], color="red", linewidth=2.5, label="Average")

    # Add to legend list
    legend_handles.append(avg_line_handle)

    plt.legend(
        handles=legend_handles,
        loc="lower right",
        title="Legende",
        frameon=True,
        fontsize=16,
        title_fontsize=16,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")
    plt.close()


def plot_overall_partition_usage(
    df_wide: pd.DataFrame,
    save_path: Optional[str] = None,
    errorbar=("ci", 95),
    summary_strategies: Optional[List[str]] = None
):
    # Identify partition columns dynamically (e.g., 'partition_0', 'partition_1', ...)
    partition_cols = [c for c in df_wide.columns if c.startswith("partition_")]

    # Melt into "long" format for Seaborn
    # This turns:
    # | summarizer | llm | partition_0 | partition_1 |
    # into:
    # | summarizer | llm | partition_str | usage |
    df_long = df_wide.melt(
        id_vars=["summarizer_name", "llm_name"],
        value_vars=partition_cols,
        var_name="partition_str",
        value_name="usage",
    )

    if summary_strategies:
        df_long = df_long[df_long["summarizer_name"].isin(summary_strategies)]

    num_summarizers = df_long["summarizer_name"].nunique()
    plot_aspect = 1.5
    if num_summarizers > 1:
        plot_height = 3.45 * 2 / num_summarizers / plot_aspect
    else:
        plot_height = 3.45 / plot_aspect

    hue = "llm_name"
    style = "llm_name"
    col = "summarizer_name"
    col_title = "Summarization Strategy"

    # Clean up the partition index (convert "partition_5" -> 5)
    df_long["partition"] = df_long["partition_str"].str.extract(r"(\d+)").astype(int) + 1

    g = sns.relplot(
        data=df_long,
        x="partition",
        y="usage",
        col=col,
        hue=hue,
        style=style,
        kind="line",
        col_wrap=max(1, df_long[col].nunique()),
        height=plot_height,
        aspect=plot_aspect,
        markers=True,
        dashes=False,
        palette="paper_models",
        linewidth=2.5,
        facet_kws={"sharey": True, "sharex": True},
        legend=True,
        errorbar=errorbar,
    )

    g.set_axis_labels(
        f"Bucket (1-{len(partition_cols)})", r"Context Utilization Rate"
    )
    custom_ticks = list(range(5, len(partition_cols) + 1, 5))
    custom_ticks = [1] + custom_ticks
    g.set(xticks=custom_ticks)

    if num_summarizers > 1:
        g.set_titles(col_title + ": {col_name}")
    else:
        g.set_titles("")

    if g.legend:
        handles = g.legend.legend_handles
        labels = [t.get_text() for t in g.legend.texts]
        g.legend.remove()

        # Let the graph take up the normal full figure space
        g.figure.tight_layout()

        # Add custom legend OUTSIDE the bottom of the figure
        g.figure.legend(
            handles=handles,
            labels=labels,
            loc="upper center",  # Anchor the top of the legend box...
            bbox_to_anchor=(0.5, 0),  # ...to the absolute bottom (0) of the figure
            ncol=len(labels) if num_summarizers > 1 else len(labels) // 2,
            frameon=False,
            fontsize=12,
        )
    else:
        g.figure.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png", pad_inches=0.2)
        plt.savefig(save_path + ".pdf", pad_inches=0.2)
    plt.close()


def plot_chunk_usage(
    doc_orders: List[List[int]],
    chunk_usages,
    b,
    bin_chunks: Dict[int, list],
    save_path: Optional[str] = None,
):
    chunk_counts = {
        doc_idx: np.zeros(len(chunks)) for doc_idx, chunks in bin_chunks.items()
    }

    for doc_order, chunk_usage in zip(doc_orders, chunk_usages):
        current_pointer = 0
        for d_idx in doc_order:
            num_chunks = len(chunk_counts[d_idx])
            doc_slice = chunk_usage[current_pointer : current_pointer + num_chunks]
            chunk_counts[d_idx] += doc_slice
            current_pointer += num_chunks

    cycle_colors = vis.PALETTE_CONTRAST[:2]

    plot_counts = []
    plot_colors = []
    x_tick_labels = []
    x_tick_positions = []

    all_x_coords = []
    current_x = 0

    gap_size = 0.5

    sorted_doc_indices = sorted(chunk_counts.keys())

    for i, doc_idx in enumerate(sorted_doc_indices):
        counts = chunk_counts[doc_idx]
        doc_name = os.path.basename(b.document_paths[doc_idx])

        plot_counts.extend(counts)

        doc_x_positions = [current_x + k for k in range(len(counts))]
        all_x_coords.extend(doc_x_positions)

        assigned_color = cycle_colors[i % len(cycle_colors)]
        plot_colors.extend([assigned_color] * len(counts))

        # Calculate text position (centered over the block)
        mid_point = current_x + (len(counts) / 2) - 0.5
        x_tick_positions.append(mid_point)
        x_tick_labels.append(wrap_text(doc_name, max_rows=2))

        current_x += len(counts) + gap_size

    plt.subplots(figsize=(15, 6))

    plt.bar(
        all_x_coords, plot_counts, color=plot_colors, width=1.0, alpha=0.8, zorder=3
    )
    plt.ylabel("Segment Usage Count", fontweight="bold", color="black")

    plt.xticks(
        x_tick_positions,
        x_tick_labels,
        rotation=45,
        ha="center",
        color="black",
        fontweight="medium",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_partition_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    data_map = defaultdict(list)
    part_cols = [c for c in df.columns if c.startswith("partition_")]

    for row in df.itertuples(index=False):
        b_idx = row.bin_idx
        partition_values = [getattr(row, col) for col in part_cols]

        data_map[b_idx].append(partition_values)

    new_data_map = {}
    for b_idx, part_usages in data_map.items():
        new_data_map[b_idx] = np.mean(part_usages, axis=0)
    data_map = new_data_map

    # Sort by bin
    sorted_bins = sorted(data_map.keys())
    if not sorted_bins:
        return

    n_partitions = len(data_map[sorted_bins[0]])

    # Create matrix
    matrix = np.zeros((len(sorted_bins), n_partitions))
    for row_idx, b_idx in enumerate(sorted_bins):
        matrix[row_idx, :] = data_map[b_idx]

    # Plotting
    plt.figure(figsize=(15, len(sorted_bins) * 0.5 + 2))  # Höhe dynamisch anpassen

    sns.heatmap(
        matrix,
        cmap="RdYlGn",
        yticklabels=sorted_bins,
        xticklabels=range(n_partitions),
        cbar_kws={"label": "Usage (%)"},
        vmax=100,
        vmin=0,
        linewidths=0.5,
        linecolor="lightgrey",
    )

    title = "\n" + title if title else ""
    plt.title(f"Bucket usage by collection{title}", fontsize=16, pad=20)
    plt.xlabel("Bucket (Relative Position in document)", fontsize=12)
    plt.ylabel("Bin ID", fontsize=12)
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _accumulate_partitioned_usage(
    tokens: List[int],
    usage_slice: np.ndarray,
    n_partitions: int,
    partition_totals: np.ndarray,
):
    """Strategy for Discrete Partitioning."""
    p_indices = partition_by_cumulative_sum(tokens, n_partitions)
    np.add.at(partition_totals, p_indices, usage_slice)


def _accumulate_interpolated_usage(
    tokens: List[int],
    usage_slice: np.ndarray,
    n_partitions: int,
    partition_totals: np.ndarray,
):
    """Strategy for Linear Interpolation."""
    total_tokens = sum(tokens)
    if total_tokens == 0:
        return

    scale = n_partitions / total_tokens
    current_pos = 0.0

    for chunk_len, u_val in zip(tokens, usage_slice):
        start = current_pos * scale
        end = (current_pos + chunk_len) * scale
        current_pos += chunk_len

        if u_val == 0:
            continue

        chunk_span = end - start
        if chunk_span <= 1e-9:
            continue

        first_bin = int(start)
        last_bin = int(np.ceil(end))

        for b in range(first_bin, last_bin):
            if b < 0 or b >= n_partitions:
                continue

            bin_start = float(b)
            bin_end = float(b + 1)

            overlap_start = max(start, bin_start)
            overlap_end = min(end, bin_end)
            overlap_len = overlap_end - overlap_start

            if overlap_len > 0:
                weight = overlap_len / chunk_span
                partition_totals[b] += u_val * weight


def _process_chunk_usage_data(
    data: List[Dict], n_partitions: int, strategy_func: Callable
):
    partition_totals = np.zeros(n_partitions)
    unique_docs = set()

    for sum_data in data:
        bin_idx = sum_data["bin_idx"]
        doc_order = sum_data["doc_order"]
        usage_arr = sum_data["chunk_usage"]
        token_map = sum_data["chunk_tokens_by_doc"]

        current_ptr = 0

        for doc_idx in doc_order:
            tokens = token_map[doc_idx]
            n_chunks = len(tokens)
            usage_slice = usage_arr[current_ptr : current_ptr + n_chunks]
            current_ptr += n_chunks
            strategy_func(tokens, usage_slice, n_partitions, partition_totals)
            unique_docs.add(f"{bin_idx}_{doc_idx}")

    return partition_totals, len(unique_docs)


def plot_chunk_usage_overall_partitioned(
    data: List[Dict],
    n_partitions: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plots usage using DISCRETE PARTITIONING via helper.
    """
    partition_totals, n_docs = _process_chunk_usage_data(
        data, n_partitions, _accumulate_partitioned_usage
    )
    avg_usage = partition_totals / len(data)

    # Plotting
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("viridis", n_partitions)

    plt.bar(range(n_partitions), avg_usage, color=colors, width=0.9)

    base_title = f"Avg. usage by bucket\n{n_docs} Documents - {len(data)} Summaries"
    plt.title(title if title else base_title)
    plt.xlabel("Document Position")
    plt.ylabel("Count Usage")

    plt.xticks(
        [0, n_partitions // 2, n_partitions - 1],
        ["Start (0%)", "Middle (50%)", "End (100%)"],
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def plot_chunk_usage_overall_interpolated(
    data: List[Dict],
    n_partitions: int = 5,
    save_path: Optional[str] = None,
):
    partition_totals, n_docs = _process_chunk_usage_data(
        data, n_partitions, _accumulate_interpolated_usage
    )
    avg_usage = partition_totals / len(data)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(
        range(n_partitions),
        avg_usage,
        color=vis.PALETTE_MODELS[0],
        width=0.9,
        alpha=0.85,
        label="Usage per Segment",
    )

    plt.xlabel("Document Position (normalized)")
    plt.ylabel("Avg. Usage")

    plt.xticks(
        [0, n_partitions // 2, n_partitions - 1],
        ["Start (0%)", "Middle (50%)", "End (100%)"],
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png")
        plt.savefig(save_path + ".pdf")


def plot_chunk_usage_lines(
    data: List[Dict],
    bins: List,
    save_path: Optional[str] = None,
    include_ci: bool = False,
):
    plt.figure(figsize=(12, 5))
    topic_traces = defaultdict(list)

    for sum_data in data:
        bin_idx = sum_data["bin_idx"]
        topic = bins[bin_idx].topic

        doc_order = sum_data["doc_order"]
        usage_arr = np.array(sum_data["chunk_usage_raw"])
        token_map = sum_data["chunk_tokens_by_doc"]

        current_ptr = 0

        for doc_idx in doc_order:
            tokens = token_map[doc_idx]
            if not tokens:
                continue

            n_chunks = len(tokens)
            usage_slice = usage_arr[current_ptr : current_ptr + n_chunks]
            normalized_usage_slice = normalize_matrix(usage_slice)

            current_ptr += n_chunks

            if np.max(normalized_usage_slice) == 0:
                continue

            total_tokens = sum(tokens)
            if total_tokens == 0:
                continue

            tokens_arr = np.array(tokens)
            cumsum = np.cumsum(tokens_arr)
            starts = cumsum - tokens_arr
            centers = (starts + cumsum) / 2.0
            norm_centers = centers / total_tokens

            # Store raw trace grouped by TOPIC
            topic_traces[topic].append((norm_centers, normalized_usage_slice))

    # Define a common grid for averaging (e.g., 100 points)
    common_x = np.linspace(0, 1, 100)

    # Prepare Colors
    sorted_topics = sorted(topic_traces.keys())
    palette = vis.PALETTE_QUALITATIVE
    topic_color_map = {t: palette[i] for i, t in enumerate(sorted_topics)}

    plotted_count = 0

    for topic in sorted_topics:
        traces = topic_traces[topic]
        if not traces:
            continue

        interpolated_ys = []

        # Align all traces to common grid
        for x, y in traces:
            iy = np.interp(common_x, x, y)
            interpolated_ys.append(iy)

        # Calculate Statistics
        ys_matrix = np.vstack(interpolated_ys)
        mean_y = np.mean(ys_matrix, axis=0)

        color = topic_color_map.get(topic, "black")

        # Plot Mean Line
        plt.plot(common_x, mean_y, color=color, linewidth=2, label=topic)

        if include_ci:
            # Standard Error of the Mean (SEM) for the shaded region
            # 1.96 * SEM gives ~95% confidence interval
            sem_y = np.std(ys_matrix, axis=0) / np.sqrt(len(traces))
            ci = 1.96 * sem_y

            plt.fill_between(
                common_x, mean_y - ci, mean_y + ci, color=color, alpha=0.15
            )

        plotted_count += 1

    plt.xlabel("Document Position (0 - 100)")
    plt.ylabel("Avg. Usage")

    plt.xlim(0, 1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.grid(axis="x", linestyle=":", alpha=0.3)

    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png", bbox_inches="tight")
        plt.savefig(save_path + ".pdf", bbox_inches="tight")
