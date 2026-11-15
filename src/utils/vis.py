import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as cm
import locale
import functools

USE_LATEX = True

# Binary Contrast
PALETTE_CONTRAST = ["#1D4ED8", "#93C5FD"]

# Heatmap (Continuous Blue)
PALETTE_HEATMAP = ["#F1F5F9", "#BFDBFE", "#3B82F6", "#1D4ED8", "#172554"]

# Qualitative
PALETTE_QUALITATIVE = [
    "#1D4ED8",  # Blue
    "#D97706",  # Amber
    "#047857",  # Emerald Green
    "#7C3AED",  # Violet
    "#0891B2",  # Teal
    "#BE123C",  # Crimson
    "#65A30D",  # Vibrant Lime
    "#DB2777",  # Pink
    "#475569",  # Slate Gray
]

# Models
PALETTE_MODELS = [
    '#10316B',  # Deep Navy
    '#8F241E',  # Oxford Maroon
    '#D4AF37',  # Metallic/Muted Gold
    '#2D5A27',  # Hunter Green
    '#467599',  # Steel Blue
    '#5A315D'   # Imperial Plum
]

PALETTE_NODES = {"active": "#1D4ED8", "linked": "#BFDBFE", "inactive": "#e6ebf2"}

# Summarizers
PALETTE_SUMMARIZERS = [
    "#93C5FD",  # Powder Royal
    "#3B82F6",  # Standard Blue
    "#1D4ED8",  # True Royal
]


def bold(val):
    return r"\textbf{\shortstack{" + str(val) + "}}" if USE_LATEX else str(val)


def initialize():
    try:
        cm.register(
            name="paper_continuous",
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "paper_blue", PALETTE_HEATMAP, N=256
            ),
        )
    except (ValueError, AttributeError):
        pass

    try:
        cm.register(name="paper_models", cmap=mcolors.ListedColormap(PALETTE_MODELS))
    except (ValueError, AttributeError):
        pass

    try:
        cm.register(
            name="paper_qualitative", cmap=mcolors.ListedColormap(PALETTE_QUALITATIVE)
        )
    except (ValueError, AttributeError):
        pass

    sns.set_theme(style="whitegrid", context="paper", palette="paper_qualitative")
    plt.style.use("src/styles.mplstyle")

    _original_heatmap = sns.heatmap

    @functools.wraps(_original_heatmap)
    def thesis_heatmap(data, **kwargs):
        annot = kwargs.get("annot")
        annot_kws = kwargs.get("annot_kws", {})

        want_bold = (
            (
                (annot_kws.pop("weight", None) == "bold")
                or (annot_kws.pop("fontweight", None) == "bold")
            )
            if USE_LATEX
            else False
        )

        if hasattr(annot, "ndim"):
            if want_bold:
                if hasattr(annot, "applymap"):  # It's a DataFrame
                    kwargs["annot"] = annot.map(bold)
                else:  # It's a Numpy Array
                    kwargs["annot"] = np.vectorize(bold)(np.array(annot))
            kwargs["fmt"] = ""

        elif annot:
            user_fmt = kwargs.get("fmt", ".2f")

            def format_value(x):
                s = format(x, user_fmt)
                return bold(s) if want_bold else s

            if hasattr(data, "applymap"):
                kwargs["annot"] = data.map(format_value)
            else:
                kwargs["annot"] = np.vectorize(format_value)(np.array(data))
            kwargs["fmt"] = ""

        kwargs["annot_kws"] = annot_kws
        ax = _original_heatmap(data, **kwargs)
        ax.grid(False)  # noqa
        return ax

    sns.heatmap = thesis_heatmap
    plt.rcParams["text.usetex"] = USE_LATEX
    plt.set_loglevel("ERROR")


def apply_name_mappings(df: pd.DataFrame, rename_columns: bool = True):
    mappings = {
        "summarizer_name": {
            "name": "Strategy",
            "values": {
                "hierarchical-merge": "Hierarchical Merge",
                "incremental-update": "Incremental Update",
                "single-pass": "Single Pass",
            },
        },
        "llm_name": {
            "name": "LLM",
            "values": {
                "gpt-4.1": "GPT-4.1",
                "gpt-4.1-mini": "GPT-4.1 Mini",
                "qwen3-4b-instruct-2507-q8_0": "Qwen3 4B",
                "gemma-4-E4B-it-UD-Q8_K_XL": "Gemma 4 E4B",
                "ministral-3-3B-instruct-UD-Q8_K_XL": "Ministral 3 3B",
                "qwen3_5-4B-UD-Q8_K_XL": "Qwen3.5 4B",
            },
        },
    }

    for k, values in mappings.items():
        if k in df.columns:
            df[k] = df[k].map(values["values"])

    if rename_columns:
        df.rename(
            columns={k: v["name"] for k, v in mappings.items() if k in df.columns},
            inplace=True,
        )

    return df


def wrap_text(text, width=15, max_rows: Optional[int] = None):
    lines = textwrap.wrap(text, width, break_long_words=True)
    safe_lines = [line.replace("_", r"\_") if USE_LATEX else line for line in lines]

    if max_rows is not None and len(safe_lines) > max_rows:
        safe_lines = safe_lines[: max_rows - 1]
        safe_lines.append("...")

    new_line_string = r"\\" if USE_LATEX else "\n"
    return bold(new_line_string.join(safe_lines))
