from typing import List, Dict, Any

import pandas as pd
from colorama import Style, Fore
from tabulate import tabulate


def pretty_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics_config: List[Dict[str, Any]],
    title: str,
):
    agg_rules = {m["col"]: m.get("agg", "mean") for m in metrics_config}

    # Perform Grouping
    grouped_stats = df.groupby(group_cols).agg(agg_rules).reset_index()

    # Calculate Ranges for Color Logic (Min/Max)
    ranges = {}
    for m in metrics_config:
        if "high_is_good" in m:
            col_name = m["col"]
            series = grouped_stats[col_name]
            ranges[col_name] = {"min": series.min(), "max": series.max()}

    # Build Table
    headers = group_cols + [m.get("header", m["col"]) for m in metrics_config]
    table_rows = []

    for _, row in grouped_stats.iterrows():
        table_row = [str(row[c]) for c in group_cols]
        for m in metrics_config:
            table_row.append(get_fmt_val(row, m, ranges, m.get("decimals", 2)))
        table_rows.append(table_row)

    print(f"\n{Style.BRIGHT}=== {title} ==={Style.RESET_ALL}")
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))


def get_fmt_val(row, config, ranges, decimals=2):
    col_key = config["col"]
    val = row[col_key]

    # Apply formatting options
    multiplier = config.get("mult", 1)
    unit = config.get("unit", "")
    decimals = config.get("decimals", decimals)

    display_val = val * multiplier
    fmt_str = f"{display_val:.{decimals}f}{unit}" if decimals or unit else display_val

    # Apply Color Logic if defined
    if "high_is_good" in config:
        val_min = ranges[col_key]["min"]
        val_max = ranges[col_key]["max"]
        high_is_good = config["high_is_good"]

        if val_min != val_max:
            color = ""
            if high_is_good:
                if val == val_max:
                    color = Fore.GREEN
                elif val == val_min:
                    color = Fore.RED
            else:  # Low is good (e.g. Duration)
                if val == val_min:
                    color = Fore.GREEN
                elif val == val_max:
                    color = Fore.RED

            return f"{color}{fmt_str}{Style.RESET_ALL}"

    return fmt_str
