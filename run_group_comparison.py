#!/usr/bin/env python3
"""
Unpaired two-group comparison: normality (Shapiro-Wilk), variance (Levene),
and t-tests (Welch + optional Student). Reads a multi-row-header CSV and
writes a summary CSV of test statistics and p-values.

Note: With small n (e.g. n=4 and n=5), normality and variance tests are
underpowered; results are descriptive.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# Default dataset: cleaned from data/VSV Manual Axon Quantification - Sheet2.csv
HARDCODED_GROUPS = {
    "group1_label": "V2L to V1",
    "group1_values": [2.74e-5, 3.44e-5, 7.15e-5, 2.99e-5],
    "group2_label": "v1 to V2L areas (human tracing)",
    "group2_values": [3.24e-7, 4.38e-8, 5.48e-8, 9.56e-8, 4.19e-8],
}


def parse_two_group_csv(csv_path: Path):
    """
    Parse multi-row-header CSV into two groups. Expects:
    - Row 0: group labels (first cell is row type, e.g. 'group')
    - Row 1: identifiers
    - Row 2: 'normalized px counts' and numeric values
    Empty columns separate the two groups.
    Returns (group1_label, group1_values, group2_label, group2_values).
    """
    df = pd.read_csv(csv_path, header=None)
    # Row 0: group labels; row 2: normalized px counts
    labels_row = df.iloc[0]
    values_row = df.iloc[2]

    # Find split: first column index where we have a non-empty label after an empty run
    cols = list(range(1, len(labels_row)))
    group1_end = None
    for i in cols:
        val = values_row.iloc[i]
        label = labels_row.iloc[i]
        if pd.isna(val) or (isinstance(val, str) and str(val).strip() == ""):
            if group1_end is None:
                group1_end = i
            continue
        if group1_end is not None:
            break

    # Build group1: columns 1 through group1_end (exclusive of empty)
    group1_label = None
    for j in range(1, len(labels_row)):
        if pd.notna(labels_row.iloc[j]) and str(labels_row.iloc[j]).strip():
            group1_label = str(labels_row.iloc[j]).strip()
            break

    group1_vals = []
    for j in range(1, len(values_row)):
        v = values_row.iloc[j]
        if pd.isna(v) or (isinstance(v, str) and str(v).strip() == ""):
            break
        try:
            group1_vals.append(float(v))
        except (ValueError, TypeError):
            pass

    # Group2: after the empty column(s)
    group2_label = None
    group2_vals = []
    found_empty = False
    for j in range(1, len(labels_row)):
        v = values_row.iloc[j]
        if pd.isna(v) or (isinstance(v, str) and str(v).strip() == ""):
            found_empty = True
            continue
        if not found_empty:
            continue
        if group2_label is None and pd.notna(labels_row.iloc[j]) and str(labels_row.iloc[j]).strip():
            group2_label = str(labels_row.iloc[j]).strip()
        try:
            group2_vals.append(float(v))
        except (ValueError, TypeError):
            pass

    if group1_label is None:
        group1_label = "Group1"
    if group2_label is None:
        group2_label = "Group2"

    return group1_label, pd.Series(group1_vals), group2_label, pd.Series(group2_vals)


def run_tests(group1: pd.Series, group2: pd.Series, label1: str, label2: str):
    """Run Shapiro-Wilk (per group), Levene, Welch and Student t-tests. Return list of result dicts."""
    results = []

    # Shapiro-Wilk per group
    for label, ser in [(label1, group1), (label2, group2)]:
        ser_clean = ser.dropna()
        if len(ser_clean) < 3:
            results.append({
                "Test": "Shapiro-Wilk",
                "Group": label,
                "Statistic": None,
                "P_value": None,
                "Note": "n too small",
            })
            continue
        stat, p = stats.shapiro(ser_clean)
        results.append({
            "Test": "Shapiro-Wilk",
            "Group": label,
            "Statistic": round(stat, 6),
            "P_value": round(p, 6),
            "Note": "",
        })

    # Levene
    g1 = group1.dropna().values
    g2 = group2.dropna().values
    stat, p = stats.levene(g1, g2)
    results.append({
        "Test": "Levene",
        "Group": "",
        "Statistic": round(stat, 6),
        "P_value": round(p, 6),
        "Note": "variance equality",
    })

    # Welch t-test
    stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    results.append({
        "Test": "Welch t-test",
        "Group": "",
        "Statistic": round(stat, 6),
        "P_value": round(p, 6),
        "Note": "",
    })

    # Student t-test
    stat, p = stats.ttest_ind(g1, g2, equal_var=True)
    results.append({
        "Test": "Student t-test",
        "Group": "",
        "Statistic": round(stat, 6),
        "P_value": round(p, 6),
        "Note": "assumes equal variances",
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Unpaired two-group stats: normality, variance, t-tests.")
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to input CSV; if omitted, uses built-in VSV axon quantification data.",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/statistics_summary.csv",
        help="Path to output summary CSV (default: data/statistics_summary.csv)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)

    if args.csv is None:
        label1 = HARDCODED_GROUPS["group1_label"]
        group1 = pd.Series(HARDCODED_GROUPS["group1_values"])
        label2 = HARDCODED_GROUPS["group2_label"]
        group2 = pd.Series(HARDCODED_GROUPS["group2_values"])
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: input file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        label1, group1, label2, group2 = parse_two_group_csv(csv_path)
        if len(group1) == 0 or len(group2) == 0:
            print("Error: could not parse two groups from CSV.", file=sys.stderr)
            sys.exit(1)

    print(f"Group 1 ({label1}): n={len(group1)}")
    print(f"Group 2 ({label2}): n={len(group2)}")

    results = run_tests(group1, group2, label1, label2)
    summary = pd.DataFrame(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Summary written to {out_path}")

    welch_row = summary[summary["Test"] == "Welch t-test"].iloc[0]
    print(f"Welch t-test p-value: {welch_row['P_value']}")


if __name__ == "__main__":
    main()
