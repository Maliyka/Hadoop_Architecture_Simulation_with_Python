"""
Part 3: MapReduce Implementation
Course: Big Data Analytics (CC-342)
Assignment: 02
"""

import os
import csv
import json
from collections import defaultdict
from functools import reduce as functools_reduce

DATASET_FILE  = "dataset.csv"
NAMENODE_FILE = "namenode.json"


# ─────────────────────────────────────────────
# Map Phase
# ─────────────────────────────────────────────

def map_word_count(lines):
    """Emit (word, 1) for every word in the text rows."""
    mapped = []
    for line in lines:
        words = line.strip().replace(",", " ").split()
        for word in words:
            word = word.strip().lower()
            if word and not word.replace(".", "").replace("-", "").isdigit():
                mapped.append((word, 1))
    return mapped


def map_category_count(rows):
    """Emit (category, 1) for each CSV row."""
    mapped = []
    for row in rows:
        if row.get("Category"):
            mapped.append((row["Category"].strip(), 1))
    return mapped


def map_sales_by_region(rows):
    """Emit (region, sales_value) for aggregation."""
    mapped = []
    for row in rows:
        try:
            mapped.append((row["Region"].strip(), float(row["Sales"])))
        except (KeyError, ValueError):
            pass
    return mapped


def map_price_by_category(rows):
    """Emit (category, price) for average price computation."""
    mapped = []
    for row in rows:
        try:
            mapped.append((row["Category"].strip(), float(row["Price"])))
        except (KeyError, ValueError):
            pass
    return mapped


# ─────────────────────────────────────────────
# Shuffle Phase
# ─────────────────────────────────────────────

def shuffle(mapped_pairs):
    """Group values by key: {key: [v1, v2, ...]}"""
    grouped = defaultdict(list)
    for key, value in mapped_pairs:
        grouped[key].append(value)
    return dict(grouped)


# ─────────────────────────────────────────────
# Reduce Phase
# ─────────────────────────────────────────────

def reduce_sum(grouped):
    """Reduce by summing values."""
    return {key: sum(values) for key, values in grouped.items()}


def reduce_count(grouped):
    """Reduce by counting occurrences."""
    return {key: sum(values) for key, values in grouped.items()}


def reduce_average(grouped):
    """Reduce by computing average."""
    return {key: round(sum(values) / len(values), 2) for key, values in grouped.items()}


# ─────────────────────────────────────────────
# Read all blocks from DataNodes
# ─────────────────────────────────────────────

def read_all_blocks():
    """Collect all data rows from every block stored in DataNodes."""
    all_lines = []
    all_rows  = []

    with open(NAMENODE_FILE) as f:
        metadata = json.load(f)

    for block_id, info in metadata["blocks"].items():
        path = info["primary"]
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            file_lines = f.readlines()

        # Skip block-metadata comment lines
        data_lines = [l for l in file_lines if not l.startswith("#")]
        if not data_lines:
            continue

        # First non-comment line is CSV header
        reader = csv.DictReader(data_lines)
        for row in reader:
            all_rows.append(row)
            all_lines.append(",".join(row.values()))

    return all_lines, all_rows


def print_results(title, results, unit="", top_n=None):
    items = sorted(results.items(), key=lambda x: -x[1])
    if top_n:
        items = items[:top_n]
    print(f"\n  ┌── {title}")
    for key, val in items:
        bar = "█" * min(int(val / max(results.values()) * 30), 30)
        print(f"  │  {key:<20} {str(round(val,2)):>10} {unit}  {bar}")
    print(f"  └{'─' * 48}")


def run_mapreduce():
    print("\n" + "=" * 60)
    print("  PART 3: MapReduce Implementation")
    print("=" * 60)

    all_lines, all_rows = read_all_blocks()
    print(f"\n  Total rows loaded from DataNodes: {len(all_rows)}")

    # ── Task 1: Word Count ──────────────────────────────────────
    print("\n  [MAP]    Word Count → mapping words...")
    wc_mapped   = map_word_count(all_lines)
    print(f"  [SHUFFLE] Grouping {len(wc_mapped)} word-value pairs...")
    wc_shuffled = shuffle(wc_mapped)
    print(f"  [REDUCE]  Reducing {len(wc_shuffled)} unique words...")
    wc_result   = reduce_count(wc_shuffled)
    print_results("TOP-15 WORD FREQUENCY", wc_result, top_n=15)

    # ── Task 2: Category Count ──────────────────────────────────
    print("\n  [MAP]    Category Count → mapping categories...")
    cc_mapped   = map_category_count(all_rows)
    cc_shuffled = shuffle(cc_mapped)
    cc_result   = reduce_count(cc_shuffled)
    print_results("CATEGORY FREQUENCY", cc_result)

    # ── Task 3: Total Sales by Region ───────────────────────────
    print("\n  [MAP]    Sales by Region → mapping sales...")
    sr_mapped   = map_sales_by_region(all_rows)
    sr_shuffled = shuffle(sr_mapped)
    sr_result   = reduce_sum(sr_shuffled)
    print_results("TOTAL SALES BY REGION", sr_result, unit="PKR")

    # ── Task 4: Average Price by Category ───────────────────────
    print("\n  [MAP]    Average Price by Category → mapping prices...")
    ap_mapped   = map_price_by_category(all_rows)
    ap_shuffled = shuffle(ap_mapped)
    ap_result   = reduce_average(ap_shuffled)
    print_results("AVG PRICE BY CATEGORY", ap_result, unit="PKR")

    print("\n✔  MapReduce completed successfully")

    return {
        "word_count"       : wc_result,
        "category_count"   : cc_result,
        "sales_by_region"  : sr_result,
        "avg_price_cat"    : ap_result,
        "all_rows"         : all_rows
    }


if __name__ == "__main__":
    run_mapreduce()
