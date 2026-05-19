"""
main.py — Big Data Analytics Assignment 02
Course: CC-342  |  University of Management & Technology
Instructor: Usama Amjad  |  Spring Semester 2026

Runs all 4 implementation parts in sequence:
  Part 1 → HDFS Simulation
  Part 2 → Data Replication
  Part 3 → MapReduce
  Part 4 → Linear Regression (ML)
"""

import sys
import os
import time

# ─── Ensure we run from the project directory ─────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from hdfs_simulation import run_hdfs_simulation
from replication     import run_replication
from mapreduce       import run_mapreduce
from ml_model        import run_ml_model


def banner(text):
    width = 62
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + text.center(width) + "║")
    print("╚" + "═" * width + "╝")


def main():
    banner("BIG DATA ANALYTICS — ASSIGNMENT 02")
    print("""
  University of Management & Technology
  Department of Artificial Intelligence
  Course  : Big Data Analytics (CC-342)
  Semester: Spring 2026
  Instructor: Usama Amjad
    """)

    start_time = time.time()

    # ── Part 1: HDFS ────────────────────────────────────────────
    banner("PART 1: HDFS Simulation")
    metadata = run_hdfs_simulation()

    # ── Part 2: Replication ─────────────────────────────────────
    banner("PART 2: Data Replication")
    metadata = run_replication(metadata)

    # ── Part 3: MapReduce ───────────────────────────────────────
    banner("PART 3: MapReduce Processing")
    mr_results = run_mapreduce()

    # ── Part 4: Machine Learning ─────────────────────────────────
    banner("PART 4: Linear Regression (ML)")
    ml_results = run_ml_model(mr_results)

    # ── Final Summary ────────────────────────────────────────────
    elapsed = time.time() - start_time
    banner("EXECUTION COMPLETE")
    print(f"""
  ✔  All 4 parts executed successfully
  ✔  NameNode metadata    → namenode.json
  ✔  HDFS blocks stored   → DataNode1/, DataNode2/, DataNode3/
  ✔  Replicas created     → 1 replica per block
  ✔  MapReduce results    → displayed above
  ✔  ML Model (R²)        → {ml_results['metrics']['test_r2']:.4f}
  ⏱  Total runtime        → {elapsed:.2f} seconds
    """)


if __name__ == "__main__":
    main()
