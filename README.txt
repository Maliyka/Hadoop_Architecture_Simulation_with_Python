═══════════════════════════════════════════════════════════════
  BIG DATA ANALYTICS — ASSIGNMENT 02
  Course: CC-342 | University of Management & Technology
  Instructor: Usama Amjad | Spring Semester 2026
═══════════════════════════════════════════════════════════════

PROJECT: Hadoop Architecture Simulation with Python

─── REQUIREMENTS ───────────────────────────────────────────────
  • Python 3.8 or higher
  • No external libraries required (pure Python standard library)

─── PROJECT STRUCTURE ──────────────────────────────────────────
  hadoop_project/
  ├── dataset.csv           ← Sales Analytics dataset (50 records)
  ├── namenode.json         ← NameNode metadata (auto-generated)
  ├── main.py               ← MAIN ENTRY POINT (run this)
  ├── hdfs_simulation.py    ← Part 1: HDFS block splitting & storage
  ├── replication.py        ← Part 2: Data replication
  ├── mapreduce.py          ← Part 3: MapReduce implementation
  ├── ml_model.py           ← Part 4: Linear Regression
  ├── report.docx           ← Part 5: Complete project report
  ├── DataNode1/            ← Simulated DataNode 1
  ├── DataNode2/            ← Simulated DataNode 2
  └── DataNode3/            ← Simulated DataNode 3

─── HOW TO RUN ─────────────────────────────────────────────────
  1. Open terminal / command prompt
  2. Navigate to the project folder:
       cd hadoop_project
  3. Run the main script:
       python main.py

  This will execute ALL 4 parts in sequence and display results.

  To run individual parts:
       python hdfs_simulation.py    ← Part 1 only
       python replication.py        ← Part 2 only
       python mapreduce.py          ← Part 3 only
       python ml_model.py           ← Part 4 only

─── EXPECTED OUTPUT ────────────────────────────────────────────
  • 5 block files stored across DataNode1, DataNode2, DataNode3
  • 5 replica files (one per block, in different DataNodes)
  • namenode.json with complete metadata
  • MapReduce results printed to console
  • Linear Regression model with R² ≈ 0.87

═══════════════════════════════════════════════════════════════
