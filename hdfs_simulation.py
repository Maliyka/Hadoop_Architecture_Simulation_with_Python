"""
Part 1: HDFS Simulation Using Python
Course: Big Data Analytics (CC-342)
Assignment: 02
"""

import os
import json
import math
import shutil
from datetime import datetime

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BLOCK_SIZE = 10          # lines per block
DATASET_FILE = "dataset.csv"
DATANODE_DIRS = ["DataNode1", "DataNode2", "DataNode3"]
NAMENODE_FILE = "namenode.json"


def initialize_datanodes():
    """Create DataNode directories (clear existing blocks first)."""
    for dn in DATANODE_DIRS:
        os.makedirs(dn, exist_ok=True)
        # Remove old blocks
        for f in os.listdir(dn):
            if f.startswith("block_"):
                os.remove(os.path.join(dn, f))
    print("✔  DataNode directories initialized:", DATANODE_DIRS)


def read_dataset(filepath):
    """Read dataset and return lines."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = lines[0]
    data   = lines[1:]
    print(f"✔  Dataset loaded: {len(data)} records  |  Header: {header.strip()}")
    return header, data


def split_into_blocks(data, block_size=BLOCK_SIZE):
    """Split data rows into fixed-size blocks."""
    blocks = []
    for i in range(0, len(data), block_size):
        blocks.append(data[i:i + block_size])
    print(f"✔  Split into {len(blocks)} blocks  (block_size={block_size} lines each)")
    return blocks


def store_blocks(blocks, header):
    """
    Distribute blocks across DataNodes in round-robin fashion.
    Returns metadata dict for NameNode.
    """
    metadata = {
        "hdfs_version"  : "3.3.x (Simulated)",
        "created_at"    : datetime.now().isoformat(),
        "dataset"       : DATASET_FILE,
        "total_records" : sum(len(b) for b in blocks),
        "block_size"    : BLOCK_SIZE,
        "total_blocks"  : len(blocks),
        "replication"   : 1,       # updated in Part 2
        "blocks"        : {}
    }

    for idx, block in enumerate(blocks):
        block_id  = f"block_{idx:03d}"
        datanode  = DATANODE_DIRS[idx % len(DATANODE_DIRS)]
        filepath  = os.path.join(datanode, f"{block_id}.txt")

        # Write block with header so each file is self-contained
        with open(filepath, "w") as f:
            f.write(f"# HDFS Block Metadata\n")
            f.write(f"# block_id   : {block_id}\n")
            f.write(f"# datanode   : {datanode}\n")
            f.write(f"# rows       : {len(block)}\n")
            f.write(f"# created_at : {datetime.now().isoformat()}\n")
            f.write("#" + "─" * 50 + "\n")
            f.write(header)          # CSV header
            f.writelines(block)      # CSV rows

        metadata["blocks"][block_id] = {
            "primary"       : filepath,
            "datanode"      : datanode,
            "rows"          : len(block),
            "replica"       : None,   # filled in Part 2
            "replica_node"  : None
        }

        print(f"   Stored {block_id}  →  {filepath}  ({len(block)} rows)")

    return metadata


def save_namenode(metadata, filepath=NAMENODE_FILE):
    """Persist NameNode metadata to JSON."""
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✔  NameNode metadata saved → {filepath}")


def display_summary(metadata):
    """Pretty-print block storage summary."""
    print("\n" + "═" * 60)
    print("  HDFS Block Storage Summary")
    print("═" * 60)
    print(f"  Dataset          : {metadata['dataset']}")
    print(f"  Total Records    : {metadata['total_records']}")
    print(f"  Block Size       : {metadata['block_size']} rows")
    print(f"  Total Blocks     : {metadata['total_blocks']}")
    print(f"  DataNodes        : {', '.join(DATANODE_DIRS)}")
    print("─" * 60)
    print(f"  {'Block ID':<14} {'DataNode':<14} {'Rows'}")
    print("─" * 60)
    for bid, info in metadata["blocks"].items():
        print(f"  {bid:<14} {info['datanode']:<14} {info['rows']}")
    print("═" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def run_hdfs_simulation():
    print("\n" + "=" * 60)
    print("  PART 1: HDFS Simulation")
    print("=" * 60)

    initialize_datanodes()
    header, data = read_dataset(DATASET_FILE)
    blocks        = split_into_blocks(data)
    metadata      = store_blocks(blocks, header)
    save_namenode(metadata)
    display_summary(metadata)

    return metadata   # pass to Part 2


if __name__ == "__main__":
    run_hdfs_simulation()
