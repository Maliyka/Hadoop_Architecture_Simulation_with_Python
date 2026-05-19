"""
Part 2: Data Replication
Course: Big Data Analytics (CC-342)
Assignment: 02
"""

import os
import json
from datetime import datetime

NAMENODE_FILE  = "namenode.json"
DATANODE_DIRS  = ["DataNode1", "DataNode2", "DataNode3"]


def load_namenode(filepath=NAMENODE_FILE):
    with open(filepath, "r") as f:
        return json.load(f)


def save_namenode(metadata, filepath=NAMENODE_FILE):
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)


def pick_replica_node(primary_node):
    """Choose a DataNode different from the primary."""
    others = [dn for dn in DATANODE_DIRS if dn != primary_node]
    # deterministic: pick the next one in list
    idx = DATANODE_DIRS.index(primary_node)
    return DATANODE_DIRS[(idx + 1) % len(DATANODE_DIRS)]


def create_replica(block_id, primary_path, replica_node):
    """
    Write a replica with added header metadata (not identical to primary).
    Replicas include: timestamp, replica tag, source node info.
    """
    # Read primary content (skip block metadata header lines)
    with open(primary_path, "r") as f:
        lines = f.readlines()

    replica_filename = f"{block_id}_replica.txt"
    replica_path     = os.path.join(replica_node, replica_filename)

    with open(replica_path, "w") as f:
        f.write(f"# ╔══════════════════════════════════════════════╗\n")
        f.write(f"# ║        REPLICA BLOCK — HDFS Simulation       ║\n")
        f.write(f"# ╚══════════════════════════════════════════════╝\n")
        f.write(f"# replica_of   : {block_id}\n")
        f.write(f"# replica_node : {replica_node}\n")
        f.write(f"# replicated_at: {datetime.now().isoformat()}\n")
        f.write(f"# source_path  : {primary_path}\n")
        f.write(f"# replication_factor: 2\n")
        f.write("#" + "─" * 50 + "\n")
        # Write original data lines (skip primary metadata header)
        data_started = False
        for line in lines:
            if line.startswith("# block_id") or line.startswith("# datanode") \
               or line.startswith("# rows") or line.startswith("# created_at") \
               or line.startswith("# HDFS") or line.startswith("#─") or line.startswith("#─"):
                continue
            f.writelines([line])

    return replica_path


def run_replication(metadata=None):
    print("\n" + "=" * 60)
    print("  PART 2: Data Replication")
    print("=" * 60)

    if metadata is None:
        metadata = load_namenode()

    replicated = 0
    print(f"\n  Replication Factor: 2  (1 primary + 1 replica per block)\n")

    for block_id, info in metadata["blocks"].items():
        primary_node  = info["datanode"]
        primary_path  = info["primary"]
        replica_node  = pick_replica_node(primary_node)
        replica_path  = create_replica(block_id, primary_path, replica_node)

        metadata["blocks"][block_id]["replica"]      = replica_path
        metadata["blocks"][block_id]["replica_node"] = replica_node
        replicated += 1

        print(f"  {block_id}  |  Primary: {primary_node}  →  Replica: {replica_node}")

    metadata["replication"] = 2
    save_namenode(metadata)

    print(f"\n✔  Replication complete: {replicated} replicas created")
    print(f"✔  NameNode metadata updated → {NAMENODE_FILE}")

    # Summary table
    print("\n" + "═" * 72)
    print("  Replication Summary")
    print("═" * 72)
    print(f"  {'Block':<14} {'Primary Node':<16} {'Replica Node':<16} {'Replica File'}")
    print("─" * 72)
    for bid, info in metadata["blocks"].items():
        rfile = os.path.basename(info["replica"]) if info["replica"] else "N/A"
        print(f"  {bid:<14} {info['datanode']:<16} {info['replica_node']:<16} {rfile}")
    print("═" * 72)

    return metadata


if __name__ == "__main__":
    run_replication()
