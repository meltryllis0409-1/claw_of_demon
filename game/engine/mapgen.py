from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple
from random import random, choice


class NodeType(Enum):
    COMBAT = auto()
    ELITE = auto()
    CAMP = auto()
    SHOP = auto()
    BOSS = auto()


@dataclass
class MapNode:
    node_id: int
    depth: int
    idx: int
    x: float
    y: float
    node_type: NodeType
    next_ids: List[int]
    cleared: bool = False


@dataclass
class MapGraph:
    nodes: Dict[int, MapNode]
    start_id: int
    boss_id: int


def _assign_types(counts: List[int]) -> Dict[Tuple[int, int], NodeType]:
    # Fixed, readable distribution for now.
    # counts expected [1,2,3,3,2,1]
    mapping: Dict[Tuple[int, int], NodeType] = {}
    mapping[(0,0)] = NodeType.COMBAT
    mapping[(1,0)] = NodeType.CAMP
    mapping[(1,1)] = NodeType.COMBAT
    mapping[(2,0)] = NodeType.COMBAT
    mapping[(2,1)] = NodeType.SHOP
    mapping[(2,2)] = NodeType.COMBAT
    mapping[(3,0)] = NodeType.COMBAT
    mapping[(3,1)] = NodeType.ELITE
    mapping[(3,2)] = NodeType.CAMP
    mapping[(3,3)] = NodeType.ELITE
    mapping[(3,4)] = NodeType.CAMP
    mapping[(4,1)] = NodeType.ELITE
    mapping[(4,2)] = NodeType.CAMP
    mapping[(4,3)] = NodeType.SHOP
    mapping[(4,4)] = NodeType.ELITE
    mapping[(5,1)] = NodeType.ELITE
    mapping[(5,2)] = NodeType.CAMP
    mapping[(5,3)] = NodeType.ELITE
    mapping[(6,1)] = NodeType.CAMP
    mapping[(6,2)] = NodeType.ELITE
    mapping[(7,0)] = NodeType.BOSS
    return mapping


def generate_map(counts: List[int]) -> MapGraph:
    type_map = _assign_types(counts)

    layers: List[List[int]] = []
    nodes: Dict[int, MapNode] = {}
    nid = 0

    depth_n = len(counts)
    for d in range(depth_n):
        c = counts[d]
        ids = []
        for i in range(c):
            x = 0.5 if c == 1 else (i / (c - 1))
            y = d / (depth_n - 1) if depth_n > 1 else 0.0
            t = type_map.get((d, i), NodeType.COMBAT if d < depth_n - 1 else NodeType.BOSS)
            node = MapNode(node_id=nid, depth=d, idx=i, x=x, y=y, node_type=t, next_ids=[])
            nodes[nid] = node
            ids.append(nid)
            nid += 1
        layers.append(ids)

    # Connect layers
    for d in range(depth_n - 1):
        cur = layers[d]
        nxt = layers[d+1]
        ncur, nnxt = len(cur), len(nxt)

        # primary connection (index-proportional)
        for i, cid in enumerate(cur):
            if ncur == 1:
                base_j = nnxt // 2
            else:
                base_j = round(i * (nnxt - 1) / (ncur - 1))
            base_j = max(0, min(nnxt - 1, base_j))
            nodes[cid].next_ids.append(nxt[base_j])

            # optional second connection (branching)
            if nnxt > 1 and random() < 0.6:
                j2 = base_j + (1 if random() < 0.5 else -1)
                j2 = max(0, min(nnxt - 1, j2))
                if nxt[j2] not in nodes[cid].next_ids:
                    nodes[cid].next_ids.append(nxt[j2])

        # ensure each next node has at least one incoming
        incoming = {nid2: 0 for nid2 in nxt}
        for cid in cur:
            for tid in nodes[cid].next_ids:
                incoming[tid] += 1

        for tid, cnt in incoming.items():
            if cnt == 0:
                # connect from closest current index
                # choose a random current node for simplicity
                nodes[choice(cur)].next_ids.append(tid)

    start_id = layers[0][0]
    boss_id = layers[-1][0]
    return MapGraph(nodes=nodes, start_id=start_id, boss_id=boss_id)
