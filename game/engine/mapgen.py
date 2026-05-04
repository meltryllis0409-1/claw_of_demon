from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from random import Random
from typing import Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    START = auto()
    COMBAT = auto()
    UNKNOWN = auto()
    ELITE = auto()
    CAMP = auto()
    SHOP = auto()
    TREASURE = auto()
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


Position = Tuple[int, int]
Edge = Tuple[Position, Position]

GRID_COLS = 7
GRID_FLOORS = 15
PATH_COUNT = 6
MAX_MAP_ATTEMPTS = 500
MAX_PATH_ATTEMPTS = 80

LOCATION_WEIGHTS: List[Tuple[NodeType, int]] = [
    (NodeType.COMBAT, 45),
    (NodeType.UNKNOWN, 22),
    (NodeType.ELITE, 16),
    (NodeType.CAMP, 12),
    (NodeType.SHOP, 5),
]

SPECIAL_CHAIN_TYPES: Set[NodeType] = {
    NodeType.ELITE,
    NodeType.CAMP,
    NodeType.SHOP,
}


def _fixed_type_for_floor(floor: int) -> Optional[NodeType]:
    if floor == 0:
        return NodeType.COMBAT
    if floor == 8:
        return NodeType.TREASURE
    if floor == 14:
        return NodeType.CAMP
    return None


def _weighted_choice(rng: Random, weighted: List[Tuple[NodeType, int]]) -> NodeType:
    total = sum(weight for _, weight in weighted)
    roll = rng.uniform(0, total)
    acc = 0.0
    for node_type, weight in weighted:
        acc += weight
        if roll <= acc:
            return node_type
    return weighted[-1][0]


def _weighted_order(rng: Random, weighted: List[Tuple[NodeType, int]]) -> List[NodeType]:
    pool = list(weighted)
    ordered: List[NodeType] = []
    while pool:
        picked = _weighted_choice(rng, pool)
        ordered.append(picked)
        pool = [(node_type, weight) for node_type, weight in pool if node_type != picked]
    return ordered


def _edge_crosses_existing(
    floor: int,
    start_col: int,
    next_col: int,
    used_edges: Dict[int, List[Edge]],
) -> bool:
    for src, dst in used_edges.get(floor, []):
        other_start = src[1]
        other_next = dst[1]
        if (start_col < other_start and next_col > other_next) or (
            start_col > other_start and next_col < other_next
        ):
            return True
    return False


def _edge_breaks_fixed_destination_rule(
    floor: int,
    start_col: int,
    next_col: int,
    used_edges: Dict[int, List[Edge]],
) -> bool:
    if _fixed_type_for_floor(floor + 1) is None:
        return False

    src_pos = (floor, start_col)
    dst_pos = (floor + 1, next_col)

    for src, dst in used_edges.get(floor, []):
        if src == src_pos and dst != dst_pos:
            return True

    return False


def _generate_paths(rng: Random) -> List[List[Position]]:
    paths: List[List[Position]] = []
    used_edges: Dict[int, List[Edge]] = {floor: [] for floor in range(GRID_FLOORS - 1)}

    for path_index in range(PATH_COUNT):
        built = False

        for _ in range(MAX_PATH_ATTEMPTS):
            start_cols = list(range(GRID_COLS))

            if path_index == 1 and paths:
                first_start_col = paths[0][0][1]
                start_cols = [col for col in start_cols if col != first_start_col]

            col = rng.choice(start_cols)
            path: List[Position] = [(0, col)]
            new_edges: List[Tuple[int, Edge]] = []
            ok = True

            for floor in range(GRID_FLOORS - 1):
                candidates = [
                    next_col
                    for next_col in (col - 1, col, col + 1)
                    if 0 <= next_col < GRID_COLS
                ]
                rng.shuffle(candidates)

                chosen: Optional[int] = None

                for next_col in candidates:
                    if _edge_crosses_existing(floor, col, next_col, used_edges):
                        continue
                    if _edge_breaks_fixed_destination_rule(floor, col, next_col, used_edges):
                        continue
                    chosen = next_col
                    break

                if chosen is None:
                    ok = False
                    break

                src = (floor, col)
                dst = (floor + 1, chosen)
                new_edges.append((floor, (src, dst)))
                path.append(dst)
                col = chosen

            if ok:
                for floor, edge in new_edges:
                    used_edges[floor].append(edge)
                paths.append(path)
                built = True
                break

        if not built:
            raise RuntimeError("failed to generate a non-crossing map path")

    return paths


def _build_position_graph(
    paths: List[List[Position]],
) -> Tuple[Set[Position], Dict[Position, Set[Position]], Dict[Position, Set[Position]]]:
    positions: Set[Position] = set()
    outgoing: Dict[Position, Set[Position]] = {}
    incoming: Dict[Position, Set[Position]] = {}

    for path in paths:
        for pos in path:
            positions.add(pos)
            outgoing.setdefault(pos, set())
            incoming.setdefault(pos, set())

        for src, dst in zip(path, path[1:]):
            outgoing.setdefault(src, set()).add(dst)
            incoming.setdefault(dst, set()).add(src)
            outgoing.setdefault(dst, set())
            incoming.setdefault(src, set())

    return positions, outgoing, incoming


def _has_impossible_fixed_destination_conflict(
    outgoing: Dict[Position, Set[Position]],
) -> bool:
    for destinations in outgoing.values():
        if len(destinations) < 2:
            continue

        seen: Set[NodeType] = set()

        for dst in destinations:
            fixed = _fixed_type_for_floor(dst[0])
            if fixed is None:
                continue
            if fixed in seen:
                return True
            seen.add(fixed)

    return False


def _base_floor_allowed(pos: Position, node_type: NodeType) -> bool:
    floor, _ = pos
    fixed = _fixed_type_for_floor(floor)

    if fixed is not None:
        return node_type == fixed

    if node_type in (NodeType.START, NodeType.TREASURE, NodeType.BOSS):
        return False

    if floor < 5 and node_type in (NodeType.ELITE, NodeType.CAMP):
        return False

    if floor == 13 and node_type == NodeType.CAMP:
        return False

    return True


def _local_type_allowed(
    pos: Position,
    node_type: NodeType,
    assigned: Dict[Position, NodeType],
    outgoing: Dict[Position, Set[Position]],
    incoming: Dict[Position, Set[Position]],
) -> bool:
    if not _base_floor_allowed(pos, node_type):
        return False

    if node_type in SPECIAL_CHAIN_TYPES:
        for parent in incoming.get(pos, set()):
            parent_type = assigned.get(parent)
            if parent_type in SPECIAL_CHAIN_TYPES:
                return False

        for child in outgoing.get(pos, set()):
            child_type = assigned.get(child)
            if child_type in SPECIAL_CHAIN_TYPES:
                return False

    for parent in incoming.get(pos, set()):
        siblings = outgoing.get(parent, set())

        if len(siblings) < 2:
            continue

        for sibling in siblings:
            if sibling != pos and assigned.get(sibling) == node_type:
                return False

    if len(outgoing.get(pos, set())) >= 2:
        seen_child_types: Set[NodeType] = set()

        for child in outgoing[pos]:
            child_type = assigned.get(child)

            if child_type is None:
                continue

            if child_type in seen_child_types:
                return False

            seen_child_types.add(child_type)

    return True


def _validate_assignment(
    positions: Set[Position],
    assigned: Dict[Position, NodeType],
    outgoing: Dict[Position, Set[Position]],
    incoming: Dict[Position, Set[Position]],
) -> bool:
    if set(assigned.keys()) != positions:
        return False

    for pos in positions:
        node_type = assigned[pos]

        if not _base_floor_allowed(pos, node_type):
            return False

    for src, destinations in outgoing.items():
        if len(destinations) >= 2:
            child_types = [assigned[dst] for dst in destinations]

            if len(child_types) != len(set(child_types)):
                return False

        src_type = assigned[src]

        for dst in destinations:
            dst_type = assigned[dst]

            if src_type in SPECIAL_CHAIN_TYPES and dst_type in SPECIAL_CHAIN_TYPES:
                return False

    for pos in positions:
        if not _local_type_allowed(pos, assigned[pos], assigned, outgoing, incoming):
            return False

    return True


def _assign_floor_recursive(
    rng: Random,
    floor_positions: List[Position],
    index: int,
    assigned: Dict[Position, NodeType],
    outgoing: Dict[Position, Set[Position]],
    incoming: Dict[Position, Set[Position]],
) -> bool:
    if index >= len(floor_positions):
        return True

    pos = floor_positions[index]

    valid_weighted = [
        (node_type, weight)
        for node_type, weight in LOCATION_WEIGHTS
        if _local_type_allowed(pos, node_type, assigned, outgoing, incoming)
    ]

    for candidate in _weighted_order(rng, valid_weighted):
        assigned[pos] = candidate

        if _assign_floor_recursive(
            rng,
            floor_positions,
            index + 1,
            assigned,
            outgoing,
            incoming,
        ):
            return True

        del assigned[pos]

    return False


def _assign_locations(
    rng: Random,
    positions: Set[Position],
    outgoing: Dict[Position, Set[Position]],
    incoming: Dict[Position, Set[Position]],
) -> Optional[Dict[Position, NodeType]]:
    assigned: Dict[Position, NodeType] = {}

    for pos in sorted(positions, key=lambda p: (p[0], p[1])):
        fixed = _fixed_type_for_floor(pos[0])

        if fixed is not None:
            assigned[pos] = fixed

    for floor in range(GRID_FLOORS):
        fixed_positions = sorted(
            (p for p in positions if p[0] == floor and p in assigned),
            key=lambda p: p[1],
        )

        for pos in fixed_positions:
            if not _local_type_allowed(pos, assigned[pos], assigned, outgoing, incoming):
                return None

        floor_positions = sorted(
            (p for p in positions if p[0] == floor and p not in assigned),
            key=lambda p: p[1],
        )

        if not floor_positions:
            continue

        if not _assign_floor_recursive(
            rng,
            floor_positions,
            0,
            assigned,
            outgoing,
            incoming,
        ):
            return None

    if _validate_assignment(positions, assigned, outgoing, incoming):
        return assigned

    return None


def _node_x(col: int) -> float:
    if GRID_COLS <= 1:
        return 0.5

    return col / (GRID_COLS - 1)


def _node_y(floor: int) -> float:
    if GRID_FLOORS <= 1:
        return 0.5

    return 1.0 - ((floor + 1) / (GRID_FLOORS + 1))


def _build_map_graph(
    positions: Set[Position],
    outgoing: Dict[Position, Set[Position]],
    assigned: Dict[Position, NodeType],
) -> MapGraph:
    positions_sorted = sorted(positions, key=lambda p: (p[0], p[1]))
    pos_to_id = {pos: idx for idx, pos in enumerate(positions_sorted)}
    nodes: Dict[int, MapNode] = {}

    for pos in positions_sorted:
        floor, col = pos
        node_id = pos_to_id[pos]

        nodes[node_id] = MapNode(
            node_id=node_id,
            depth=floor,
            idx=col,
            x=_node_x(col),
            y=_node_y(floor),
            node_type=assigned[pos],
            next_ids=[],
        )

    boss_id = len(nodes)

    nodes[boss_id] = MapNode(
        node_id=boss_id,
        depth=GRID_FLOORS,
        idx=GRID_COLS // 2,
        x=0.5,
        y=0.0,
        node_type=NodeType.BOSS,
        next_ids=[],
    )

    for src in positions_sorted:
        src_id = pos_to_id[src]
        destinations = sorted(outgoing.get(src, set()), key=lambda p: (p[0], p[1]))
        nodes[src_id].next_ids = [pos_to_id[dst] for dst in destinations]

    final_floor_positions = sorted(
        (p for p in positions if p[0] == GRID_FLOORS - 1),
        key=lambda p: p[1],
    )

    for pos in final_floor_positions:
        src_id = pos_to_id[pos]

        if boss_id not in nodes[src_id].next_ids:
            nodes[src_id].next_ids.append(boss_id)

    first_floor_positions = sorted(
        (p for p in positions if p[0] == 0),
        key=lambda p: p[1],
    )

    start_id = len(nodes)

    nodes[start_id] = MapNode(
        node_id=start_id,
        depth=-1,
        idx=GRID_COLS // 2,
        x=0.5,
        y=1.0,
        node_type=NodeType.START,
        next_ids=[pos_to_id[pos] for pos in first_floor_positions],
    )

    return MapGraph(
        nodes=nodes,
        start_id=start_id,
        boss_id=boss_id,
    )


def generate_map(
    counts: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> MapGraph:
    rng = Random(seed)

    for _ in range(MAX_MAP_ATTEMPTS):
        try:
            paths = _generate_paths(rng)
        except RuntimeError:
            continue

        positions, outgoing, incoming = _build_position_graph(paths)

        if _has_impossible_fixed_destination_conflict(outgoing):
            continue

        assigned = _assign_locations(rng, positions, outgoing, incoming)

        if assigned is None:
            continue

        return _build_map_graph(positions, outgoing, assigned)

    raise RuntimeError("failed to generate a valid 7x15 non-crossing map")