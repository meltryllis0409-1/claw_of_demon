from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .mapgen import MapGraph, generate_map


@dataclass
class RunState:
    max_hp: int = 300
    hp: int = 300
    gold: int = 50

    unlocked_slots: int = 1
    owned_sigils: Set[str] = field(default_factory=set)
    equipped_sigils: List[str] = field(default_factory=list)
    sigil_slot_buy_count: int = 0

    greed_potion_battles_left: int = 0
    stone_skin_potion_battles_left: int = 0
    rage_potion_battles_left: int = 0
    resistance_potion_battles_left: int = 0
    regen_potion_turns_left: int = 0
    regen_potion_amount: int = 10

    map_counts: List[int] = field(default_factory=lambda: [1, 2, 3, 3, 2, 1])
    map_graph: MapGraph = field(default_factory=lambda: generate_map([1, 2, 3, 3, 2, 1]))
    current_node_id: int = 0

    def reset_map(self) -> None:
        self.map_graph = generate_map(self.map_counts)
        self.current_node_id = self.map_graph.start_id
