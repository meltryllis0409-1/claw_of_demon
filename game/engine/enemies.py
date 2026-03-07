from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Enemy:
    enemy_id: str
    tier: str
    name: str
    max_hp: int
    hp: int
    attack_damage: int
    attack_timer_max: int
    attack_timer: int

    # Optional mechanics (backward compatible with old JSON)
    heal_on_attack: int = 0
    frenzy_hp_threshold: Optional[int] = None
    frenzy_attacks: int = 1

    disable_suit_on_attack: bool = False
    disable_suit_turns: int = 0

    double_damage_chance: float = 0.0  # 0.0 ~ 1.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Enemy":
        return Enemy(
            enemy_id=d.get("id") or d.get("enemy_id") or d.get("enemyId") or "",
            tier=str(d.get("tier", "normal")),
            name=str(d.get("name", "")),
            max_hp=int(d.get("hp", d.get("max_hp", 0))),
            hp=int(d.get("hp", d.get("max_hp", 0))),
            attack_damage=int(d.get("attack_damage", d.get("atk", 0))),
            attack_timer_max=int(d.get("attack_timer", d.get("timer", 1))),
            attack_timer=int(d.get("attack_timer", d.get("timer", 1))),
            heal_on_attack=int(d.get("heal_on_attack", 0)),
            frenzy_hp_threshold=(int(d["frenzy_hp_threshold"]) if "frenzy_hp_threshold" in d else None),
            frenzy_attacks=int(d.get("frenzy_attacks", 1)),
            disable_suit_on_attack=bool(d.get("disable_suit_on_attack", False)),
            disable_suit_turns=int(d.get("disable_suit_turns", 0)),
            double_damage_chance=float(d.get("double_damage_chance", 0.0)),
        )

    def reset_timer(self) -> None:
        self.attack_timer = self.attack_timer_max

    def attack_count(self) -> int:
        if self.frenzy_hp_threshold is not None and self.hp < self.frenzy_hp_threshold:
            return max(1, self.frenzy_attacks)
        return 1
