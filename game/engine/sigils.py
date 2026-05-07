from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from .evaluator import AttackType


SIGIL_RARITY_COMMON = "常見"
SIGIL_RARITY_RARE = "稀有"
SIGIL_RARITY_EPIC = "史詩"
SIGIL_RARITY_LEGENDARY = "傳說"

SIGIL_RARITY_ORDER: Dict[str, int] = {
    SIGIL_RARITY_COMMON: 1,
    SIGIL_RARITY_RARE: 2,
    SIGIL_RARITY_EPIC: 3,
    SIGIL_RARITY_LEGENDARY: 4,
}

SIGIL_RARITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    SIGIL_RARITY_COMMON: (86, 220, 116),
    SIGIL_RARITY_RARE: (92, 166, 255),
    SIGIL_RARITY_EPIC: (190, 106, 255),
    SIGIL_RARITY_LEGENDARY: (245, 190, 55),
}

SIGIL_RARITY_ALIASES: Dict[str, str] = {
    "common": SIGIL_RARITY_COMMON,
    "rare": SIGIL_RARITY_RARE,
    "epic": SIGIL_RARITY_EPIC,
    "legendary": SIGIL_RARITY_LEGENDARY,
    "常見": SIGIL_RARITY_COMMON,
    "稀有": SIGIL_RARITY_RARE,
    "史詩": SIGIL_RARITY_EPIC,
    "傳說": SIGIL_RARITY_LEGENDARY,
}


def normalize_sigil_rarity(value: Any) -> str:
    return SIGIL_RARITY_ALIASES.get(str(value), SIGIL_RARITY_COMMON)


def sigil_rarity_color(rarity: Any) -> Tuple[int, int, int]:
    return SIGIL_RARITY_COLORS.get(normalize_sigil_rarity(rarity), SIGIL_RARITY_COLORS[SIGIL_RARITY_COMMON])


@dataclass(frozen=True)
class Sigil:
    sigil_id: str
    name: str
    type: str
    value: int
    cost: int = 0
    desc: str = ""
    rarity: str = SIGIL_RARITY_COMMON

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Sigil":
        return Sigil(
            sigil_id=d["id"],
            name=d.get("name", d["id"]),
            type=d["type"],
            value=int(d.get("value", 0)),
            cost=int(d.get("cost", 0)),
            desc=str(d.get("desc", "")),
            rarity=normalize_sigil_rarity(d.get("rarity", SIGIL_RARITY_COMMON)),
        )

    @property
    def rarity_color(self) -> Tuple[int, int, int]:
        return sigil_rarity_color(self.rarity)

    @property
    def rarity_rank(self) -> int:
        return SIGIL_RARITY_ORDER.get(normalize_sigil_rarity(self.rarity), 1)


def regen_amount(sigils: List[Sigil]) -> int:
    return sum(s.value for s in sigils if s.type == "regen_per_turn")


# --- Compatibility helpers ---
# Some older combat.py versions import apply_sigils_flat_damage(damage, sigils)
def apply_sigils_flat_damage(damage: int, sigils: List[Sigil]) -> int:
    add = sum(s.value for s in sigils if s.type == "flat_damage")
    return damage + add


# Newer pipeline uses apply_damage_sigils(damage, sigils, played_count, attack_type) -> (damage, gold_gain)
def apply_damage_sigils(
    damage: int,
    sigils: List[Sigil],
    played_count: int,
    attack_type: AttackType,
) -> Tuple[int, int]:
    dmg = apply_sigils_flat_damage(damage, sigils)

    # 5-card multiplier (團結一心)
    if any(s.type == "five_card_multiplier" for s in sigils):
        if played_count == 5 and attack_type != AttackType.SINGLE:
            dmg = (dmg * 125) // 100

    gold_gain = 0
    # Theft (竊盜高手): each 75 dmg => +1 gold, if damage > 75
    if any(s.type == "theft" for s in sigils):
        if dmg > 75:
            gold_gain = dmg // 75

    return dmg, gold_gain
