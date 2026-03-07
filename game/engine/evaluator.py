from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from collections import Counter
from typing import List, Optional

from .cards import Card


class AttackType(Enum):
    SINGLE = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE = auto()
    STRAIGHT = auto()
    FLUSH3 = auto()
    FLUSH4 = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR = auto()
    STRAIGHT_FLUSH = auto()
    DEMONS_CLAW = auto()

    def label(self) -> str:
        return {
            AttackType.SINGLE: "單挑",
            AttackType.PAIR: "二元體",
            AttackType.TWO_PAIR: "雙重二元",
            AttackType.THREE: "三重奏",
            AttackType.STRAIGHT: "行軍",
            AttackType.FLUSH3: "部落(3)",
            AttackType.FLUSH4: "部落(4)",
            AttackType.FLUSH: "部落(5)",
            AttackType.FULL_HOUSE: "戰團",
            AttackType.FOUR: "四分體",
            AttackType.STRAIGHT_FLUSH: "戰爭部落",
            AttackType.DEMONS_CLAW: "惡魔之爪",
        }[self]


BASE_DAMAGE = {
    AttackType.SINGLE: 10,
    AttackType.PAIR: 20,
    AttackType.TWO_PAIR: 40,
    AttackType.THREE: 80,
    AttackType.STRAIGHT: 100,
    AttackType.FLUSH3: 40,
    AttackType.FLUSH4: 75,
    AttackType.FLUSH: 110,
    AttackType.FULL_HOUSE: 175,
    AttackType.FOUR: 400,
    AttackType.STRAIGHT_FLUSH: 600,
    AttackType.DEMONS_CLAW: 2000,
}


def card_value(card: Card) -> int:
    r = card.rank
    # 2-10 => face value; J/Q/K/A => 10
    if r >= 11:
        return 10
    return r


@dataclass(frozen=True)
class EvalResult:
    attack_type: AttackType
    base_damage: int
    point_sum: int
    played_count: int
    used_highest_single: bool = False

    @property
    def damage(self) -> int:
        return self.base_damage + self.point_sum


def _is_straight(ranks: List[int]) -> bool:
    r = sorted(set(ranks))
    if len(r) != 5:
        return False
    if r[-1] - r[0] == 4:
        return True
    return r == [2, 3, 4, 5, 14]


def _evaluate_5_attack_type(cards: List[Card]) -> Optional[AttackType]:
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1
    is_straight = _is_straight(ranks)

    if is_flush and set(ranks) == {10, 11, 12, 13, 14}:
        return AttackType.DEMONS_CLAW
    if is_flush and is_straight:
        return AttackType.STRAIGHT_FLUSH
    if counts == [4, 1]:
        return AttackType.FOUR
    if counts == [3, 2]:
        return AttackType.FULL_HOUSE
    if is_flush:
        return AttackType.FLUSH
    if is_straight:
        return AttackType.STRAIGHT
    if counts == [3, 1, 1]:
        return AttackType.THREE
    if counts == [2, 2, 1]:
        return AttackType.TWO_PAIR
    if counts == [2, 1, 1, 1]:
        return AttackType.PAIR
    return None


def _fallback_single(values: List[int], played_count: int) -> EvalResult:
    at = AttackType.SINGLE
    return EvalResult(
        attack_type=at,
        base_damage=BASE_DAMAGE[at],
        point_sum=max(values),
        played_count=played_count,
        used_highest_single=(played_count != 1),
    )


def evaluate(cards: List[Card]) -> EvalResult:
    n = len(cards)
    if n not in (1, 2, 3, 4, 5):
        raise ValueError("您必須使用 1 到 5 張卡牌")

    values = [card_value(c) for c in cards]

    if n == 1:
        return _fallback_single(values, played_count=1)

    if n == 2:
        if cards[0].rank == cards[1].rank:
            at = AttackType.PAIR
            return EvalResult(at, BASE_DAMAGE[at], sum(values), played_count=2)
        return _fallback_single(values, played_count=2)

    if n == 3:
        if len({c.rank for c in cards}) == 1:
            at = AttackType.THREE
            return EvalResult(at, BASE_DAMAGE[at], sum(values), played_count=3)
        if len({c.suit for c in cards}) == 1:
            at = AttackType.FLUSH3
            return EvalResult(at, BASE_DAMAGE[at], sum(values), played_count=3)
        return _fallback_single(values, played_count=3)

    if n == 4:
        if len({c.rank for c in cards}) == 1:
            at = AttackType.FOUR
            return EvalResult(at, BASE_DAMAGE[at], sum(values), played_count=4)
        if len({c.suit for c in cards}) == 1:
            at = AttackType.FLUSH4
            return EvalResult(at, BASE_DAMAGE[at], sum(values), played_count=4)
        return _fallback_single(values, played_count=4)

    at5 = _evaluate_5_attack_type(cards)
    if at5 is None:
        return _fallback_single(values, played_count=5)

    return EvalResult(at5, BASE_DAMAGE[at5], sum(values), played_count=5)
