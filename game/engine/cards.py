from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import shuffle
from typing import List


class Suit(Enum):
    A = 0  # 黑桃
    B = 1  # 紅心
    C = 2  # 方塊
    D = 3  # 梅花


@dataclass(frozen=True)
class Card:
    rank: int
    suit: Suit

    def rank_str(self) -> str:
        r = self.rank
        if r == 14:
            return "A"
        if r == 13:
            return "K"
        if r == 12:
            return "Q"
        if r == 11:
            return "J"
        return str(r)


class Deck:
    def __init__(self) -> None:
        self._cards: List[Card] = []
        self.reset()

    def reset(self) -> None:
        self._cards = [Card(rank=r, suit=s) for s in Suit for r in range(2, 15)]
        shuffle(self._cards)

    def draw(self) -> Card:
        if not self._cards:
            self.reset()
        return self._cards.pop()

    def draw_many(self, n: int) -> List[Card]:
        return [self.draw() for _ in range(n)]
