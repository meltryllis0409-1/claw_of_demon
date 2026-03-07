from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from random import choice, random

from .cards import Card, Deck, Suit
from .enemies import Enemy
from .evaluator import evaluate, EvalResult, AttackType
from .sigils import Sigil, apply_damage_sigils, regen_amount


@dataclass
class CombatState:
    player_max_hp: int = 9999
    player_hp: int = 9999
    gold: int = 0

    deck: Deck = field(default_factory=Deck)
    hand: List[Card] = field(default_factory=list)

    discard_uses_left: int = 3
    enemy: Optional[Enemy] = None
    sigils: List[Sigil] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    # suit -> remaining turns (counts down on each player action)
    disabled_suits: Dict[Suit, int] = field(default_factory=dict)

    def start(self, enemy: Enemy) -> None:
        self.enemy = enemy
        self.enemy.hp = self.enemy.max_hp
        self.enemy.reset_timer()

        self.discard_uses_left = 3
        self.deck.reset()
        self.hand = self.deck.draw_many(8)

        self.disabled_suits = {}
        self.log = [f"遭遇敵人: {enemy.name}"]

    def is_over(self) -> bool:
        if self.enemy is None:
            return True
        return self.player_hp <= 0 or self.enemy.hp <= 0

    def suit_disabled_turns(self, suit: Suit) -> int:
        return int(self.disabled_suits.get(suit, 0))

    def is_card_disabled(self, card: Card) -> bool:
        return self.suit_disabled_turns(card.suit) > 0

    def _tick_disabled_suits(self) -> None:
        if not self.disabled_suits:
            return
        to_del = []
        for s, t in list(self.disabled_suits.items()):
            nt = t - 1
            if nt <= 0:
                to_del.append(s)
            else:
                self.disabled_suits[s] = nt
        for s in to_del:
            del self.disabled_suits[s]
            self.log.append(f"花色禁用解除: {s.name}")

    def _draw_one_non_disabled(self) -> Card:
        tries = 0
        while True:
            c = self.deck.draw()
            if self.is_card_disabled(c):
                self.log.append(f"強制丟棄補牌: {c.rank_str()}({c.suit.name})")
                tries += 1
                if tries >= 64:
                    # safety escape
                    self.log.append("警告: 過多補牌被禁用，暫時允許抽牌。")
                    return c
                continue
            return c

    def _refill_hand_to_8(self) -> None:
        while len(self.hand) < 8:
            self.hand.append(self._draw_one_non_disabled())

    def _purge_disabled_from_hand(self) -> None:
        if not self.disabled_suits:
            return
        kept: List[Card] = []
        purged: List[Card] = []
        for c in self.hand:
            if self.is_card_disabled(c):
                purged.append(c)
            else:
                kept.append(c)
        if purged:
            self.hand = kept
            self.log.append(f"花色禁用：強制丟棄手牌 {len(purged)} 張")
            self._refill_hand_to_8()

    def _start_of_turn(self) -> None:
        if self.enemy is None or self.is_over():
            return

        # a "turn" advances on each player action (play/discard)
        self._tick_disabled_suits()

        r = regen_amount(self.sigils)
        if r > 0:
            before = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + r)
            healed = self.player_hp - before
            if healed > 0:
                self.log.append(f"堅毅回復 {healed} 點生命值。")

        self._purge_disabled_from_hand()

    def _disable_random_suit(self) -> None:
        if self.enemy is None:
            return
        if not self.enemy.disable_suit_on_attack or self.enemy.disable_suit_turns <= 0:
            return

        available = [s for s in Suit if s not in self.disabled_suits]
        s = choice(available) if available else choice(list(Suit))
        self.disabled_suits[s] = self.enemy.disable_suit_turns
        self.log.append(f"{self.enemy.name} 禁用花色: {s.name}（{self.enemy.disable_suit_turns} 回合）")

        # immediate forced discard + keep refills clean
        self._purge_disabled_from_hand()

    def _enemy_attack_once(self) -> None:
        if self.enemy is None:
            return

        dmg = self.enemy.attack_damage
        doubled = False
        if getattr(self.enemy, "double_damage_chance", 0.0) and random() < float(self.enemy.double_damage_chance):
            dmg *= 2
            doubled = True

        self.player_hp = max(0, self.player_hp - dmg)
        if doubled:
            self.log.append(f"{self.enemy.name} 攻擊你，造成 {dmg} 點傷害。（雙倍）")
        else:
            self.log.append(f"{self.enemy.name} 攻擊你，造成 {dmg} 點傷害。")

        # heal-on-attack (elite blood gellzunk)
        if getattr(self.enemy, "heal_on_attack", 0) and self.enemy.hp > 0:
            before = self.enemy.hp
            self.enemy.hp = min(self.enemy.max_hp, self.enemy.hp + int(self.enemy.heal_on_attack))
            healed = self.enemy.hp - before
            if healed > 0:
                self.log.append(f"{self.enemy.name} 回復 {healed} 點生命值。")

        self._disable_random_suit()

    def _after_player_action(self) -> None:
        if self.enemy is None or self.is_over():
            return

        self.enemy.attack_timer -= 1
        if self.enemy.attack_timer <= 0:
            count = 1
            if getattr(self.enemy, "frenzy_hp_threshold", None) is not None and self.enemy.hp < int(self.enemy.frenzy_hp_threshold):
                count = max(1, int(getattr(self.enemy, "frenzy_attacks", 2)))

            for _ in range(count):
                self._enemy_attack_once()
                if self.player_hp <= 0:
                    self.log.append("戰敗!")
                    break

            self.enemy.reset_timer()

    def _validate_indices(self, indices: List[int]) -> None:
        for i in indices:
            c = self.hand[i]
            if self.is_card_disabled(c):
                raise ValueError("選到被禁用花色的卡牌")

    def play(self, indices: List[int]) -> EvalResult:
        if self.enemy is None:
            raise RuntimeError("戰鬥尚未開始")
        if self.is_over():
            raise ValueError("戰鬥結束")
        if not indices:
            raise ValueError("請選擇要使用的卡牌")

        self._start_of_turn()
        self._validate_indices(indices)

        cards = [self.hand[i] for i in indices]
        result = evaluate(cards)

        dmg, gold_gain = apply_damage_sigils(
            damage=result.damage,
            sigils=self.sigils,
            played_count=result.played_count,
            attack_type=result.attack_type,
        )
        if gold_gain > 0:
            self.gold += gold_gain
            self.log.append(f"竊盜高手獲得 {gold_gain} 金幣。")

        self.enemy.hp = max(0, self.enemy.hp - dmg)

        used = set(indices)
        self.hand = [c for j, c in enumerate(self.hand) if j not in used]
        self._refill_hand_to_8()

        if result.used_highest_single:
            self.log.append(f"你使用 單挑(取最高單張) 造成 {dmg} 點傷害。")
        else:
            self.log.append(f"你使用 {result.attack_type.label()} 造成 {dmg} 點傷害。")

        if self.enemy.hp <= 0:
            self.log.append("勝利!")
        else:
            self._after_player_action()

        return result

    def discard(self, indices: List[int]) -> None:
        if self.enemy is None:
            raise RuntimeError("戰鬥尚未開始")
        if self.is_over():
            raise ValueError("戰鬥結束")
        if self.discard_uses_left <= 0:
            raise ValueError("已耗盡棄牌次數")
        if not indices:
            raise ValueError("請選擇要丟棄的卡牌")
        if len(indices) > 5:
            raise ValueError("您最多僅可以捨棄五枚卡牌")

        self._start_of_turn()
        self._validate_indices(indices)

        used = set(indices)
        self.hand = [c for j, c in enumerate(self.hand) if j not in used]
        self._refill_hand_to_8()
        self.discard_uses_left -= 1

        self.log.append(f"您捨棄了 {len(indices)} 張卡牌，還剩餘 {self.discard_uses_left} 次棄牌次數。")
        self._after_player_action()


def load_enemies_json(path: Path) -> List[Enemy]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Enemy.from_dict(e) for e in data.get("enemies", [])]


def load_sigils_json(path: Path) -> List[Sigil]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Sigil.from_dict(s) for s in data.get("sigils", [])]
