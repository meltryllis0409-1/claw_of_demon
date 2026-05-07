from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from random import choice, random, sample

from .cards import Card, Deck, Suit
from .enemies import Enemy
from .evaluator import evaluate, EvalResult, AttackType
from .sigils import Sigil, apply_damage_sigils, regen_amount


def _sigil_type(sigil: Sigil) -> str:
    return str(
        getattr(
            sigil,
            "type",
            getattr(
                sigil,
                "sigil_type",
                getattr(
                    sigil,
                    "effect_type",
                    getattr(sigil, "kind", getattr(sigil, "sigil_id", getattr(sigil, "id", ""))),
                ),
            ),
        )
    )


def _sigil_id(sigil: Sigil) -> str:
    return str(getattr(sigil, "sigil_id", getattr(sigil, "id", "")))


def _sigil_matches(sigil: Sigil, sigil_type: str) -> bool:
    aliases = {
        "first_attack_multiplier": {"first_attack_multiplier", "preemptive_strike"},
        "lifesteal_percent": {"lifesteal_percent", "soul_siphon"},
        "hand_limit_bonus": {"hand_limit_bonus", "faith_strength"},
        "gold_shield": {"gold_shield", "golden_cloth_armor"},
        "low_hp_damage_reduction": {"low_hp_damage_reduction", "blood_armor"},
        "low_hp_damage_bonus": {"low_hp_damage_bonus", "berserker"},
        "diamond_gold_bonus": {"diamond_gold_bonus", "diamond_greed"},
        "heart_heal": {"heart_heal", "heart_oath"},
        "march_damage_bonus": {"march_damage_bonus", "chain_march_boots"},
        "shadow_slash": {"shadow_slash", "shadow_cut"},
        "crown_rift": {"crown_rift", "crown裂印", "crown裂印"},
        "blood_stair": {"blood_stair", "blood_stairway"},
        "steel_alliance": {"steel_alliance", "iron_alliance"},
        "dual_serpent_ring": {"dual_serpent_ring", "double_snake_ring"},
        "three_eye_totem": {"three_eye_totem", "triple_eye_totem"},
        "full_moon_bone": {"full_moon_bone", "full_moon_beast_bone"},
        "four_corner_cage": {"four_corner_cage", "fourfold_cage"},
        "decapitation_flag": {"decapitation_flag", "execution_flag"},
        "gambler_finger": {"gambler_finger", "gambler_broken_finger"},
        "mirror_card_box": {"mirror_card_box", "mirror_deck_box"},
        "reverse_gear": {"reverse_gear", "reverse_sequence_gear"},
        "starbreaker_gauntlet": {"starbreaker_gauntlet", "star_shatter_gauntlet"},
        "judgement_scale": {"judgement_scale", "judgment_scale"},
    }
    accepted = aliases.get(sigil_type, {sigil_type})
    return _sigil_type(sigil) in accepted or _sigil_id(sigil) in accepted


def _sigil_value(sigil: Sigil, default: int = 0) -> int:
    try:
        return int(getattr(sigil, "value", default))
    except Exception:
        return default


@dataclass
class CombatState:
    player_max_hp: int = 9999
    player_hp: int = 9999
    gold: int = 0
    player_shield: int = 0
    player_block: int = 0
    enemy_shield: int = 0
    enemy_block: int = 0
    enemy_double_next_attack: bool = False
    enemy_skip_next_attack: bool = False

    deck: Deck = field(default_factory=Deck)
    hand: List[Card] = field(default_factory=list)

    discard_uses_left: int = 3
    enemy: Optional[Enemy] = None
    sigils: List[Sigil] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    damage_events: List[Dict[str, Any]] = field(default_factory=list)
    enemy_turn_pending: bool = False

    disabled_suits: Dict[Suit, int] = field(default_factory=dict)

    player_poison_stacks: int = 0
    player_poison_damage_per_stack: int = 2

    awakened_card_ids: Set[int] = field(default_factory=set)
    player_attack_count: int = 0
    battle_diamond_bonus_gold: int = 0
    enemy_wound_stacks: int = 0
    gambler_eye_used: bool = False
    last_stand_used: bool = False
    last_stand_damage_ready: bool = False
    demon_counter_cards: int = 0
    demon_counter_ready: bool = False
    previous_attack_type: Optional[AttackType] = None
    next_enemy_damage_reduction_percent: int = 0
    next_enemy_flat_damage_reduction: int = 0
    four_corner_cage_triggers: int = 0

    potion_damage_bonus_percent: int = 0
    potion_gold_bonus_percent: int = 0
    potion_damage_taken_bonus_percent: int = 0
    potion_damage_taken_reduction_percent: int = 0
    potion_regen_turns_left: int = 0
    potion_regen_amount: int = 10

    def start(self, enemy: Enemy) -> None:
        self.enemy = enemy
        self.enemy.hp = self.enemy.max_hp
        self.enemy.reset_timer()

        self.discard_uses_left = 3
        self.deck.reset()
        self.hand = self.deck.draw_many(8)

        self.disabled_suits = {}
        self.player_poison_stacks = 0
        self.player_poison_damage_per_stack = int(getattr(enemy, "poison_damage_per_stack", 2) or 2)
        self.awakened_card_ids = set()
        self.player_attack_count = 0
        self.battle_diamond_bonus_gold = 0
        self.enemy_wound_stacks = 0
        self.gambler_eye_used = False
        self.last_stand_used = False
        self.last_stand_damage_ready = False
        self.demon_counter_cards = 0
        self.demon_counter_ready = False
        self.previous_attack_type = None
        self.next_enemy_damage_reduction_percent = 0
        self.next_enemy_flat_damage_reduction = 0
        self.four_corner_cage_triggers = 0
        self.potion_damage_bonus_percent = 0
        self.potion_gold_bonus_percent = 0
        self.potion_damage_taken_bonus_percent = 0
        self.potion_damage_taken_reduction_percent = 0
        self.potion_regen_turns_left = 0
        self.potion_regen_amount = 10
        self.log = [f"遭遇敵人: {enemy.name}"]
        self.damage_events = []
        self.enemy_turn_pending = False
        self.player_shield = 0
        self.player_block = 0
        self.enemy_shield = 0
        self.enemy_block = 0
        self.enemy_double_next_attack = False
        self.enemy_skip_next_attack = False

        starting_shield = max(0, int(getattr(enemy, "starting_shield", 0) or 0))
        if starting_shield > 0:
            self.enemy_shield = starting_shield
            self.log.append(f"{enemy.name} 帶有 {starting_shield} 點護盾。")

    def refresh_sigil_state(self) -> None:
        self._refill_hand_to_8()
        self._roll_awakened_cards()

    def is_over(self) -> bool:
        if self.enemy is None:
            return True
        return self.player_hp <= 0 or self.enemy.hp <= 0

    def _has_sigil(self, sigil_type: str) -> bool:
        return any(_sigil_matches(s, sigil_type) for s in self.sigils)

    def _sigil_total(self, sigil_type: str, default: int = 0) -> int:
        values = [_sigil_value(s, default) for s in self.sigils if _sigil_matches(s, sigil_type)]
        return sum(values) if values else 0

    def _sigil_max(self, sigil_type: str, default: int = 0) -> int:
        values = [_sigil_value(s, default) for s in self.sigils if _sigil_matches(s, sigil_type)]
        return max(values) if values else 0

    def hand_limit(self) -> int:
        return max(1, 8 + self._sigil_total("hand_limit_bonus", 0))

    def is_awakened(self, card: Card) -> bool:
        return id(card) in self.awakened_card_ids

    def _roll_awakened_cards(self) -> None:
        self.awakened_card_ids = set()
        if not self._has_sigil("awakening") or not self.hand:
            return
        amount = max(1, self._sigil_max("awakening", 2))
        candidates = [c for c in self.hand if not self.is_card_disabled(c)]
        if not candidates:
            candidates = list(self.hand)
        picked = sample(candidates, min(amount, len(candidates)))
        self.awakened_card_ids = {id(c) for c in picked}
        self.log.append(f"覺醒：{len(picked)} 張手牌獲得紫色爪痕。")

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
                    self.log.append("警告: 過多補牌被禁用，暫時允許抽牌。")
                    return c
                continue
            return c

    def _refill_hand_to_8(self) -> None:
        while len(self.hand) < self.hand_limit():
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
            self.awakened_card_ids = {cid for cid in self.awakened_card_ids if any(id(c) == cid for c in self.hand)}
            self.log.append(f"花色禁用：強制丟棄手牌 {len(purged)} 張")
            self._refill_hand_to_8()

    def _start_of_turn(self) -> None:
        if self.enemy is None or self.is_over():
            return

        self._tick_disabled_suits()

        r = regen_amount(self.sigils)
        if r > 0:
            before = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + r)
            healed = self.player_hp - before
            if healed > 0:
                self.log.append(f"堅毅回復 {healed} 點生命值。")

        if self.potion_regen_turns_left > 0:
            before = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + max(0, int(self.potion_regen_amount)))
            healed = self.player_hp - before
            self.potion_regen_turns_left = max(0, self.potion_regen_turns_left - 1)
            if healed > 0:
                self.log.append(f"再生劑回復 {healed} 點生命值。剩餘 {self.potion_regen_turns_left} 回合。")
            else:
                self.log.append(f"再生劑效果經過 1 回合。剩餘 {self.potion_regen_turns_left} 回合。")

        if self.enemy is not None:
            enemy_block_per_turn = max(0, int(getattr(self.enemy, "block_per_turn", 0) or 0))
            if enemy_block_per_turn > 0:
                self._enemy_gain_block(enemy_block_per_turn)

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
        self._purge_disabled_from_hand()

    def _apply_enemy_poison_on_attack(self) -> None:
        if self.enemy is None:
            return

        stacks = int(getattr(self.enemy, "poison_stacks_on_attack", 0) or 0)
        if stacks <= 0 and bool(getattr(self.enemy, "poison_on_attack", False)):
            stacks = 1

        if stacks <= 0:
            return

        per_stack = int(getattr(self.enemy, "poison_damage_per_stack", self.player_poison_damage_per_stack) or self.player_poison_damage_per_stack)
        self.player_poison_damage_per_stack = max(1, per_stack)
        self.player_poison_stacks += stacks
        self.log.append(
            f"{self.enemy.name} 使你中毒 {stacks} 層。現在每打出 1 張牌會受到 {self.poison_damage_per_card()} 點傷害。"
        )

    def poison_damage_per_card(self) -> int:
        return max(0, self.player_poison_stacks * self.player_poison_damage_per_stack)

    def _record_damage_event(self, target: str, amount: int, damage_type: str = "physical", source: str = "unknown") -> None:
        amount = max(0, int(amount))
        if amount <= 0:
            return
        self.damage_events.append({
            "target": str(target),
            "amount": amount,
            "damage_type": str(damage_type),
            "source": str(source),
        })

    def gain_shield(self, amount: int) -> int:
        gained = max(0, int(amount))
        if gained <= 0:
            return 0
        self.player_shield += gained
        self.log.append(f"獲得 {gained} 點護盾。")
        return gained

    def gain_block(self, amount: int) -> int:
        gained = max(0, int(amount))
        if gained <= 0:
            return 0
        self.player_block += gained
        self.log.append(f"獲得 {gained} 點格擋。")
        return gained

    def _clear_player_block(self) -> None:
        if self.player_block > 0:
            lost = self.player_block
            self.player_block = 0
            self.log.append(f"回合結束，{lost} 點格擋消失。")

    def _apply_block_and_shield(self, dmg: int) -> int:
        dmg = max(0, int(dmg))
        if dmg <= 0:
            return 0

        if self.player_block > 0:
            used = min(self.player_block, dmg)
            self.player_block -= used
            dmg -= used
            self.log.append(f"格擋抵銷 {used} 點傷害。")

        if dmg > 0 and self.player_shield > 0:
            used = min(self.player_shield, dmg)
            self.player_shield -= used
            dmg -= used
            self.log.append(f"護盾抵銷 {used} 點傷害。")

        return dmg

    def _enemy_gain_shield(self, amount: int) -> int:
        gained = max(0, int(amount))
        if gained <= 0:
            return 0
        self.enemy_shield += gained
        if self.enemy is not None:
            self.log.append(f"{self.enemy.name} 獲得 {gained} 點護盾。")
        return gained

    def _enemy_gain_block(self, amount: int) -> int:
        gained = max(0, int(amount))
        if gained <= 0:
            return 0
        self.enemy_block += gained
        if self.enemy is not None:
            self.log.append(f"{self.enemy.name} 獲得 {gained} 點格擋。")
        return gained

    def _clear_enemy_block(self) -> None:
        if self.enemy_block > 0:
            lost = self.enemy_block
            self.enemy_block = 0
            if self.enemy is not None:
                self.log.append(f"回合結束，{self.enemy.name} 的 {lost} 點格擋消失。")

    def _apply_enemy_block_and_shield(self, dmg: int) -> int:
        dmg = max(0, int(dmg))
        if dmg <= 0:
            return 0

        if self.enemy_block > 0:
            used = min(self.enemy_block, dmg)
            self.enemy_block -= used
            dmg -= used
            if self.enemy is not None:
                self.log.append(f"{self.enemy.name} 的格擋抵銷 {used} 點傷害。")

        if dmg > 0 and self.enemy_shield > 0:
            used = min(self.enemy_shield, dmg)
            self.enemy_shield -= used
            dmg -= used
            if self.enemy is not None:
                self.log.append(f"{self.enemy.name} 的護盾抵銷 {used} 點傷害。")

        return dmg

    def _deal_damage_to_enemy(self, raw_damage: int, damage_type: str = "physical", source: str = "player_attack") -> int:
        if self.enemy is None:
            return 0
        dmg = max(0, int(raw_damage))
        dmg = self._apply_enemy_block_and_shield(dmg)
        before = self.enemy.hp
        self.enemy.hp = max(0, self.enemy.hp - dmg)
        actual = before - self.enemy.hp
        self._record_damage_event("enemy", actual, damage_type, source)
        return actual

    def _arbiter_player_turn_damage_percent(self, turn_damage: int) -> int:
        if self.enemy is None:
            return 100

        threshold = int(getattr(self.enemy, "arbiter_damage_threshold", 0) or 0)
        if threshold <= 0:
            return 100

        turn_damage = max(0, int(turn_damage))
        if turn_damage < threshold:
            reduction_percent = max(0, min(95, int(getattr(self.enemy, "arbiter_damage_reduction_percent", 50) or 50)))
            damage_percent = max(0, 100 - reduction_percent)
            attacks = max(1, int(getattr(self.enemy, "arbiter_low_damage_attacks", 2) or 2))
            self.enemy_double_next_attack = attacks >= 2
            self.enemy_skip_next_attack = False
            reduced_total = max(0, int(round(turn_damage * damage_percent / 100.0)))
            self.log.append(f"仲裁者判定：本回合傷害低於 {threshold}，總傷害 {turn_damage} → {reduced_total}，下一次攻擊變為 {attacks} 連擊。")
            return damage_percent

        if bool(getattr(self.enemy, "arbiter_high_damage_skip_attack", False)):
            self.enemy_skip_next_attack = True
            self.enemy_double_next_attack = False
            self.log.append(f"仲裁者判定：本回合傷害達到 {threshold}，下一次攻擊被取消。")
        return 100

    def _apply_played_card_poison(self, played_count: int) -> None:
        if played_count <= 0 or self.player_poison_stacks <= 0:
            return

        dmg = self.poison_damage_per_card() * played_count
        if dmg <= 0:
            return

        before = self.player_hp
        self.player_hp = max(0, self.player_hp - dmg)
        actual = before - self.player_hp
        self._record_damage_event("player", actual, "poison", "poison_on_played_cards")
        self.log.append(f"毒素侵蝕：你打出 {played_count} 張牌，受到 {actual} 點傷害。")
        if self.player_hp <= 0:
            self.log.append("戰敗!")

    def _apply_potion_player_damage_bonus(self, dmg: int) -> int:
        dmg = max(0, int(dmg))
        percent = max(0, int(self.potion_damage_bonus_percent))
        if dmg <= 0 or percent <= 0:
            return dmg
        before = dmg
        dmg = max(1, int(round(dmg * (100 + percent) / 100.0)))
        bonus = dmg - before
        if bonus > 0:
            self.log.append(f"憤怒藥劑追加 {bonus} 點傷害。")
        return dmg

    def _apply_potion_incoming_damage_modifiers(self, dmg: int) -> int:
        dmg = max(0, int(dmg))
        if dmg <= 0:
            return 0

        bonus_percent = max(0, int(self.potion_damage_taken_bonus_percent))
        if bonus_percent > 0:
            before = dmg
            dmg = max(1, int(round(dmg * (100 + bonus_percent) / 100.0)))
            increased = dmg - before
            if increased > 0:
                self.log.append(f"貪婪藥水副作用：受到傷害 +{increased}。")

        reduction_percent = max(0, min(95, int(self.potion_damage_taken_reduction_percent)))
        if reduction_percent > 0 and dmg > 0:
            before = dmg
            dmg = max(0, int(round(dmg * (100 - reduction_percent) / 100.0)))
            reduced = before - dmg
            if reduced > 0:
                self.log.append(f"抗擊藥劑減少 {reduced} 點傷害。")

        return dmg

    def _apply_enemy_damage_reduction(self, dmg: int) -> int:
        max_reduce = self._sigil_max("low_hp_damage_reduction", 0)
        if max_reduce <= 0 or self.player_max_hp <= 0 or dmg <= 0:
            return dmg
        missing_ratio = max(0.0, min(1.0, 1.0 - (self.player_hp / self.player_max_hp)))
        reduce_ratio = missing_ratio * (max_reduce / 100.0)
        reduced = max(0, int(round(dmg * (1.0 - reduce_ratio))))
        blocked = dmg - reduced
        if blocked > 0:
            self.log.append(f"以血為鎧減少 {blocked} 點傷害。")
        return reduced

    def _apply_gold_shield(self, dmg: int) -> int:
        if dmg <= 0 or not self._has_sigil("gold_shield"):
            return dmg
        spent = min(self.gold, dmg)
        if spent > 0:
            self.gold -= spent
            dmg -= spent
            self.log.append(f"金布衫消耗 {spent} 金幣抵擋傷害。")
        return dmg

    def _deal_enemy_attack_damage_to_player(self, raw_damage: int, damage_type: str = "physical", source: str = "enemy_attack") -> int:
        dmg = max(0, int(raw_damage))
        dmg = self._apply_potion_incoming_damage_modifiers(dmg)
        dmg = self._apply_enemy_damage_reduction(dmg)
        dmg = self._apply_block_and_shield(dmg)
        dmg = self._apply_gold_shield(dmg)

        before = self.player_hp
        if dmg > 0 and before - dmg <= 0 and self._has_sigil("last_stand") and not self.last_stand_used:
            self.last_stand_used = True
            self.last_stand_damage_ready = True
            self.player_hp = 1
            actual = max(0, before - self.player_hp)
            self._record_damage_event("player", actual, damage_type, source)
            self.log.append("破釜沉舟發動：你保留 1 HP，下一次攻擊傷害翻倍。")
            return actual

        self.player_hp = max(0, self.player_hp - dmg)
        actual = before - self.player_hp
        self._record_damage_event("player", actual, damage_type, source)
        return actual

    def _apply_pending_enemy_attack_modifiers(self, dmg: int) -> int:
        dmg = max(0, int(dmg))

        if self.next_enemy_flat_damage_reduction > 0:
            blocked = min(dmg, self.next_enemy_flat_damage_reduction)
            dmg -= blocked
            self.log.append(f"三眼圖騰削弱敵人攻擊，減少 {blocked} 點傷害。")
            self.next_enemy_flat_damage_reduction = 0

        if self.next_enemy_damage_reduction_percent > 0 and dmg > 0:
            percent = max(0, min(95, self.next_enemy_damage_reduction_percent))
            before = dmg
            dmg = max(0, int(round(dmg * (100 - percent) / 100.0)))
            reduced = before - dmg
            if reduced > 0:
                self.log.append(f"鋼鐵同盟削弱敵人攻擊，減少 {reduced} 點傷害。")
            self.next_enemy_damage_reduction_percent = 0

        return dmg

    def _enemy_attack_once(self) -> None:
        if self.enemy is None:
            return

        dmg = self.enemy.attack_damage
        doubled = False
        if getattr(self.enemy, "double_damage_chance", 0.0) and random() < float(self.enemy.double_damage_chance):
            dmg *= 2
            doubled = True

        dmg = self._apply_pending_enemy_attack_modifiers(dmg)
        actual = self._deal_enemy_attack_damage_to_player(dmg)
        if doubled:
            self.log.append(f"{self.enemy.name} 攻擊你，造成 {actual} 點傷害。（雙倍）")
        else:
            self.log.append(f"{self.enemy.name} 攻擊你，造成 {actual} 點傷害。")

        if getattr(self.enemy, "heal_on_attack", 0) and self.enemy.hp > 0:
            before = self.enemy.hp
            self.enemy.hp = min(self.enemy.max_hp, self.enemy.hp + int(self.enemy.heal_on_attack))
            healed = self.enemy.hp - before
            if healed > 0:
                self.log.append(f"{self.enemy.name} 回復 {healed} 點生命值。")

        self._disable_random_suit()
        self._apply_enemy_poison_on_attack()

    def _queue_enemy_turn_after_player_action(self) -> None:
        if self.enemy is None or self.is_over():
            return

        self.enemy.attack_timer -= 1
        if self.enemy.attack_timer <= 0:
            self.enemy_turn_pending = True
        else:
            self._roll_awakened_cards()

    def resolve_enemy_turn(self) -> None:
        if self.enemy is None or self.is_over() or not self.enemy_turn_pending:
            self.enemy_turn_pending = False
            return

        self.enemy_turn_pending = False

        if self.enemy_skip_next_attack:
            self.enemy_skip_next_attack = False
            self.enemy_double_next_attack = False
            self.log.append(f"{self.enemy.name} 的攻擊被取消。")
            self.enemy.reset_timer()
            self._clear_player_block()
            if not self.is_over():
                self._roll_awakened_cards()
            return

        count = 1
        if getattr(self.enemy, "frenzy_hp_threshold", None) is not None and self.enemy.hp < int(self.enemy.frenzy_hp_threshold):
            count = max(1, int(getattr(self.enemy, "frenzy_attacks", 2)))
        if self.enemy_double_next_attack:
            count = max(count, max(2, int(getattr(self.enemy, "arbiter_low_damage_attacks", 2) or 2)))
            self.enemy_double_next_attack = False

        for _ in range(count):
            self._enemy_attack_once()
            if self.player_hp <= 0:
                self.log.append("戰敗!")
                break

        self.enemy.reset_timer()
        self._clear_player_block()

        if not self.is_over():
            self._roll_awakened_cards()

    def _after_player_action(self) -> None:
        self._queue_enemy_turn_after_player_action()

    def _validate_indices(self, indices: List[int]) -> None:
        for i in indices:
            c = self.hand[i]
            if self.is_card_disabled(c):
                raise ValueError("選到被禁用花色的卡牌")

    def _is_suit(self, card: Card, suit_name: str) -> bool:
        return card.suit.name == suit_name

    def _attack_type_is_two_pair_or_better(self, result: EvalResult) -> bool:
        if getattr(result, "used_highest_single", False):
            return False
        label = result.attack_type.label()
        weak_words = ["單", "一對", "高牌"]
        return not any(word in label for word in weak_words)

    def _attack_rank(self, attack_type: AttackType) -> int:
        order = {
            AttackType.SINGLE: 1,
            AttackType.PAIR: 2,
            AttackType.TWO_PAIR: 3,
            AttackType.THREE: 4,
            AttackType.FLUSH3: 4,
            AttackType.FLUSH4: 5,
            AttackType.STRAIGHT: 6,
            AttackType.FLUSH: 7,
            AttackType.FULL_HOUSE: 8,
            AttackType.FOUR: 9,
            AttackType.STRAIGHT_FLUSH: 10,
            AttackType.DEMONS_CLAW: 11,
        }
        return order.get(attack_type, 0)

    def _is_flush_family(self, attack_type: AttackType) -> bool:
        return attack_type in (AttackType.FLUSH3, AttackType.FLUSH4, AttackType.FLUSH)

    def _is_low_straight(self, cards: List[Card]) -> bool:
        ranks = {c.rank for c in cards}
        if ranks == {14, 2, 3, 4, 5}:
            return True
        if len(ranks) == 5 and max(ranks) <= 6 and max(ranks) - min(ranks) == 4:
            return True
        return False

    def _apply_player_attack_sigils(self, dmg: int, cards: List[Card], result: EvalResult) -> int:
        played_count = result.played_count

        if self.enemy_wound_stacks > 0:
            bonus = self.enemy_wound_stacks * self._sigil_max("club_fang", 2)
            dmg += bonus
            self.log.append(f"梅花毒牙裂傷觸發，追加 {bonus} 點傷害。")
            self.enemy_wound_stacks = 0

        awakened_count = sum(1 for c in cards if self.is_awakened(c))
        if awakened_count > 0:
            bonus = awakened_count * 10
            dmg += bonus
            self.log.append(f"覺醒卡牌追加 {bonus} 點傷害。")

        resonance_value = self._sigil_max("card_soul_resonance", 0)
        if resonance_value > 0 and played_count >= 5:
            bonus = played_count * resonance_value
            dmg += bonus
            self.log.append(f"牌魂共鳴追加 {bonus} 點傷害。")

        spade_value = self._sigil_max("spade_brand", 0)
        spade_count = sum(1 for c in cards if self._is_suit(c, "A"))
        if spade_value > 0 and spade_count > 0:
            bonus = spade_count * spade_value
            if spade_count == played_count:
                bonus += 20
            dmg += bonus
            self.log.append(f"黑桃烙印追加 {bonus} 點傷害。")

        if self._has_sigil("march_damage_bonus") and result.attack_type == AttackType.STRAIGHT:
            percent = self._sigil_max("march_damage_bonus", 20)
            before = dmg
            dmg = max(0, int(round(dmg * (100 + percent) / 100.0)))
            self.log.append(f"鎖鏈軍靴發動：行軍傷害 {before} → {dmg}。")

        if self._has_sigil("crown_rift") and result.attack_type == AttackType.DEMONS_CLAW:
            bonus = self._sigil_max("crown_rift", 100)
            dmg += bonus
            self.log.append(f"王冠裂印發動：惡魔之爪追加 {bonus} 點傷害。")

        if self._has_sigil("dual_serpent_ring") and result.attack_type == AttackType.PAIR:
            bonus = self._sigil_max("dual_serpent_ring", 12)
            if self.previous_attack_type == AttackType.PAIR:
                bonus = 30
            dmg += bonus
            self.log.append(f"雙蛇戒發動：二元體追加 {bonus} 點傷害。")

        if self._has_sigil("full_moon_bone") and result.attack_type == AttackType.FULL_HOUSE:
            percent = self._sigil_max("full_moon_bone", 25)
            before = dmg
            dmg = max(0, int(round(dmg * (100 + percent) / 100.0)))
            self.log.append(f"滿月獸骨發動：戰團傷害 {before} → {dmg}。")

        if self._has_sigil("decapitation_flag") and self.enemy is not None and self.enemy.max_hp > 0:
            if self.enemy.hp / self.enemy.max_hp <= 0.25 and not getattr(result, "used_highest_single", False):
                percent = self._sigil_max("decapitation_flag", 40)
                before = dmg
                dmg = max(0, int(round(dmg * (100 + percent) / 100.0)))
                self.log.append(f"斷頭旗發動：斬殺傷害 {before} → {dmg}。")

        if self._has_sigil("gambler_finger") and result.attack_type == AttackType.SINGLE:
            chance = self._sigil_max("gambler_finger", 35) / 100.0
            before = dmg
            if random() < chance:
                dmg *= 2
                self.log.append(f"賭徒的殘指成功：單挑傷害 {before} → {dmg}。")
            else:
                dmg = max(0, int(round(dmg * 0.7)))
                self.log.append(f"賭徒的殘指失敗：單挑傷害 {before} → {dmg}。")

        if self._has_sigil("mirror_card_box") and self.previous_attack_type == result.attack_type:
            percent = self._sigil_max("mirror_card_box", 35)
            before = dmg
            dmg = max(0, int(round(dmg * (100 + percent) / 100.0)))
            self.log.append(f"鏡像牌匣發動：連續牌型傷害 {before} → {dmg}。")

        if self._has_sigil("reverse_gear") and result.attack_type == AttackType.STRAIGHT and self._is_low_straight(cards):
            bonus = self._sigil_max("reverse_gear", 40)
            dmg += bonus
            self.log.append(f"逆序齒輪發動：低階行軍追加 {bonus} 點傷害。")

        if self._has_sigil("starbreaker_gauntlet") and played_count >= 5:
            per_card = self._sigil_max("starbreaker_gauntlet", 8)
            bonus = played_count * per_card
            dmg += bonus
            self.log.append(f"碎星手套發動：追加 {bonus} 點傷害。")

        if self._has_sigil("judgement_scale") and self.previous_attack_type is not None:
            current_rank = self._attack_rank(result.attack_type)
            previous_rank = self._attack_rank(self.previous_attack_type)
            if current_rank < previous_rank:
                percent = self._sigil_max("judgement_scale", 25)
                before = dmg
                dmg = max(0, int(round(dmg * (100 + percent) / 100.0)))
                self.log.append(f"審判天秤發動：牌型下降，傷害 {before} → {dmg}。")

        if self._has_sigil("gambler_eye") and not self.gambler_eye_used and self._attack_type_is_two_pair_or_better(result):
            self.gambler_eye_used = True
            reward = choice(["gold", "heal", "damage"])
            value = self._sigil_max("gambler_eye", 30)
            if reward == "gold":
                self.gold += value
                self.log.append(f"賭徒之眼發動：獲得 {value} 金幣。")
            elif reward == "heal":
                before = self.player_hp
                self.player_hp = min(self.player_max_hp, self.player_hp + 10)
                self.log.append(f"賭徒之眼發動：回復 {self.player_hp - before} HP。")
            else:
                dmg += value
                self.log.append(f"賭徒之眼發動：追加 {value} 點傷害。")

        if self._has_sigil("first_attack_multiplier") and self.player_attack_count == 0:
            mult = self._sigil_max("first_attack_multiplier", 150)
            if mult <= 0:
                mult = 150
            before = dmg
            dmg = max(0, int(round(dmg * mult / 100.0)))
            self.log.append(f"先發制人發動：傷害 {before} → {dmg}。")

        low_hp_bonus = self._sigil_max("low_hp_damage_bonus", 0)
        if low_hp_bonus > 0 and self.player_max_hp > 0:
            missing_ratio = max(0.0, min(1.0, 1.0 - (self.player_hp / self.player_max_hp)))
            bonus_ratio = missing_ratio * (low_hp_bonus / 100.0)
            if bonus_ratio > 0:
                before = dmg
                dmg = max(0, int(round(dmg * (1.0 + bonus_ratio))))
                if dmg > before:
                    self.log.append(f"狂戰士發動：傷害 {before} → {dmg}。")

        if self.last_stand_damage_ready:
            before = dmg
            dmg *= 2
            self.last_stand_damage_ready = False
            self.log.append(f"破釜沉舟反擊：傷害 {before} → {dmg}。")

        if self.demon_counter_ready:
            bonus = self._sigil_max("demon_counter", 66)
            dmg += bonus
            self.demon_counter_ready = False
            before_hp = self.player_hp
            self.player_hp = max(0, self.player_hp - 6)
            self._record_damage_event("player", before_hp - self.player_hp, "ritual", "demon_counter")
            self.log.append(f"惡魔計數器發動：追加 {bonus} 點傷害，失去 6 HP。")
            if self.player_hp <= 0:
                self.log.append("戰敗!")

        if self._has_sigil("steel_alliance") and self._is_flush_family(result.attack_type):
            value = self._sigil_max("steel_alliance", 30)
            self.next_enemy_damage_reduction_percent = max(self.next_enemy_damage_reduction_percent, value)
            self.log.append(f"鋼鐵同盟發動：下一次敵人攻擊降低 {value}% 傷害。")

        if self._has_sigil("three_eye_totem") and result.attack_type == AttackType.THREE:
            value = self._sigil_max("three_eye_totem", 5)
            self.next_enemy_flat_damage_reduction += value
            self.log.append(f"三眼圖騰發動：下一次敵人攻擊傷害 -{value}。")

        if self._has_sigil("four_corner_cage") and result.attack_type == AttackType.FOUR:
            limit = 2
            if self.four_corner_cage_triggers < limit and self.enemy is not None:
                self.enemy.attack_timer += 1
                self.four_corner_cage_triggers += 1
                self.log.append(f"四角牢籠發動：敵人攻擊倒數 +1。（本場 {self.four_corner_cage_triggers}/{limit}）")

        return max(0, dmg)

    def _apply_after_attack_sigils(self, cards: List[Card], dealt_damage: int, result: EvalResult) -> None:
        if dealt_damage > 0:
            lifesteal = self._sigil_max("lifesteal_percent", 0)
            if lifesteal > 0:
                heal = max(1, int(dealt_damage * lifesteal / 100.0))
                before = self.player_hp
                self.player_hp = min(self.player_max_hp, self.player_hp + heal)
                healed = self.player_hp - before
                if healed > 0:
                    self.log.append(f"靈魂虹吸回復 {healed} HP。")

        heart_value = self._sigil_max("heart_heal", 0)
        if heart_value > 0:
            heart_count = sum(1 for c in cards if self._is_suit(c, "B"))
            if heart_count > 0:
                heal = heart_count * heart_value
                if heart_count >= 3:
                    heal += 5
                before = self.player_hp
                self.player_hp = min(self.player_max_hp, self.player_hp + heal)
                healed = self.player_hp - before
                if healed > 0:
                    self.log.append(f"紅心誓約回復 {healed} HP。")

        diamond_value = self._sigil_max("diamond_gold_bonus", 0)
        if diamond_value > 0:
            diamond_count = sum(1 for c in cards if self._is_suit(c, "C"))
            if diamond_count > 0:
                gained = diamond_count * diamond_value
                self.battle_diamond_bonus_gold += gained
                self.log.append(f"方塊貪婪累積 {gained} 金幣獎勵。")

        club_value = self._sigil_max("club_fang", 0)
        if club_value > 0:
            club_count = sum(1 for c in cards if self._is_suit(c, "D"))
            if club_count > 0:
                self.enemy_wound_stacks += club_count
                self.log.append(f"梅花毒牙賦予敵人 {club_count} 層裂傷。")

        if dealt_damage > 0 and self._has_sigil("blood_stair") and result.attack_type == AttackType.STRAIGHT:
            percent = self._sigil_max("blood_stair", 8)
            heal = max(1, int(dealt_damage * percent / 100.0))
            before = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + heal)
            healed = self.player_hp - before
            if healed > 0:
                self.log.append(f"血色階梯回復 {healed} HP。")

        if self._has_sigil("full_moon_bone") and result.attack_type == AttackType.FULL_HOUSE:
            before = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + 5)
            healed = self.player_hp - before
            if healed > 0:
                self.log.append(f"滿月獸骨回復 {healed} HP。")

        if self._has_sigil("judgement_scale") and self.previous_attack_type is not None:
            current_rank = self._attack_rank(result.attack_type)
            previous_rank = self._attack_rank(self.previous_attack_type)
            if current_rank > previous_rank:
                before = self.player_hp
                self.player_hp = min(self.player_max_hp, self.player_hp + 5)
                healed = self.player_hp - before
                if healed > 0:
                    self.log.append(f"審判天秤發動：牌型上升，回復 {healed} HP。")

        if self._has_sigil("demon_counter"):
            self.demon_counter_cards += result.played_count
            if self.demon_counter_cards >= 6:
                self.demon_counter_cards %= 6
                self.demon_counter_ready = True
                self.log.append("惡魔計數器充能完成：下一次攻擊追加傷害。")

    def _grant_victory_battle_rewards(self) -> None:
        if self.battle_diamond_bonus_gold <= 0:
            return
        if self.enemy is not None and getattr(self.enemy, "enemy_id", "") == "training_dummy":
            self.battle_diamond_bonus_gold = 0
            return
        reward = self.battle_diamond_bonus_gold
        self.gold += reward
        self.battle_diamond_bonus_gold = 0
        self.log.append(f"方塊貪婪結算：額外獲得 {reward} 金幣。")

    def play(self, indices: List[int]) -> EvalResult:
        if self.enemy is None:
            raise RuntimeError("戰鬥尚未開始")
        if self.is_over():
            raise ValueError("戰鬥結束")
        if self.enemy_turn_pending:
            raise ValueError("敵人正在準備攻擊")
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

        dmg = self._apply_player_attack_sigils(dmg, cards, result)
        dmg = self._apply_potion_player_damage_bonus(dmg)

        shadow_bonus_damage = 0
        shadow_slash_ready = self._has_sigil("shadow_slash") and result.attack_type == AttackType.TWO_PAIR
        if shadow_slash_ready:
            shadow_bonus_damage = max(1, int(round(dmg * 0.5)))

        arbiter_damage_percent = self._arbiter_player_turn_damage_percent(dmg + shadow_bonus_damage)
        if arbiter_damage_percent != 100:
            dmg = max(0, int(round(dmg * arbiter_damage_percent / 100.0)))
            shadow_bonus_damage = max(0, int(round(shadow_bonus_damage * arbiter_damage_percent / 100.0)))

        dealt_damage = self._deal_damage_to_enemy(dmg, source="player_attack")

        if shadow_slash_ready and shadow_bonus_damage > 0 and self.enemy.hp > 0:
            shadow_dealt = self._deal_damage_to_enemy(shadow_bonus_damage, source="shadow_slash")
            dealt_damage += shadow_dealt
            self.log.append(f"影斬發動：第二擊造成 {shadow_bonus_damage} 點傷害。")

        used = set(indices)
        self.hand = [c for j, c in enumerate(self.hand) if j not in used]
        self.awakened_card_ids = {cid for cid in self.awakened_card_ids if any(id(c) == cid for c in self.hand)}
        self._refill_hand_to_8()

        if result.used_highest_single:
            self.log.append(f"你使用 單挑(取最高單張) 造成 {dmg} 點傷害。")
        else:
            self.log.append(f"你使用 {result.attack_type.label()} 造成 {dmg} 點傷害。")

        self._apply_after_attack_sigils(cards, dealt_damage, result)
        self._apply_played_card_poison(result.played_count)
        self.player_attack_count += 1
        self.previous_attack_type = result.attack_type
        self._clear_enemy_block()

        if self.enemy.hp <= 0:
            self._grant_victory_battle_rewards()
            self.log.append("勝利!")
            self._clear_player_block()
        elif self.player_hp <= 0:
            self._clear_player_block()
        else:
            self._queue_enemy_turn_after_player_action()
            if not self.enemy_turn_pending:
                self._clear_player_block()

        return result

    def discard(self, indices: List[int]) -> None:
        if self.enemy is None:
            raise RuntimeError("戰鬥尚未開始")
        if self.is_over():
            raise ValueError("戰鬥結束")
        if self.enemy_turn_pending:
            raise ValueError("敵人正在準備攻擊，無法棄牌")
        if self.discard_uses_left <= 0:
            raise ValueError("已耗盡棄牌次數")
        if not indices:
            raise ValueError("請選擇要丟棄的卡牌")
        if len(indices) > 5:
            raise ValueError("您最多僅可以捨棄五枚卡牌")

        self._validate_indices(indices)

        used = set(indices)
        self.hand = [c for j, c in enumerate(self.hand) if j not in used]
        self.awakened_card_ids = {cid for cid in self.awakened_card_ids if any(id(c) == cid for c in self.hand)}
        self._refill_hand_to_8()
        self.discard_uses_left -= 1

        self.log.append(f"您捨棄了 {len(indices)} 張卡牌，還剩餘 {self.discard_uses_left} 次棄牌次數。")


def load_enemies_json(path: Path) -> List[Enemy]:
    data = json.loads(path.read_text(encoding="utf-8"))
    enemies: List[Enemy] = []
    for raw in data.get("enemies", []):
        enemy = Enemy.from_dict(raw)

        if "poison_on_attack" in raw:
            setattr(enemy, "poison_on_attack", bool(raw.get("poison_on_attack", False)))
        if "poison_stacks_on_attack" in raw:
            setattr(enemy, "poison_stacks_on_attack", int(raw.get("poison_stacks_on_attack", 0)))
        if "poison_damage_per_stack" in raw:
            setattr(enemy, "poison_damage_per_stack", int(raw.get("poison_damage_per_stack", 2)))

        for key in (
            "starting_shield",
            "block_per_turn",
            "arbiter_damage_threshold",
            "arbiter_damage_reduction_percent",
            "arbiter_low_damage_attacks",
        ):
            if key in raw:
                setattr(enemy, key, int(raw.get(key, 0)))
        if "arbiter_high_damage_skip_attack" in raw:
            setattr(enemy, "arbiter_high_damage_skip_attack", bool(raw.get("arbiter_high_damage_skip_attack", False)))

        enemies.append(enemy)
    return enemies


def load_sigils_json(path: Path) -> List[Sigil]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Sigil.from_dict(s) for s in data.get("sigils", [])]
