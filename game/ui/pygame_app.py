from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from random import choice

import pygame

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.engine.state import RunState
from game.engine.mapgen import NodeType, MapNode
from game.engine.combat import CombatState, load_enemies_json, load_sigils_json
from game.engine.enemies import Enemy
from game.engine.sigils import Sigil
from game.engine.evaluator import evaluate, card_value
from game.engine.cards import Card, Suit

BASE_W, BASE_H = 1100, 720
CARD_W, CARD_H = 100, 140
HAND_GAP = 12
ICON_BASE = 36
ICON_SIZE = ICON_BASE * 2
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_IMAGE_CACHE: dict[tuple[str, Optional[Tuple[int, int]]], Optional[pygame.Surface]] = {}


@dataclass
class Button:
    rect: pygame.Rect
    label: str

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, enabled: bool = True) -> None:
        bg = (230, 230, 230) if enabled else (180, 180, 180)
        pygame.draw.rect(screen, bg, self.rect, border_radius=10)
        pygame.draw.rect(screen, (30, 30, 30), self.rect, width=2, border_radius=10)
        t = font.render(self.label, True, (20, 20, 20))
        screen.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


def pick_font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "Arial Unicode MS"]:
        f = pygame.font.SysFont(name, size, bold=bold)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size, bold=bold)


def suit_color(s: Suit) -> Tuple[int, int, int]:
    if s in (Suit.B, Suit.C):
        return (190, 40, 40)
    return (20, 20, 20)


def _load_image(*parts: str, size: Optional[Tuple[int, int]] = None) -> Optional[pygame.Surface]:
    key = (str(Path(*parts)), size)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]
    p = ASSETS_DIR.joinpath(*parts)
    if not p.exists():
        _IMAGE_CACHE[key] = None
        return None
    try:
        surf = pygame.image.load(str(p)).convert_alpha()
        if size is not None:
            surf = pygame.transform.smoothscale(surf, size)
        _IMAGE_CACHE[key] = surf
        return surf
    except Exception:
        _IMAGE_CACHE[key] = None
        return None


def _fit_surface_keep_ratio(surf: pygame.Surface, box_size: Tuple[int, int]) -> pygame.Surface:
    bw, bh = box_size
    w, h = surf.get_size()
    if w <= 0 or h <= 0:
        return surf
    scale = min(bw / w, bh / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    if (nw, nh) == (w, h):
        return surf
    return pygame.transform.smoothscale(surf, (nw, nh))


def _blit_text_outline(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    fg: Tuple[int, int, int] = (0, 0, 0),
    outline: Tuple[int, int, int] = (255, 255, 255),
    center: bool = False,
    extra_bold: bool = False,
) -> pygame.Rect:
    base = font.render(text, True, fg)
    rect = base.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    offsets = [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)]
    for ox, oy in offsets:
        outline_surf = font.render(text, True, outline)
        orect = outline_surf.get_rect()
        if center:
            orect.center = (pos[0] + ox, pos[1] + oy)
        else:
            orect.topleft = (pos[0] + ox, pos[1] + oy)
        screen.blit(outline_surf, orect)
    screen.blit(base, rect)
    if extra_bold:
        rect2 = base.get_rect()
        if center:
            rect2.center = (pos[0] + 1, pos[1])
        else:
            rect2.topleft = (pos[0] + 1, pos[1])
        screen.blit(base, rect2)
    return rect


def _draw_glow(screen: pygame.Surface, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], alpha: int) -> None:
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*color, alpha), (radius, radius), radius)
    screen.blit(surf, (center[0] - radius, center[1] - radius))


def draw_suit(screen: pygame.Surface, suit: Suit, center: Tuple[int, int], size: int) -> None:
    cx, cy = center
    col = suit_color(suit)
    if suit == Suit.C:
        pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
        pygame.draw.polygon(screen, col, pts)
    elif suit == Suit.B:
        r = size * 0.55
        pygame.draw.circle(screen, col, (int(cx - r), int(cy - r * 0.35)), int(r))
        pygame.draw.circle(screen, col, (int(cx + r), int(cy - r * 0.35)), int(r))
        pts = [(cx - size, cy), (cx + size, cy), (cx, cy + int(size * 1.35))]
        pygame.draw.polygon(screen, col, pts)
    elif suit == Suit.A:
        r = size * 0.55
        pygame.draw.circle(screen, col, (int(cx - r), int(cy + r * 0.25)), int(r))
        pygame.draw.circle(screen, col, (int(cx + r), int(cy + r * 0.25)), int(r))
        pts = [(cx - size, cy), (cx + size, cy), (cx, cy - int(size * 1.5))]
        pygame.draw.polygon(screen, col, pts)
        stem = pygame.Rect(0, 0, int(size * 0.6), int(size * 1.2))
        stem.center = (cx, cy + int(size * 1.35))
        pygame.draw.rect(screen, col, stem)
    else:
        r = int(size * 0.55)
        pygame.draw.circle(screen, col, (cx, cy - r), r)
        pygame.draw.circle(screen, col, (cx - r, cy), r)
        pygame.draw.circle(screen, col, (cx + r, cy), r)
        pygame.draw.circle(screen, col, (cx, cy + r), r)
        stem = pygame.Rect(0, 0, int(size * 0.6), int(size * 1.25))
        stem.center = (cx, cy + int(size * 1.9))
        pygame.draw.rect(screen, col, stem)


def draw_card(screen: pygame.Surface, font: pygame.font.Font, card: Card, rect: pygame.Rect, selected: bool) -> None:
    face = _load_image("cards", f"{card.rank_str()}_{card.suit.name}.png", size=(rect.w, rect.h))
    if face is not None:
        screen.blit(face, rect.topleft)
        pygame.draw.rect(screen, (230, 180, 60) if selected else (30, 30, 30), rect, width=(5 if selected else 2), border_radius=12)
        return
    bg = (255, 255, 255) if not selected else (255, 245, 200)
    pygame.draw.rect(screen, bg, rect, border_radius=12)
    pygame.draw.rect(screen, (30, 30, 30), rect, width=2, border_radius=12)
    rank = card.rank_str()
    col = suit_color(card.suit)
    screen.blit(font.render(rank, True, col), (rect.x + 10, rect.y + 8))
    draw_suit(screen, card.suit, (rect.right - 24, rect.y + 26), 10)
    draw_suit(screen, card.suit, rect.center, 16)
    val = card_value(card)
    screen.blit(font.render(f"DMG {val}", True, (70, 70, 70)), (rect.x + 10, rect.bottom - 30))


def hand_layout(screen_w: int, y: int) -> List[pygame.Rect]:
    total_w = 8 * CARD_W + 7 * HAND_GAP
    x0 = max(20, (screen_w - total_w) // 2)
    return [pygame.Rect(x0 + i * (CARD_W + HAND_GAP), y, CARD_W, CARD_H) for i in range(8)]


RANK_ORDER = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12}


def sort_hand_by_value(hand: List[Card]) -> None:
    hand.sort(key=lambda c: (RANK_ORDER.get(c.rank, 99), c.suit.value))


def sort_hand_by_suit(hand: List[Card]) -> None:
    hand.sort(key=lambda c: (c.suit.value, RANK_ORDER.get(c.rank, 99)))


class MapScene:
    def __init__(self, run: RunState, font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.show_equip_panel = False
        self._equip_item_rects: List[Tuple[str, pygame.Rect]] = []
        self._btn_equip = Button(pygame.Rect(0, 0, 160, 46), "裝備紋章")
        self._btn_training = Button(pygame.Rect(0, 0, 160, 46), "進入訓練場")

    def _node_screen_pos(self, node: MapNode, rect: pygame.Rect) -> Tuple[int, int]:
        pad_x = 90
        pad_y = 90
        x = rect.x + pad_x + int(node.x * (rect.w - 2 * pad_x))
        y = rect.y + pad_y + int(node.y * (rect.h - 2 * pad_y))
        return x, y

    def _label(self, t: NodeType) -> str:
        return {NodeType.COMBAT: "戰鬥", NodeType.ELITE: "戰鬥-菁英", NodeType.CAMP: "帳篷", NodeType.SHOP: "商店", NodeType.BOSS: "BOSS戰"}[t]

    def _icon_name(self, t: NodeType) -> str:
        return {NodeType.COMBAT: "combat.png", NodeType.ELITE: "elite.png", NodeType.CAMP: "camp.png", NodeType.SHOP: "shop.png", NodeType.BOSS: "boss.png"}[t]

    def clickable_nodes(self) -> List[int]:
        g = self.run.map_graph
        cur = g.nodes[self.run.current_node_id]
        if not cur.cleared:
            return [cur.node_id]
        return list(cur.next_ids)

    def _hovered_node(self, mouse_pos: Tuple[int, int], map_rect: pygame.Rect) -> Optional[int]:
        g = self.run.map_graph
        for nid, node in g.nodes.items():
            x, y = self._node_screen_pos(node, map_rect)
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= 18 ** 2:
                return nid
        return None

    def handle_map_click(self, pos: Tuple[int, int], map_rect: pygame.Rect) -> Optional[int]:
        g = self.run.map_graph
        clickables = set(self.clickable_nodes())
        for nid, node in g.nodes.items():
            x, y = self._node_screen_pos(node, map_rect)
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 18 ** 2 and nid in clickables:
                return nid
        return None

    def handle_event(self, event: pygame.event.Event, sw: int, sh: int) -> Optional[str]:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        pos = event.pos
        self._btn_equip.rect.topleft = (sw - 350, 20)
        self._btn_training.rect.topleft = (sw - 180, 20)
        if self._btn_equip.hit(pos):
            self.show_equip_panel = not self.show_equip_panel
            return None
        if self._btn_training.hit(pos):
            return "training"
        if self.show_equip_panel:
            for sid, rr in self._equip_item_rects:
                if rr.collidepoint(pos):
                    if sid in self.run.equipped_sigils:
                        self.run.equipped_sigils.remove(sid)
                        self.msg = "已卸下紋章"
                    else:
                        if len(self.run.equipped_sigils) >= self.run.unlocked_slots:
                            self.msg = "插槽不足"
                        else:
                            self.run.equipped_sigils.append(sid)
                            self.msg = "已裝備紋章"
                    return None
        return None

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "map_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((245, 246, 250))
        _blit_text_outline(screen, self.font_big, "地圖", (30, 20))
        _blit_text_outline(screen, self.font, f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 60))

        self._btn_equip.rect.topleft = (sw - 350, 20)
        self._btn_training.rect.topleft = (sw - 180, 20)
        self._btn_equip.draw(screen, self.font, True)
        self._btn_training.draw(screen, self.font, True)

        map_rect = pygame.Rect(30, 90, sw - 60, sh - 140)
        g = self.run.map_graph
        clickables = set(self.clickable_nodes())
        hovered = self._hovered_node(pygame.mouse.get_pos(), map_rect)

        for node in g.nodes.values():
            x1, y1 = self._node_screen_pos(node, map_rect)
            for nid2 in node.next_ids:
                n2 = g.nodes[nid2]
                x2, y2 = self._node_screen_pos(n2, map_rect)
                pygame.draw.line(screen, (160, 160, 160), (x1, y1), (x2, y2), 4)

        for node in g.nodes.values():
            x, y = self._node_screen_pos(node, map_rect)
            if node.node_id in clickables:
                _draw_glow(screen, (x, y), 36, (255, 230, 120), 80)
            if hovered == node.node_id:
                _draw_glow(screen, (x, y), 52, (255, 255, 200), 120)

            icon = _load_image("map_icons", self._icon_name(node.node_type), size=(ICON_SIZE, ICON_SIZE))
            if icon is not None:
                icon = icon.copy()
                if node.node_id not in clickables and hovered != node.node_id:
                    icon.set_alpha(160)
                screen.blit(icon, icon.get_rect(center=(x, y)))
            else:
                pygame.draw.circle(screen, (90, 120, 200), (x, y), 18)
                pygame.draw.circle(screen, (30, 30, 30), (x, y), 18, 2)

            if hovered == node.node_id:
                _blit_text_outline(screen, self.font, self._label(node.node_type), (x, y - 58), center=True)
            if node.cleared:
                _blit_text_outline(screen, self.font, "CLEAR!", (x, y + 46), center=True)

        cur = g.nodes[self.run.current_node_id]
        hint = "點選目前節點" if not cur.cleared else "點選下一個節點"
        _blit_text_outline(screen, self.font, hint, (30, sh - 40))
        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (260, sh - 40), fg=(180, 30, 30))

        self._equip_item_rects = []
        if self.show_equip_panel:
            panel = pygame.Rect(sw - 360, 80, 330, sh - 120)
            pygame.draw.rect(screen, (250, 250, 250), panel, border_radius=16)
            pygame.draw.rect(screen, (30, 30, 30), panel, 2, border_radius=16)
            _blit_text_outline(screen, self.font_big, "裝備紋章", (panel.x + 20, panel.y + 12))
            owned = [s for s in getattr(self.run, "all_sigils_ref", []) if s.sigil_id in self.run.owned_sigils]
            if not owned:
                _blit_text_outline(screen, self.font, "未擁有任何紋章", (panel.x + 20, panel.y + 70))
            else:
                _blit_text_outline(screen, self.font, f"插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (panel.x + 20, panel.y + 70))
                y = panel.y + 110
                for s in owned:
                    r = pygame.Rect(panel.x + 16, y, panel.w - 32, 52)
                    pygame.draw.rect(screen, (255, 255, 255), r, border_radius=10)
                    pygame.draw.rect(screen, (30, 30, 30), r, 2, border_radius=10)
                    self._equip_item_rects.append((s.sigil_id, r))
                    icon = _load_image("sigils", f"{s.sigil_id}.png", size=(36, 36))
                    if icon is not None:
                        screen.blit(icon, (r.x + 8, r.y + 8))
                    mark = "已裝備" if s.sigil_id in self.run.equipped_sigils else "可裝備"
                    _blit_text_outline(screen, self.font, s.name, (r.x + 52, r.y + 8))
                    _blit_text_outline(screen, self.font, mark, (r.x + 52, r.y + 26))
                    y += 60


class ShopScene:
    def __init__(self, run: RunState, all_sigils: List[Sigil], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.all_sigils = all_sigils
        self.font = font
        self.font_big = font_big
        self.msg = ""

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "shop_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((245, 246, 250))
        _blit_text_outline(screen, self.font_big, "SHOP", (30, 20))
        _blit_text_outline(screen, self.font, f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {self.run.unlocked_slots}", (30, 60))
        panel = pygame.Rect(30, 90, sw - 60, sh - 160)
        pygame.draw.rect(screen, (255, 255, 255), panel, border_radius=18)
        pygame.draw.rect(screen, (30, 30, 30), panel, width=2, border_radius=18)

        y = panel.y + 20
        self.btn_buy_slot = Button(pygame.Rect(panel.x + 20, y, 280, 46), "Buy Slot (50G)")
        self.btn_buy_potion = Button(pygame.Rect(panel.x + 320, y, 280, 46), "Buy Potion +10HP (20G)")
        self.btn_leave = Button(pygame.Rect(panel.right - 160, panel.bottom - 60, 140, 46), "Leave")
        self.btn_buy_slot.draw(screen, self.font, enabled=(self.run.unlocked_slots < 5 and self.run.gold >= 50))
        self.btn_buy_potion.draw(screen, self.font, enabled=(self.run.gold >= 20))
        self.btn_leave.draw(screen, self.font, enabled=True)

        potion = _load_image("items", "potion.png", size=(32, 32))
        if potion is not None:
            screen.blit(potion, (panel.x + 330, y + 7))

        y += 70
        _blit_text_outline(screen, self.font_big, "Sigils", (panel.x + 20, y))
        y += 46
        self.sigil_buttons: List[Tuple[str, Button]] = []
        for s in self.all_sigils:
            owned = s.sigil_id in self.run.owned_sigils
            cost = getattr(s, "cost", 0)
            desc = getattr(s, "desc", "")
            label = f"{s.name} ({cost}G) - {'OWNED' if owned else desc}"
            b = Button(pygame.Rect(panel.x + 70, y, panel.w - 90, 40), label)
            self.sigil_buttons.append((s.sigil_id, b))
            b.draw(screen, self.font, enabled=(not owned and self.run.gold >= cost))
            icon = _load_image("sigils", f"{s.sigil_id}.png", size=(38, 38))
            if icon is not None:
                screen.blit(icon, (panel.x + 20, y + 1))
            y += 48

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (30, sh - 52), fg=(180, 30, 30))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        if self.btn_leave.hit(pos):
            return "leave"
        if self.btn_buy_slot.hit(pos):
            if self.run.unlocked_slots >= 5:
                self.msg = "插槽已滿"
            elif self.run.gold < 50:
                self.msg = "金幣不足"
            else:
                self.run.gold -= 50
                self.run.unlocked_slots += 1
                self.msg = "已購買插槽 +1"
            return None
        if self.btn_buy_potion.hit(pos):
            if self.run.gold < 20:
                self.msg = "金幣不足"
            else:
                self.run.gold -= 20
                before = self.run.hp
                self.run.hp = min(self.run.max_hp, self.run.hp + 10)
                self.msg = f"回復 {self.run.hp - before} HP"
            return None
        for sid, b in self.sigil_buttons:
            if b.hit(pos):
                s = next((x for x in self.all_sigils if x.sigil_id == sid), None)
                if s is None:
                    return None
                cost = getattr(s, "cost", 0)
                if sid in self.run.owned_sigils:
                    self.msg = "已擁有"
                elif self.run.gold < cost:
                    self.msg = "金幣不足"
                else:
                    self.run.gold -= cost
                    self.run.owned_sigils.add(sid)
                    self.msg = f"購買 {s.name}"
                return None
        return None


class CombatScene:
    def __init__(self, run: RunState, combat: CombatState, all_sigils: List[Sigil], font: pygame.font.Font, font_big: pygame.font.Font, font_huge: pygame.font.Font) -> None:
        self.run = run
        self.combat = combat
        self.all_sigils = all_sigils
        self.font = font
        self.font_big = font_big
        self.font_huge = font_huge
        self.selected: List[int] = []
        self.msg = ""

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "combat_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((245, 246, 250))

        enemy = self.combat.enemy
        if enemy is None:
            return

        portrait_box = pygame.Rect(0, 0, 360, 260)
        portrait_box.midtop = (sw // 2, 18)
        portrait = _load_image("enemies", f"{enemy.enemy_id}.png")
        if portrait is not None:
            fitted = _fit_surface_keep_ratio(portrait, (portrait_box.w, portrait_box.h))
            screen.blit(fitted, fitted.get_rect(center=portrait_box.center))
        else:
            pygame.draw.rect(screen, (220, 220, 225), portrait_box, border_radius=16)
        pygame.draw.rect(screen, (30, 30, 30), portrait_box, width=2, border_radius=16)

        hand_y = sh - CARD_H - 30
        hand_rects = hand_layout(sw, hand_y)
        btn_play = Button(pygame.Rect(30, hand_y - 70, 120, 48), "攻擊")
        btn_discard = Button(pygame.Rect(160, hand_y - 70, 120, 48), "棄牌")
        btn_sort_v = Button(pygame.Rect(290, hand_y - 70, 150, 48), "依照面值排序")
        btn_sort_s = Button(pygame.Rect(450, hand_y - 70, 140, 48), "依照花色排序")
        btn_back = Button(pygame.Rect(610, hand_y - 70, 160, 48), "返回地圖")

        over = self.combat.is_over()
        is_training = getattr(enemy, "enemy_id", "") == "training_dummy"

        _blit_text_outline(screen, self.font_big, f"Enemy: {enemy.name}", (30, 20))
        _blit_text_outline(screen, self.font, f"Enemy HP: {enemy.hp}/{enemy.max_hp}", (30, 60))
        _blit_text_outline(screen, self.font, f"ATK: {enemy.attack_damage}  Timer: {enemy.attack_timer}", (30, 88))
        _blit_text_outline(screen, self.font, f"Player HP: {self.combat.player_hp}/{self.combat.player_max_hp}", (30, 120))
        _blit_text_outline(screen, self.font, f"Gold: {getattr(self.combat, 'gold', 0)}", (30, 148))
        _blit_text_outline(screen, self.font, f"Discards Left: {self.combat.discard_uses_left}", (30, 176))
        _blit_text_outline(screen, self.font, f"Sigil Slots: {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 204))

        disabled = getattr(self.combat, "disabled_suits", {})
        if disabled:
            ds = ", ".join([f"{s.name}:{t}" for s, t in disabled.items()])
            _blit_text_outline(screen, self.font, f"Disabled suit: {ds}", (30, 232), fg=(180, 30, 30))

        btn_play.draw(screen, self.font, enabled=not over)
        btn_discard.draw(screen, self.font, enabled=(not over and self.combat.discard_uses_left > 0))
        btn_sort_v.draw(screen, self.font, enabled=True)
        btn_sort_s.draw(screen, self.font, enabled=True)
        btn_back.draw(screen, self.font, enabled=(over or is_training))

        preview = ""
        preview_red = False
        if self.selected and not over:
            if len(self.selected) > 5:
                preview = "最多可選 5 張卡牌"
                preview_red = True
            else:
                try:
                    cards = [self.combat.hand[i] for i in self.selected]
                    r = evaluate(cards)
                    preview = f"Selected {len(self.selected)} | {r.attack_type.label()} | Base {r.base_damage} + Points {r.point_sum} = {r.damage}"
                    if getattr(r, "used_highest_single", False):
                        preview += " (fallback: highest single)"
                except Exception as e:
                    preview = str(e)
                    preview_red = True
        if preview:
            _blit_text_outline(screen, self.font, preview, (30, hand_y - 105), fg=((180, 30, 30) if preview_red else (0, 0, 0)))

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (30, hand_y - 130), fg=(180, 30, 30))

        for i, c in enumerate(self.combat.hand[:8]):
            draw_card(screen, self.font, c, hand_rects[i], selected=(i in self.selected))

        log_x = 30
        log_y = 270
        _blit_text_outline(screen, self.font_big, "Log", (log_x, log_y))
        log_y += 36
        for line in self.combat.log[-4:]:
            color = (180, 30, 30) if "攻擊你，造成" in line or "雙倍" in line else (0, 0, 0)
            _blit_text_outline(screen, self.font, line, (log_x, log_y), fg=color)
            log_y += 28

        if self.combat.player_hp <= 0:
            _blit_text_outline(screen, self.font_huge, "DEFEAT", (sw // 2, 170), fg=(190, 40, 40), center=True, extra_bold=True)
        elif enemy.hp <= 0 and not is_training:
            _blit_text_outline(screen, self.font_huge, "VICTORY", (sw // 2, 170), fg=(40, 140, 60), center=True, extra_bold=True)

        self._btn_play = btn_play
        self._btn_discard = btn_discard
        self._btn_sort_v = btn_sort_v
        self._btn_sort_s = btn_sort_s
        self._btn_back = btn_back
        self._hand_rects = hand_rects

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        pos = event.pos
        enemy = self.combat.enemy
        is_training = enemy is not None and getattr(enemy, "enemy_id", "") == "training_dummy"
        if self._btn_sort_v.hit(pos):
            sort_hand_by_value(self.combat.hand)
            self.selected = []
            self.msg = "已依面值排序"
            return None
        if self._btn_sort_s.hit(pos):
            sort_hand_by_suit(self.combat.hand)
            self.selected = []
            self.msg = "已依花色排序"
            return None
        if self._btn_back.hit(pos):
            if self.combat.is_over() or is_training:
                return "back_to_map"
            self.msg = "戰鬥尚未結束"
            return None

        over = self.combat.is_over()
        if self._btn_play.hit(pos):
            if over and not is_training:
                self.msg = "戰鬥結束"
                return None
            if len(self.selected) > 5:
                self.msg = "最多可選 5 張卡牌"
                return None
            try:
                self.combat.play(self.selected)
                self.selected = []
                self.msg = ""
                if is_training and enemy is not None and enemy.hp <= 0:
                    enemy.hp = enemy.max_hp
                    self.combat.log.append(f"{enemy.name} 恢復為 {enemy.max_hp} 點生命值。")
            except Exception as e:
                self.msg = str(e)
            return None

        if self._btn_discard.hit(pos):
            if over and not is_training:
                self.msg = "戰鬥結束"
                return None
            try:
                self.combat.discard(self.selected)
                self.selected = []
                self.msg = ""
            except Exception as e:
                self.msg = str(e)
            return None

        for i, rr in enumerate(self._hand_rects):
            if rr.collidepoint(pos):
                if i in self.selected:
                    self.selected.remove(i)
                else:
                    self.selected.append(i)
                self.selected.sort()
                return None
        return None


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Claw of Demon")

    info = pygame.display.Info()
    w = min(BASE_W, max(980, info.current_w - 120))
    h = min(BASE_H, max(640, info.current_h - 120))
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    font = pick_font(20, bold=True)
    font_big = pick_font(30, bold=True)
    font_huge = pick_font(72, bold=True)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    enemies_all = load_enemies_json(data_dir / "enemies.json")
    sigils_all = load_sigils_json(data_dir / "sigils.json")

    def tier_of(e: Enemy) -> str:
        return str(getattr(e, "tier", "normal"))

    normals = [e for e in enemies_all if tier_of(e) == "normal" and getattr(e, "enemy_id", "") != "training_dummy"]
    elites = [e for e in enemies_all if tier_of(e) == "elite"]
    bosses = [e for e in enemies_all if tier_of(e) == "boss"]
    training_dummy = next((e for e in enemies_all if getattr(e, "enemy_id", "") == "training_dummy"), None)
    if training_dummy is None:
        training_dummy = next((e for e in enemies_all if "訓練" in getattr(e, "name", "")), None)

    run = RunState()
    if not hasattr(run, "owned_sigils"):
        run.owned_sigils = set()
    if not hasattr(run, "equipped_sigils"):
        run.equipped_sigils = []
    if not hasattr(run, "unlocked_slots"):
        run.unlocked_slots = 1
    run.all_sigils_ref = sigils_all
    if not hasattr(run, "current_node_id"):
        run.current_node_id = run.map_graph.start_id

    fullscreen = False
    windowed_size = (w, h)

    scene = "map"
    map_scene = MapScene(run, font, font_big)
    shop_scene: Optional[ShopScene] = None
    combat_scene: Optional[CombatScene] = None

    def start_combat(enemy: Enemy) -> None:
        nonlocal combat_scene, scene
        c = CombatState()
        c.player_max_hp = run.max_hp
        c.player_hp = run.hp
        if hasattr(c, "gold"):
            c.gold = run.gold
        c.start(enemy)
        c.sigils = [s for s in sigils_all if s.sigil_id in run.equipped_sigils]
        combat_scene = CombatScene(run, c, sigils_all, font, font_big, font_huge)
        scene = "combat"

    def finish_combat_and_return() -> None:
        nonlocal scene, combat_scene
        if combat_scene is None:
            return
        enemy = combat_scene.combat.enemy
        run.hp = combat_scene.combat.player_hp
        if hasattr(combat_scene.combat, "gold"):
            run.gold = combat_scene.combat.gold
        node = run.map_graph.nodes[run.current_node_id]
        is_training = enemy is not None and getattr(enemy, "enemy_id", "") == "training_dummy"
        if (not is_training) and enemy and enemy.hp <= 0:
            node.cleared = True
            if node.node_type == NodeType.COMBAT:
                run.gold += 20
            elif node.node_type == NodeType.ELITE:
                run.gold += 40
            elif node.node_type == NodeType.BOSS:
                run.gold += 100
        combat_scene = None
        scene = "map"

    running = True
    while running:
        sw, sh = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                if not fullscreen:
                    windowed_size = screen.get_size()
                    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
                    fullscreen = True
                else:
                    screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
                    fullscreen = False
            if event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            if scene == "map":
                action = map_scene.handle_event(event, sw, sh)
                if action == "training" and training_dummy is not None:
                    start_combat(training_dummy)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not map_scene.show_equip_panel:
                    map_rect = pygame.Rect(30, 90, sw - 60, sh - 140)
                    nid = map_scene.handle_map_click(event.pos, map_rect)
                    if nid is None:
                        continue
                    run.current_node_id = nid
                    node = run.map_graph.nodes[nid]
                    if node.node_type == NodeType.CAMP:
                        before = run.hp
                        run.hp = min(run.max_hp, run.hp + 20)
                        node.cleared = True
                        map_scene.msg = f"CAMP 回復 {run.hp - before} HP"
                    elif node.node_type == NodeType.SHOP:
                        shop_scene = ShopScene(run, sigils_all, font, font_big)
                        scene = "shop"
                    elif node.node_type == NodeType.COMBAT and normals:
                        start_combat(choice(normals))
                    elif node.node_type == NodeType.ELITE and elites:
                        start_combat(choice(elites))
                    elif node.node_type == NodeType.BOSS and bosses:
                        start_combat(choice(bosses))

            elif scene == "shop":
                if shop_scene is None:
                    shop_scene = ShopScene(run, sigils_all, font, font_big)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    res = shop_scene.handle_click(event.pos)
                    if res == "leave":
                        run.map_graph.nodes[run.current_node_id].cleared = True
                        shop_scene = None
                        scene = "map"

            elif scene == "combat":
                if combat_scene is None:
                    continue
                action = combat_scene.handle_event(event)
                if action == "back_to_map":
                    finish_combat_and_return()

        if scene == "map":
            map_scene.render(screen, sw, sh)
        elif scene == "shop":
            if shop_scene is None:
                shop_scene = ShopScene(run, sigils_all, font, font_big)
            shop_scene.render(screen, sw, sh)
        elif scene == "combat" and combat_scene is not None:
            combat_scene.render(screen, sw, sh)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
