from __future__ import annotations

import sys
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from random import choice, random, sample

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


def _blit_text_soft_outline(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    fg: Tuple[int, int, int] = (238, 166, 46),
    outline: Tuple[int, int, int] = (10, 10, 10),
    outline_alpha: int = 150,
    center: bool = False,
) -> pygame.Rect:
    base = font.render(text, True, fg)
    rect = base.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos

    outline_surf = font.render(text, True, outline)
    outline_surf.set_alpha(max(0, min(255, outline_alpha)))
    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        orect = outline_surf.get_rect()
        if center:
            orect.center = (pos[0] + ox, pos[1] + oy)
        else:
            orect.topleft = (pos[0] + ox, pos[1] + oy)
        screen.blit(outline_surf, orect)
    screen.blit(base, rect)
    return rect


def _draw_map_hp_bar(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    hp: int,
    max_hp: int,
) -> None:
    max_hp = max(1, int(max_hp))
    hp = max(0, min(int(hp), max_hp))
    inner = rect.inflate(-4, -4)
    ratio = hp / max_hp
    fill_w = int(round(inner.w * ratio))

    shadow = pygame.Surface((rect.w + 8, rect.h + 8), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 105), shadow.get_rect(), border_radius=10)
    screen.blit(shadow, (rect.x - 4, rect.y - 4))

    pygame.draw.rect(screen, (30, 30, 36), rect, border_radius=9)
    pygame.draw.rect(screen, (75, 45, 45), inner, border_radius=7)
    if fill_w > 0:
        fill = pygame.Rect(inner.x, inner.y, fill_w, inner.h)
        pygame.draw.rect(screen, (56, 165, 82), fill, border_radius=7)
    pygame.draw.rect(screen, (18, 18, 22), rect, width=2, border_radius=9)

    text = f"HP {hp}/{max_hp}"
    _blit_text_outline(screen, font, text, rect.center, fg=(245, 245, 245), outline=(20, 20, 20), center=True)


def _draw_map_gold(screen: pygame.Surface, font: pygame.font.Font, x: int, y: int, gold: int) -> None:
    icon = _load_image("items", "gold.png", size=(30, 30))
    if icon is not None:
        screen.blit(icon, (x, y))
    else:
        pygame.draw.circle(screen, (222, 157, 38), (x + 15, y + 15), 13)
        pygame.draw.circle(screen, (80, 50, 10), (x + 15, y + 15), 13, 2)
        pygame.draw.circle(screen, (255, 210, 85), (x + 11, y + 10), 4)

    _blit_text_soft_outline(
        screen,
        font,
        str(int(gold)),
        (x + 38, y + 3),
        fg=(246, 174, 48),
        outline=(5, 5, 5),
        outline_alpha=165,
    )


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


def draw_card(screen: pygame.Surface, font: pygame.font.Font, card: Card, rect: pygame.Rect, selected: bool, awakened: bool = False) -> None:
    face = _load_image("cards", f"{card.rank_str()}_{card.suit.name}.png", size=(rect.w, rect.h))
    if face is not None:
        screen.blit(face, rect.topleft)
        pygame.draw.rect(screen, (230, 180, 60) if selected else (30, 30, 30), rect, width=(5 if selected else 2), border_radius=12)
        if awakened:
            pygame.draw.rect(screen, (150, 70, 220), rect.inflate(8, 8), width=2, border_radius=15)
        return
    bg = (255, 255, 255) if not selected else (255, 245, 200)
    pygame.draw.rect(screen, bg, rect, border_radius=12)
    pygame.draw.rect(screen, (30, 30, 30), rect, width=2, border_radius=12)
    if awakened:
        pygame.draw.rect(screen, (150, 70, 220), rect.inflate(8, 8), width=2, border_radius=15)
    rank = card.rank_str()
    col = suit_color(card.suit)
    screen.blit(font.render(rank, True, col), (rect.x + 10, rect.y + 8))
    draw_suit(screen, card.suit, (rect.right - 24, rect.y + 26), 10)
    draw_suit(screen, card.suit, rect.center, 16)
    val = card_value(card)
    screen.blit(font.render(f"DMG {val}", True, (70, 70, 70)), (rect.x + 10, rect.bottom - 30))


def hand_layout(screen_w: int, y: int, count: int = 8) -> List[pygame.Rect]:
    count = max(1, count)
    available_w = max(360, screen_w - 40)
    card_w = min(CARD_W, max(64, int((available_w - (count - 1) * HAND_GAP) / count)))
    card_h = max(90, int(CARD_H * card_w / CARD_W))
    total_w = count * card_w + (count - 1) * HAND_GAP
    x0 = max(20, (screen_w - total_w) // 2)
    return [pygame.Rect(x0 + i * (card_w + HAND_GAP), y + (CARD_H - card_h), card_w, card_h) for i in range(count)]


RANK_ORDER = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12}


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ease_out_cubic(t: float) -> float:
    t = clamp_float(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


DAMAGE_FX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "physical": (220, 40, 40),
    "poison": (125, 70, 185),
    "burn": (240, 115, 35),
    "frost": (90, 185, 240),
    "lightning": (245, 220, 80),
    "ritual": (155, 50, 170),
}


def sort_hand_by_value(hand: List[Card]) -> None:
    hand.sort(key=lambda c: (RANK_ORDER.get(c.rank, 99), c.suit.value))


def sort_hand_by_suit(hand: List[Card]) -> None:
    hand.sort(key=lambda c: (c.suit.value, RANK_ORDER.get(c.rank, 99)))


def load_events_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, dict):
        events = data.get("events", [])
    else:
        events = data

    if not isinstance(events, list):
        return []

    return [event for event in events if isinstance(event, dict)]


def load_start_events_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, dict):
        events = data.get("start_events", data.get("options", []))
    else:
        events = data

    if not isinstance(events, list):
        return []

    return [event for event in events if isinstance(event, dict)]


def _wrap_text(font: pygame.font.Font, text: str, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in str(text).split("\n"):
        line = ""
        for ch in paragraph:
            if not line:
                line = ch
            elif font.size(line + ch)[0] <= max_width:
                line += ch
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
        elif not paragraph:
            lines.append("")
    return lines


def _requirement_met(run: RunState, requirement: Dict[str, Any]) -> bool:
    req_type = str(requirement.get("type", ""))
    amount = int(requirement.get("amount", 0))

    if req_type in ("min_gold", "gold_at_least"):
        return run.gold >= amount
    if req_type in ("hp_more_than", "hp_gt"):
        return run.hp > amount
    if req_type in ("max_hp_more_than", "max_hp_gt"):
        return run.max_hp > amount
    if req_type == "has_owned_sigil":
        return len(run.owned_sigils) > 0
    return True


def _option_enabled(run: RunState, option: Dict[str, Any]) -> bool:
    for requirement in option.get("requires", []):
        if isinstance(requirement, dict) and not _requirement_met(run, requirement):
            return False

    for effect in option.get("effects", []):
        if not isinstance(effect, dict):
            continue
        effect_type = str(effect.get("type", ""))
        amount = int(effect.get("amount", 0))
        if effect_type == "lose_gold" and run.gold < amount:
            return False
        if effect_type == "lose_hp" and run.hp <= amount:
            return False
        if effect_type == "lose_max_hp" and run.max_hp <= amount + 1:
            return False
    return True


class MapScene:
    def __init__(self, run: RunState, font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.show_equip_panel = False
        self._equip_item_rects: List[Tuple[str, pygame.Rect]] = []
        self._equip_icon_rects: List[Tuple[Sigil, pygame.Rect]] = []
        self.scroll_y = 0
        self.scroll_speed = 70
        self._initial_scroll_done = False
        self._btn_equip = Button(pygame.Rect(0, 0, 160, 46), "裝備紋章")
        self._btn_training = Button(pygame.Rect(0, 0, 160, 46), "進入訓練場")

    def _map_view_rect(self, sw: int, sh: int) -> pygame.Rect:
        return pygame.Rect(30, 90, sw - 20, sh - 140)

    def _map_content_height(self, view_rect: pygame.Rect) -> int:
        g = self.run.map_graph
        if not g.nodes:
            return view_rect.h

        max_depth = max(node.depth for node in g.nodes.values())
        floor_count = max_depth + 1
        return max(view_rect.h, floor_count * 105 + 240)

    def _max_scroll_y(self, view_rect: pygame.Rect) -> int:
        return max(0, self._map_content_height(view_rect) - view_rect.h)

    def _clamp_scroll(self, view_rect: pygame.Rect) -> None:
        self.scroll_y = max(0, min(self.scroll_y, self._max_scroll_y(view_rect)))

    def _map_content_rect(self, sw: int, sh: int) -> pygame.Rect:
        view_rect = self._map_view_rect(sw, sh)
        self._clamp_scroll(view_rect)
        return pygame.Rect(
            view_rect.x,
            view_rect.y - self.scroll_y,
            view_rect.w,
            self._map_content_height(view_rect),
        )

    def _node_screen_pos(self, node: MapNode, rect: pygame.Rect) -> Tuple[int, int]:
        pad_x = 90
        pad_y = 90
        x = rect.x + pad_x + int(node.x * (rect.w - 2 * pad_x))
        y = rect.y + pad_y + int(node.y * (rect.h - 2 * pad_y))
        return x, y

    def _label(self, t: NodeType) -> str:
        labels = {
            "START": "開始",
            "COMBAT": "戰鬥",
            "UNKNOWN": "未知",
            "ELITE": "戰鬥-菁英",
            "CAMP": "休息點",
            "SHOP": "商店",
            "TREASURE": "寶箱",
            "BOSS": "BOSS戰",
        }
        return labels.get(t.name, t.name)

    def _icon_name(self, t: NodeType) -> str:
        icons = {
            "START": "start.png",
            "COMBAT": "combat.png",
            "UNKNOWN": "unknown.png",
            "ELITE": "elite.png",
            "CAMP": "camp.png",
            "SHOP": "shop.png",
            "TREASURE": "treasure.png",
            "BOSS": "boss.png",
        }
        return icons.get(t.name, "combat.png")

    def clickable_nodes(self) -> List[int]:
        g = self.run.map_graph
        cur = g.nodes[self.run.current_node_id]
        if not cur.cleared:
            return [cur.node_id]
        return list(cur.next_ids)

    def _hovered_node(self, mouse_pos: Tuple[int, int], content_rect: pygame.Rect, view_rect: pygame.Rect) -> Optional[int]:
        if not view_rect.collidepoint(mouse_pos):
            return None

        g = self.run.map_graph
        for nid, node in g.nodes.items():
            x, y = self._node_screen_pos(node, content_rect)
            if not view_rect.collidepoint((x, y)):
                continue
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= 18 ** 2:
                return nid
        return None

    def handle_map_click(self, pos: Tuple[int, int], sw: int, sh: int) -> Optional[int]:
        view_rect = self._map_view_rect(sw, sh)
        if not view_rect.collidepoint(pos):
            return None

        content_rect = self._map_content_rect(sw, sh)
        g = self.run.map_graph
        clickables = set(self.clickable_nodes())

        for nid, node in g.nodes.items():
            x, y = self._node_screen_pos(node, content_rect)
            if not view_rect.collidepoint((x, y)):
                continue
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 18 ** 2 and nid in clickables:
                return nid
        return None

    def _equip_panel_rect(self, sw: int, sh: int) -> pygame.Rect:
        return pygame.Rect(sw - 360, 80, 330, sh - 120)

    def _draw_sigil_tooltip(
        self,
        screen: pygame.Surface,
        sw: int,
        sh: int,
        mouse_pos: Tuple[int, int],
        sigil: Sigil,
    ) -> None:
        title = str(getattr(sigil, "name", "未知紋章"))
        desc = str(getattr(sigil, "desc", "")).strip()
        if not desc:
            desc = "沒有說明。"

        max_text_w = 270
        lines = [title] + _wrap_text(self.font, desc, max_text_w)
        padding_x = 12
        padding_y = 10
        line_h = 24
        tooltip_w = min(320, max(160, max(self.font.size(line)[0] for line in lines) + padding_x * 2))
        tooltip_h = padding_y * 2 + len(lines) * line_h

        x = mouse_pos[0] + 18
        y = mouse_pos[1] + 18
        if x + tooltip_w > sw - 8:
            x = mouse_pos[0] - tooltip_w - 18
        if y + tooltip_h > sh - 8:
            y = sh - tooltip_h - 8
        x = max(8, x)
        y = max(8, y)

        tooltip = pygame.Surface((tooltip_w, tooltip_h), pygame.SRCALPHA)
        rect = tooltip.get_rect()
        pygame.draw.rect(tooltip, (0, 0, 0, 230), rect, border_radius=8)
        pygame.draw.rect(tooltip, (255, 255, 255, 80), rect, width=1, border_radius=8)
        screen.blit(tooltip, (x, y))

        text_y = y + padding_y
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (245, 245, 245))
            screen.blit(surf, (x + padding_x, text_y))
            text_y += line_h
            if i == 0 and len(lines) > 1:
                pygame.draw.line(
                    screen,
                    (245, 245, 245),
                    (x + padding_x, text_y - 3),
                    (x + tooltip_w - padding_x, text_y - 3),
                    1,
                )

    def handle_event(self, event: pygame.event.Event, sw: int, sh: int) -> Optional[str]:
        if event.type == pygame.MOUSEWHEEL:
            view_rect = self._map_view_rect(sw, sh)
            if not self.show_equip_panel and view_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_y -= event.y * self.scroll_speed
                self._clamp_scroll(view_rect)
            return None

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        pos = event.pos
        self._btn_equip.rect.topleft = (sw - 350, 20)
        self._btn_training.rect.topleft = (sw - 180, 20)
        if self._btn_equip.hit(pos):
            self.show_equip_panel = not self.show_equip_panel
            return "consume"
        if self._btn_training.hit(pos):
            return "training"
        if self.show_equip_panel:
            panel = self._equip_panel_rect(sw, sh)
            if not panel.collidepoint(pos):
                self.show_equip_panel = False
                return "consume"

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
                    return "consume"
            return "consume"
        return None

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        screen.fill((24, 24, 28))
        ui_bg = _load_image("backgrounds", "map_frame_bg.png", size=(sw, sh))
        if ui_bg is not None:
            screen.blit(ui_bg, (0, 0))

        _blit_text_outline(screen, self.font_big, "地圖", (30, 20), fg=(245, 245, 245), outline=(20, 20, 20))
        _draw_map_hp_bar(screen, self.font, pygame.Rect(30, 58, 240, 28), self.run.hp, self.run.max_hp)
        _draw_map_gold(screen, self.font, 300, 56, self.run.gold)
        _blit_text_outline(
            screen,
            self.font,
            f"紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}",
            (410, 60),
            fg=(245, 245, 245),
            outline=(20, 20, 20),
        )

        self._btn_equip.rect.topleft = (sw - 350, 20)
        self._btn_training.rect.topleft = (sw - 180, 20)
        self._btn_equip.draw(screen, self.font, True)
        self._btn_training.draw(screen, self.font, True)

        view_rect = self._map_view_rect(sw, sh)
        if not self._initial_scroll_done:
            self.scroll_y = self._max_scroll_y(view_rect)
            self._initial_scroll_done = True

        content_rect = self._map_content_rect(sw, sh)
        content_h = self._map_content_height(view_rect)
        g = self.run.map_graph
        clickables = set(self.clickable_nodes())
        hovered = self._hovered_node(pygame.mouse.get_pos(), content_rect, view_rect)

        old_clip = screen.get_clip()
        screen.set_clip(view_rect)

        map_bg = _load_image("backgrounds", "map_bg.png", size=(view_rect.w, content_h))
        if map_bg is not None:
            screen.blit(map_bg, (view_rect.x, view_rect.y - self.scroll_y))
        else:
            pygame.draw.rect(screen, (245, 246, 250), view_rect, border_radius=18)

        for node in g.nodes.values():
            x1, y1 = self._node_screen_pos(node, content_rect)
            for nid2 in node.next_ids:
                n2 = g.nodes[nid2]
                x2, y2 = self._node_screen_pos(n2, content_rect)
                pygame.draw.line(screen, (35, 35, 35), (x1, y1), (x2, y2), 7)
                pygame.draw.line(screen, (225, 225, 215), (x1, y1), (x2, y2), 4)

        for node in g.nodes.values():
            x, y = self._node_screen_pos(node, content_rect)
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
                fallback_colors = {
                    "START": (80, 130, 200),
                    "COMBAT": (90, 120, 200),
                    "UNKNOWN": (150, 90, 190),
                    "ELITE": (190, 80, 70),
                    "CAMP": (80, 160, 100),
                    "SHOP": (210, 160, 60),
                    "TREASURE": (220, 185, 60),
                    "BOSS": (90, 40, 40),
                }
                pygame.draw.circle(screen, fallback_colors.get(node.node_type.name, (90, 120, 200)), (x, y), 18)
                pygame.draw.circle(screen, (30, 30, 30), (x, y), 18, 2)

            if hovered == node.node_id:
                _blit_text_outline(screen, self.font, self._label(node.node_type), (x, y - 58), center=True)
            if node.cleared:
                _blit_text_outline(screen, self.font, "CLEAR!", (x, y + 46), center=True)

        screen.set_clip(old_clip)

        pygame.draw.rect(screen, (20, 20, 22), view_rect, width=2, border_radius=18)

        max_scroll = self._max_scroll_y(view_rect)
        if max_scroll > 0:
            track = pygame.Rect(view_rect.right - 14, view_rect.y + 18, 8, view_rect.h - 36)
            pygame.draw.rect(screen, (210, 210, 210), track, border_radius=4)
            thumb_h = max(36, int(track.h * view_rect.h / self._map_content_height(view_rect)))
            thumb_y = track.y + int((track.h - thumb_h) * (self.scroll_y / max_scroll))
            pygame.draw.rect(screen, (90, 90, 90), pygame.Rect(track.x, thumb_y, track.w, thumb_h), border_radius=4)

        cur = g.nodes[self.run.current_node_id]
        hint = "滑鼠滾輪捲動地圖；點選目前節點" if not cur.cleared else "滑鼠滾輪捲動地圖；點選下一個節點"
        _blit_text_outline(screen, self.font, hint, (30, sh - 40))
        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (260, sh - 40), fg=(180, 30, 30))

        self._equip_item_rects = []
        self._equip_icon_rects = []
        if self.show_equip_panel:
            panel = self._equip_panel_rect(sw, sh)
            pygame.draw.rect(screen, (250, 250, 250), panel, border_radius=16)
            pygame.draw.rect(screen, (30, 30, 30), panel, 2, border_radius=16)
            _blit_text_outline(screen, self.font_big, "裝備紋章", (panel.x + 20, panel.y + 12))
            owned = [s for s in getattr(self.run, "all_sigils_ref", []) if s.sigil_id in self.run.owned_sigils]
            hovered_sigil: Optional[Sigil] = None
            mouse_pos = pygame.mouse.get_pos()

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

                    icon_rect = pygame.Rect(r.x + 8, r.y + 8, 36, 36)
                    self._equip_icon_rects.append((s, icon_rect))
                    icon = _load_image("sigils", f"{s.sigil_id}.png", size=(36, 36))
                    if icon is not None:
                        screen.blit(icon, icon_rect.topleft)
                    else:
                        pygame.draw.circle(screen, (130, 90, 180), icon_rect.center, 17)
                        pygame.draw.circle(screen, (30, 30, 30), icon_rect.center, 17, 2)

                    if icon_rect.collidepoint(mouse_pos):
                        hovered_sigil = s

                    mark = "已裝備" if s.sigil_id in self.run.equipped_sigils else "可裝備"
                    _blit_text_outline(screen, self.font, s.name, (r.x + 52, r.y + 8))
                    _blit_text_outline(screen, self.font, mark, (r.x + 52, r.y + 26))
                    y += 60

            if hovered_sigil is not None:
                self._draw_sigil_tooltip(screen, sw, sh, mouse_pos, hovered_sigil)


class ShopScene:
    def __init__(self, run: RunState, all_sigils: List[Sigil], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.all_sigils = all_sigils
        available_sigils = [s for s in all_sigils if s.sigil_id not in self.run.owned_sigils]
        self.shop_sigils = sample(available_sigils, min(4, len(available_sigils))) if available_sigils else []
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
        if not self.shop_sigils:
            _blit_text_outline(screen, self.font, "目前沒有可購買的紋章", (panel.x + 20, y), fg=(90, 90, 90))
        for s in self.shop_sigils:
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
                s = next((x for x in self.shop_sigils if x.sigil_id == sid), None)
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


class StartScene:
    def __init__(self, run: RunState, options: List[Dict[str, Any]], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.options = options
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.option_buttons: List[Tuple[Dict[str, Any], Button, bool]] = []

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "start_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((232, 226, 216))

        _blit_text_outline(screen, self.font_big, "起始房間", (30, 20))
        _blit_text_outline(screen, self.font, f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 60))

        panel = pygame.Rect(60, 100, sw - 120, sh - 150)
        pygame.draw.rect(screen, (255, 255, 255), panel, border_radius=18)
        pygame.draw.rect(screen, (30, 30, 30), panel, width=2, border_radius=18)

        _blit_text_outline(screen, self.font_big, "惡魔之爪的低語", (panel.x + 28, panel.y + 24))

        description = "一道漆黑的爪痕浮現在地面上。它沒有命令你前進，只是靜靜地提出交易。選擇一項祝福，然後開始這趟旅程。"
        y = panel.y + 76
        for line in _wrap_text(self.font, description, panel.w - 56):
            _blit_text_outline(screen, self.font, line, (panel.x + 28, y))
            y += 28

        y += 18
        self.option_buttons = []
        options = [opt for opt in self.options if isinstance(opt, dict)]
        if not options:
            options = [{"label": "踏上旅程", "text": "無事發生。", "effects": []}]

        for idx, option in enumerate(options, start=1):
            enabled = _option_enabled(self.run, option)
            r = pygame.Rect(panel.x + 28, y, panel.w - 56, 82)
            label = str(option.get("label", f"選項 {idx}"))
            text = str(option.get("text", ""))
            b = Button(r, f"{idx}. {label}")
            b.draw(screen, self.font, enabled=enabled)

            text_color = (50, 50, 50) if enabled else (130, 130, 130)
            text_lines = _wrap_text(self.font, text, r.w - 28)
            for line_i, line in enumerate(text_lines[:2]):
                _blit_text_outline(screen, self.font, line, (r.x + 14, r.y + 42 + line_i * 22), fg=text_color)

            self.option_buttons.append((option, b, enabled))
            y += 96

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (panel.x + 28, panel.bottom - 42), fg=(180, 30, 30))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        for option, button, enabled in self.option_buttons:
            if button.hit(pos):
                if not enabled:
                    self.msg = "目前條件不足，無法選擇這個選項。"
                    return None
                return option
        return None


class EventScene:
    def __init__(self, run: RunState, event_data: Dict[str, Any], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.event_data = event_data
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.option_buttons: List[Tuple[Dict[str, Any], Button, bool]] = []

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "event_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((238, 232, 222))

        _blit_text_outline(screen, self.font_big, "未知房間", (30, 20))
        _blit_text_outline(screen, self.font, f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 60))

        panel = pygame.Rect(60, 100, sw - 120, sh - 150)
        pygame.draw.rect(screen, (255, 255, 255), panel, border_radius=18)
        pygame.draw.rect(screen, (30, 30, 30), panel, width=2, border_radius=18)

        title = str(self.event_data.get("title", "未知事件"))
        description = str(self.event_data.get("description", "你遇見了一件無法解釋的事。"))

        _blit_text_outline(screen, self.font_big, title, (panel.x + 28, panel.y + 24))

        y = panel.y + 76
        for line in _wrap_text(self.font, description, panel.w - 56):
            _blit_text_outline(screen, self.font, line, (panel.x + 28, y))
            y += 28

        y += 18
        self.option_buttons = []
        options = [opt for opt in self.event_data.get("options", []) if isinstance(opt, dict)]
        if not options:
            options = [{"label": "離開", "text": "無事發生。", "effects": []}]

        for idx, option in enumerate(options, start=1):
            enabled = _option_enabled(self.run, option)
            r = pygame.Rect(panel.x + 28, y, panel.w - 56, 76)
            label = str(option.get("label", f"選項 {idx}"))
            text = str(option.get("text", ""))
            b = Button(r, f"{idx}. {label}")
            b.draw(screen, self.font, enabled=enabled)
            text_color = (50, 50, 50) if enabled else (130, 130, 130)
            text_lines = _wrap_text(self.font, text, r.w - 28)
            if text_lines:
                _blit_text_outline(screen, self.font, text_lines[0], (r.x + 14, r.y + 42), fg=text_color)
            self.option_buttons.append((option, b, enabled))
            y += 90

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (panel.x + 28, panel.bottom - 42), fg=(180, 30, 30))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        for option, button, enabled in self.option_buttons:
            if button.hit(pos):
                if not enabled:
                    self.msg = "目前條件不足，無法選擇這個選項。"
                    return None
                return option
        return None


class SigilChoiceScene:
    def __init__(
        self,
        run: RunState,
        choices: List[Sigil],
        title: str,
        description: str,
        font: pygame.font.Font,
        font_big: pygame.font.Font,
    ) -> None:
        self.run = run
        self.choices = choices
        self.title = title
        self.description = description
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.sigil_buttons: List[Tuple[Sigil, pygame.Rect]] = []

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "event_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((238, 232, 222))

        _blit_text_outline(screen, self.font_big, self.title, (30, 20))
        _blit_text_outline(screen, self.font, f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 60))

        panel = pygame.Rect(60, 100, sw - 120, sh - 150)
        pygame.draw.rect(screen, (255, 255, 255), panel, border_radius=18)
        pygame.draw.rect(screen, (30, 30, 30), panel, width=2, border_radius=18)

        _blit_text_outline(screen, self.font_big, "選擇一枚紋章", (panel.x + 28, panel.y + 24))

        y = panel.y + 76
        for line in _wrap_text(self.font, self.description, panel.w - 56):
            _blit_text_outline(screen, self.font, line, (panel.x + 28, y))
            y += 28

        y += 18
        self.sigil_buttons = []

        if not self.choices:
            _blit_text_outline(screen, self.font, "目前沒有可選擇的新紋章。", (panel.x + 28, y), fg=(130, 130, 130))
        else:
            for idx, sigil in enumerate(self.choices, start=1):
                r = pygame.Rect(panel.x + 28, y, panel.w - 56, 92)
                pygame.draw.rect(screen, (248, 248, 252), r, border_radius=14)
                pygame.draw.rect(screen, (30, 30, 30), r, width=2, border_radius=14)
                self.sigil_buttons.append((sigil, r))

                icon = _load_image("sigils", f"{sigil.sigil_id}.png", size=(48, 48))
                if icon is not None:
                    screen.blit(icon, (r.x + 14, r.y + 20))
                else:
                    pygame.draw.circle(screen, (130, 90, 180), (r.x + 38, r.y + 44), 22)
                    pygame.draw.circle(screen, (30, 30, 30), (r.x + 38, r.y + 44), 22, 2)

                _blit_text_outline(screen, self.font_big, f"{idx}. {sigil.name}", (r.x + 76, r.y + 10))
                desc = str(getattr(sigil, "desc", ""))
                for line_i, line in enumerate(_wrap_text(self.font, desc, r.w - 100)[:2]):
                    _blit_text_outline(screen, self.font, line, (r.x + 76, r.y + 48 + line_i * 22), fg=(55, 55, 55))
                y += 106

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (panel.x + 28, panel.bottom - 42), fg=(180, 30, 30))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Sigil]:
        for sigil, rect in self.sigil_buttons:
            if rect.collidepoint(pos):
                return sigil
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
        self.effects: List[dict] = []
        self.log_toasts: List[dict] = []
        self._seen_log_count = 0
        self.enemy_flash_ms = 0
        self.player_flash_ms = 0
        self.screen_shake_ms = 0
        self.big_hit_threshold = 15
        self._enemy_fx_center: Tuple[int, int] = (550, 150)
        self._player_fx_center: Tuple[int, int] = (170, 130)
        self.pending_enemy_turn_ms = 0
        self.enemy_turn_delay_ms = 2000
        self.input_locked = False
        self.border_flash_ms = 0
        self.border_flash_type = "physical"
        self.border_flash_duration_ms = 260
        self._seen_damage_event_count = 0
        enemy = self.combat.enemy
        enemy_hp = float(enemy.hp) if enemy is not None else 0.0
        player_hp = float(self.combat.player_hp)
        self.hp_bar_duration_ms = 600
        self.enemy_hp_from = enemy_hp
        self.enemy_hp_display = enemy_hp
        self.enemy_hp_target = enemy_hp
        self.enemy_hp_anim_ms = self.hp_bar_duration_ms
        self.player_hp_from = player_hp
        self.player_hp_display = player_hp
        self.player_hp_target = player_hp
        self.player_hp_anim_ms = self.hp_bar_duration_ms

    def update(self, dt_ms: int) -> None:
        self.enemy_flash_ms = max(0, self.enemy_flash_ms - dt_ms)
        self.player_flash_ms = max(0, self.player_flash_ms - dt_ms)
        self.screen_shake_ms = max(0, self.screen_shake_ms - dt_ms)
        self.border_flash_ms = max(0, self.border_flash_ms - dt_ms)
        self._update_hp_bar_animation(dt_ms)

        if self.pending_enemy_turn_ms > 0:
            self.pending_enemy_turn_ms = max(0, self.pending_enemy_turn_ms - dt_ms)
            if self.pending_enemy_turn_ms <= 0:
                try:
                    self.combat.resolve_enemy_turn()
                except Exception as e:
                    self.msg = str(e)
                self.input_locked = False

        if self.combat.is_over():
            self.pending_enemy_turn_ms = 0
            self.input_locked = False

        self._consume_damage_events()

        if self._seen_log_count > len(self.combat.log):
            self._seen_log_count = 0

        new_logs = self.combat.log[self._seen_log_count:]
        self._seen_log_count = len(self.combat.log)
        for line in new_logs:
            self.log_toasts.append({
                "text": str(line),
                "age": 0,
                "hold": 4000,
                "fade": 900,
                "duration": 4900,
            })
        if len(self.log_toasts) > 6:
            self.log_toasts = self.log_toasts[-6:]

        alive: List[dict] = []
        for fx in self.effects:
            fx["age"] = int(fx.get("age", 0)) + dt_ms
            if int(fx.get("age", 0)) < int(fx.get("duration", 1)):
                alive.append(fx)
        self.effects = alive

        alive_logs: List[dict] = []
        for toast in self.log_toasts:
            toast["age"] = int(toast.get("age", 0)) + dt_ms
            if int(toast.get("age", 0)) < int(toast.get("duration", 1)):
                alive_logs.append(toast)
        self.log_toasts = alive_logs

    def _update_hp_bar_animation(self, dt_ms: int) -> None:
        enemy = self.combat.enemy
        if enemy is not None:
            target = float(enemy.hp)
            if target != self.enemy_hp_target:
                self.enemy_hp_from = self.enemy_hp_display
                self.enemy_hp_target = target
                self.enemy_hp_anim_ms = 0
            self.enemy_hp_anim_ms = min(self.hp_bar_duration_ms, self.enemy_hp_anim_ms + dt_ms)
            t = ease_out_cubic(self.enemy_hp_anim_ms / max(1, self.hp_bar_duration_ms))
            self.enemy_hp_display = self.enemy_hp_from + (self.enemy_hp_target - self.enemy_hp_from) * t
            self.enemy_hp_display = clamp_float(self.enemy_hp_display, 0.0, float(max(1, enemy.max_hp)))

        target_player = float(self.combat.player_hp)
        if target_player != self.player_hp_target:
            self.player_hp_from = self.player_hp_display
            self.player_hp_target = target_player
            self.player_hp_anim_ms = 0
        self.player_hp_anim_ms = min(self.hp_bar_duration_ms, self.player_hp_anim_ms + dt_ms)
        t = ease_out_cubic(self.player_hp_anim_ms / max(1, self.hp_bar_duration_ms))
        self.player_hp_display = self.player_hp_from + (self.player_hp_target - self.player_hp_from) * t
        self.player_hp_display = clamp_float(self.player_hp_display, 0.0, float(max(1, self.combat.player_max_hp)))

    def _draw_hp_bar(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        actual_hp: int,
        max_hp: int,
        display_hp: float,
        fill_color: Tuple[int, int, int],
        shield: int = 0,
        block: int = 0,
    ) -> None:
        max_hp = max(1, int(max_hp))
        actual_hp = max(0, min(int(actual_hp), max_hp))
        display_hp = clamp_float(display_hp, 0.0, float(max_hp))
        shield = max(0, int(shield))
        block = max(0, int(block))

        base_outer = rect
        base_inner = base_outer.inflate(-4, -4)
        px_per_hp = base_inner.w / max_hp
        total_resource = actual_hp + shield + block
        extended_inner_w = max(base_inner.w, int(round(total_resource * px_per_hp)))
        outer = pygame.Rect(base_outer.x, base_outer.y, extended_inner_w + 4, base_outer.h)
        inner = outer.inflate(-4, -4)

        actual_w = int(round(actual_hp * px_per_hp))
        delayed_w = int(round(display_hp * px_per_hp))
        shield_w = int(round(shield * px_per_hp))
        block_w = int(round(block * px_per_hp))

        actual_rect = pygame.Rect(inner.x, inner.y, actual_w, inner.h)

        pygame.draw.rect(screen, (35, 35, 40), outer, border_radius=9)
        pygame.draw.rect(screen, (70, 70, 78), inner, border_radius=7)

        max_hp_marker_x = inner.x + base_inner.w
        if outer.w > base_outer.w:
            extended_bg = pygame.Rect(max_hp_marker_x, inner.y, max(0, inner.right - max_hp_marker_x), inner.h)
            if extended_bg.w > 0:
                pygame.draw.rect(screen, (52, 52, 58), extended_bg, border_radius=7)

        if delayed_w > actual_w:
            damage_rect = pygame.Rect(inner.x + actual_w, inner.y, delayed_w - actual_w, inner.h)
            pygame.draw.rect(screen, (238, 168, 48), damage_rect, border_radius=7)
        elif delayed_w < actual_w:
            heal_rect = pygame.Rect(inner.x + delayed_w, inner.y, actual_w - delayed_w, inner.h)
            pygame.draw.rect(screen, (95, 205, 105), heal_rect, border_radius=7)

        if actual_w > 0:
            pygame.draw.rect(screen, fill_color, actual_rect, border_radius=7)

        extra_x = inner.x + actual_w
        if shield_w > 0:
            shield_rect = pygame.Rect(extra_x, inner.y, shield_w, inner.h)
            pygame.draw.rect(screen, (165, 165, 172), shield_rect, border_radius=7)
            extra_x += shield_w
        if block_w > 0:
            block_rect = pygame.Rect(extra_x, inner.y, block_w, inner.h)
            pygame.draw.rect(screen, (110, 205, 235), block_rect, border_radius=7)

        if outer.w > base_outer.w:
            pygame.draw.line(screen, (25, 25, 28), (max_hp_marker_x, inner.y), (max_hp_marker_x, inner.bottom), 2)

        pygame.draw.rect(screen, (20, 20, 22), outer, width=2, border_radius=9)

        extras: List[str] = []
        if shield > 0:
            extras.append(f"盾{shield}")
        if block > 0:
            extras.append(f"擋{block}")
        suffix = ("  " + " ".join(extras)) if extras else ""
        text = f"{label}: {actual_hp}/{max_hp}{suffix}"
        shadow = self.font.render(text, True, (20, 20, 20))
        screen.blit(shadow, shadow.get_rect(center=(outer.centerx + 1, outer.centery + 1)))
        surf = self.font.render(text, True, (255, 255, 255))
        screen.blit(surf, surf.get_rect(center=outer.center))

    def _shake_offset(self) -> Tuple[int, int]:
        if self.screen_shake_ms <= 0:
            return (0, 0)
        strength = 8 if self.screen_shake_ms > 120 else 4
        return (choice([-strength, -strength // 2, 0, strength // 2, strength]), choice([-strength, -strength // 2, 0, strength // 2, strength]))

    def _spawn_enemy_attack_fx(self, damage: int) -> None:
        x, y = self._enemy_fx_center
        self.effects.append({"kind": "slash", "x": x, "y": y, "age": 0, "duration": 200})

        if damage > 0:
            self.enemy_flash_ms = max(self.enemy_flash_ms, 120)
            self.effects.append({"kind": "damage_text", "text": f"-{damage}", "x": x, "y": y - 70, "age": 0, "duration": 720, "target": "enemy"})

        if damage >= self.big_hit_threshold:
            self.screen_shake_ms = max(self.screen_shake_ms, 220)

    def _spawn_player_damage_fx(self, damage: int, damage_type: str = "physical") -> None:
        if damage <= 0:
            return
        color_key = damage_type if damage_type in DAMAGE_FX_COLORS else "physical"
        x, y = self._player_fx_center
        self.player_flash_ms = max(self.player_flash_ms, 130)
        self.screen_shake_ms = max(self.screen_shake_ms, 240)
        self.border_flash_ms = max(self.border_flash_ms, self.border_flash_duration_ms)
        self.border_flash_type = color_key
        self.effects.append({
            "kind": "damage_text",
            "text": f"-{damage}",
            "x": x,
            "y": y,
            "age": 0,
            "duration": 720,
            "target": "player",
            "damage_type": color_key,
        })

    def _spawn_player_hit_fx(self, damage: int) -> None:
        self._spawn_player_damage_fx(damage, "physical")

    def _consume_damage_events(self) -> None:
        events = getattr(self.combat, "damage_events", [])
        if self._seen_damage_event_count > len(events):
            self._seen_damage_event_count = 0
        new_events = events[self._seen_damage_event_count:]
        self._seen_damage_event_count = len(events)
        for event in new_events:
            if not isinstance(event, dict):
                continue
            target = str(event.get("target", ""))
            amount = int(event.get("amount", 0) or 0)
            damage_type = str(event.get("damage_type", "physical"))
            if target == "player" and amount > 0:
                self._spawn_player_damage_fx(amount, damage_type)

    def _render_border_flash(self, target_screen: pygame.Surface, sw: int, sh: int) -> None:
        if self.border_flash_ms <= 0:
            return
        color = DAMAGE_FX_COLORS.get(self.border_flash_type, DAMAGE_FX_COLORS["physical"])
        progress = max(0.0, min(1.0, self.border_flash_ms / max(1, self.border_flash_duration_ms)))
        alpha = max(0, min(170, int(170 * progress)))
        border = pygame.Surface((sw, sh), pygame.SRCALPHA)
        pygame.draw.rect(border, (*color, alpha), border.get_rect(), width=12)
        pygame.draw.rect(border, (*color, max(0, alpha // 3)), border.get_rect().inflate(-18, -18), width=6)
        target_screen.blit(border, (0, 0))

    def _draw_damage_text(self, screen: pygame.Surface, fx: dict) -> None:
        duration = max(1, int(fx.get("duration", 1)))
        age = int(fx.get("age", 0))
        progress = min(1.0, age / duration)
        x = int(fx.get("x", 0))
        y = int(fx.get("y", 0)) - int(progress * 56)
        alpha = max(0, min(255, int(255 * (1.0 - progress))))
        text = str(fx.get("text", ""))

        outline = self.font_big.render(text, True, (255, 255, 255))
        outline.set_alpha(alpha)
        for ox, oy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            screen.blit(outline, outline.get_rect(center=(x + ox, y + oy)))

        if fx.get("target") == "enemy":
            color = (210, 35, 35)
        else:
            color = DAMAGE_FX_COLORS.get(str(fx.get("damage_type", "physical")), DAMAGE_FX_COLORS["physical"])
        surf = self.font_big.render(text, True, color)
        surf.set_alpha(alpha)
        screen.blit(surf, surf.get_rect(center=(x, y)))

    def _draw_slash(self, screen: pygame.Surface, fx: dict) -> None:
        duration = max(1, int(fx.get("duration", 1)))
        age = int(fx.get("age", 0))
        progress = min(1.0, age / duration)
        alpha = max(0, min(255, int(255 * (1.0 - progress))))
        x = int(fx.get("x", 0))
        y = int(fx.get("y", 0))

        surf = pygame.Surface((220, 150), pygame.SRCALPHA)
        white = (255, 255, 255, alpha)
        red = (230, 30, 35, max(0, int(alpha * 0.85)))
        glow = (255, 80, 80, max(0, int(alpha * 0.28)))

        offset = int(progress * 18)
        pygame.draw.line(surf, glow, (28 + offset, 22), (188 + offset, 118), 18)
        pygame.draw.line(surf, red, (35 + offset, 28), (181 + offset, 112), 9)
        pygame.draw.line(surf, white, (42 + offset, 35), (172 + offset, 105), 4)
        pygame.draw.line(surf, (255, 255, 255, max(0, int(alpha * 0.7))), (75 + offset, 28), (182 + offset, 82), 3)

        screen.blit(surf, surf.get_rect(center=(x, y)))

    def _render_effects(self, screen: pygame.Surface) -> None:
        for fx in self.effects:
            kind = fx.get("kind")
            if kind == "slash":
                self._draw_slash(screen, fx)
            elif kind == "damage_text":
                self._draw_damage_text(screen, fx)

    def _log_text_color(self, text: str) -> Tuple[int, int, int]:
        danger_words = ["攻擊你", "雙倍", "失去", "中毒", "戰敗", "造成"]
        good_words = ["獲得", "回復", "勝利", "發動"]
        if any(word in text for word in danger_words):
            return (180, 30, 30)
        if any(word in text for word in good_words):
            return (35, 120, 55)
        return (25, 25, 25)

    def _render_log_toasts(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        if not self.log_toasts:
            return

        width = min(390, max(280, sw - 60))
        x = sw - width - 8
        base_y = 24
        gap = 10

        visible = list(reversed(self.log_toasts[-6:]))
        current_y = base_y
        for toast in visible:
            age = int(toast.get("age", 0))
            hold = int(toast.get("hold", 4000))
            fade = max(1, int(toast.get("fade", 900)))
            if age <= hold:
                alpha = 255
                offset_y = 0
            else:
                progress = min(1.0, (age - hold) / fade)
                alpha = max(0, min(255, int(255 * (1.0 - progress))))
                offset_y = -int(42 * progress)

            text = str(toast.get("text", ""))
            lines = _wrap_text(self.font, text, width - 28)[:2]
            if not lines:
                lines = [""]

            height = len(lines) * 24
            y = current_y + offset_y

            color = self._log_text_color(text)
            for i, line in enumerate(lines):
                txt = self.font.render(line, True, color)
                txt.set_alpha(alpha)
                screen.blit(txt, (x, y + i * 24))

            current_y += height + gap

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        target_screen = screen
        shake_x, shake_y = self._shake_offset()
        if self.screen_shake_ms > 0:
            screen = pygame.Surface((sw, sh), pygame.SRCALPHA)

        bg = _load_image("backgrounds", "combat_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((245, 246, 250))

        enemy = self.combat.enemy
        if enemy is None:
            if screen is not target_screen:
                target_screen.fill((20, 20, 20))
                target_screen.blit(screen, (shake_x, shake_y))
            return

        portrait_box = pygame.Rect(0, 0, 360, 260)
        portrait_box.midtop = (sw // 2, 18)
        self._enemy_fx_center = portrait_box.center
        self._player_fx_center = (160, 130)

        portrait = _load_image("enemies", f"{enemy.enemy_id}.png")
        if portrait is not None:
            fitted = _fit_surface_keep_ratio(portrait, (portrait_box.w, portrait_box.h))
            screen.blit(fitted, fitted.get_rect(center=portrait_box.center))
        else:
            pygame.draw.rect(screen, (220, 220, 225), portrait_box, border_radius=16)

        if self.enemy_flash_ms > 0:
            flash_alpha = max(0, min(180, int(180 * self.enemy_flash_ms / 120)))
            flash = pygame.Surface(portrait_box.size, pygame.SRCALPHA)
            flash.fill((255, 255, 255, flash_alpha))
            screen.blit(flash, portrait_box.topleft)

        pygame.draw.rect(screen, (30, 30, 30), portrait_box, width=2, border_radius=16)
        self._render_effects(screen)

        hand_y = sh - CARD_H - 30
        hand_rects = hand_layout(sw, hand_y, len(self.combat.hand))
        btn_play = Button(pygame.Rect(30, hand_y - 70, 120, 48), "攻擊")
        btn_discard = Button(pygame.Rect(160, hand_y - 70, 120, 48), "棄牌")
        btn_sort_v = Button(pygame.Rect(290, hand_y - 70, 150, 48), "依照面值排序")
        btn_sort_s = Button(pygame.Rect(450, hand_y - 70, 140, 48), "依照花色排序")
        btn_back = Button(pygame.Rect(610, hand_y - 70, 160, 48), "返回地圖")

        over = self.combat.is_over()
        is_training = getattr(enemy, "enemy_id", "") == "training_dummy"

        if self.player_flash_ms > 0:
            player_flash_alpha = max(0, min(90, int(90 * self.player_flash_ms / 130)))
            player_flash = pygame.Surface((270, 230), pygame.SRCALPHA)
            player_flash.fill((255, 80, 80, player_flash_alpha))
            screen.blit(player_flash, (20, 18))

        _blit_text_outline(screen, self.font_big, f"Enemy: {enemy.name}", (30, 20))
        self._draw_hp_bar(screen, pygame.Rect(30, 60, 280, 24), "Enemy HP", enemy.hp, enemy.max_hp, self.enemy_hp_display, (190, 45, 45))
        _blit_text_outline(screen, self.font, f"ATK: {enemy.attack_damage}  Timer: {enemy.attack_timer}", (30, 92))
        self._draw_hp_bar(
            screen,
            pygame.Rect(30, 124, 280, 24),
            "Player HP",
            self.combat.player_hp,
            self.combat.player_max_hp,
            self.player_hp_display,
            (45, 150, 75),
            shield=getattr(self.combat, "player_shield", 0),
            block=getattr(self.combat, "player_block", 0),
        )
        _blit_text_outline(screen, self.font, f"Gold: {getattr(self.combat, 'gold', 0)}", (30, 156))
        _blit_text_outline(screen, self.font, f"Discards Left: {self.combat.discard_uses_left}", (30, 184))
        _blit_text_outline(screen, self.font, f"Sigil Slots: {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}", (30, 212))

        disabled = getattr(self.combat, "disabled_suits", {})
        if disabled:
            ds = ", ".join([f"{s.name}:{t}" for s, t in disabled.items()])
            _blit_text_outline(screen, self.font, f"Disabled suit: {ds}", (30, 240), fg=(180, 30, 30))

        locked = self.input_locked or self.pending_enemy_turn_ms > 0
        btn_play.draw(screen, self.font, enabled=(not over and not locked))
        btn_discard.draw(screen, self.font, enabled=(not over and not locked and self.combat.discard_uses_left > 0))
        btn_sort_v.draw(screen, self.font, enabled=not locked)
        btn_sort_s.draw(screen, self.font, enabled=not locked)
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

        for i, c in enumerate(self.combat.hand):
            if i >= len(hand_rects):
                break
            awakened = bool(getattr(self.combat, "is_awakened", lambda _card: False)(c))
            draw_card(screen, self.font, c, hand_rects[i], selected=(i in self.selected), awakened=awakened)

        self._render_log_toasts(screen, sw, sh)

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

        if screen is not target_screen:
            target_screen.fill((20, 20, 20))
            target_screen.blit(screen, (shake_x, shake_y))

        self._render_border_flash(target_screen, sw, sh)

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        pos = event.pos
        enemy = self.combat.enemy
        is_training = enemy is not None and getattr(enemy, "enemy_id", "") == "training_dummy"
        if self._btn_back.hit(pos):
            if self.combat.is_over() or is_training:
                return "back_to_map"
            self.msg = "戰鬥尚未結束"
            return None
        if self.input_locked or self.pending_enemy_turn_ms > 0:
            return None
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
        over = self.combat.is_over()
        if self._btn_play.hit(pos):
            if over and not is_training:
                self.msg = "戰鬥結束"
                return None
            if len(self.selected) > 5:
                self.msg = "最多可選 5 張卡牌"
                return None
            try:
                enemy_hp_before = enemy.hp if enemy is not None else 0

                self.combat.play(self.selected)

                enemy_damage = max(0, enemy_hp_before - enemy.hp) if enemy is not None else 0

                self._spawn_enemy_attack_fx(enemy_damage)

                self.selected = []
                if getattr(self.combat, "enemy_turn_pending", False) and not self.combat.is_over():
                    self.pending_enemy_turn_ms = self.enemy_turn_delay_ms
                    self.input_locked = True
                    self.msg = ""
                else:
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
    events_all = load_events_json(data_dir / "events.json")
    start_events_all = load_start_events_json(data_dir / "start_events.json")

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
    if not hasattr(run, "early_enemy_hp_down_uses"):
        run.early_enemy_hp_down_uses = 0
    if not hasattr(run, "early_enemy_hp_down_ratio"):
        run.early_enemy_hp_down_ratio = 0.3
    run.all_sigils_ref = sigils_all
    run.current_node_id = run.map_graph.start_id

    fullscreen = False
    windowed_size = (w, h)

    scene = "map"
    map_scene = MapScene(run, font, font_big)
    shop_scene: Optional[ShopScene] = None
    combat_scene: Optional[CombatScene] = None
    event_scene: Optional[EventScene] = None
    start_scene: Optional[StartScene] = None
    sigil_choice_scene: Optional[SigilChoiceScene] = None
    pending_combat_reward_tier: Optional[str] = None
    pending_sigil_choice_choices: List[Sigil] = []
    pending_sigil_choice_source: Optional[str] = None
    pending_sigil_choice_messages: List[str] = []

    def start_combat(enemy: Enemy, reward_tier: Optional[str] = None) -> None:
        nonlocal combat_scene, scene, pending_combat_reward_tier
        pending_combat_reward_tier = reward_tier
        combat_enemy = deepcopy(enemy)

        if (
            reward_tier == "normal"
            and getattr(run, "early_enemy_hp_down_uses", 0) > 0
            and getattr(combat_enemy, "enemy_id", "") != "training_dummy"
        ):
            ratio = max(0.0, min(0.9, float(getattr(run, "early_enemy_hp_down_ratio", 0.3))))
            original_max_hp = max(1, int(getattr(combat_enemy, "max_hp", 1)))
            reduced_max_hp = max(1, int(round(original_max_hp * (1.0 - ratio))))
            combat_enemy.max_hp = reduced_max_hp
            combat_enemy.hp = min(max(1, int(getattr(combat_enemy, "hp", reduced_max_hp))), reduced_max_hp)
            run.early_enemy_hp_down_uses = max(0, int(getattr(run, "early_enemy_hp_down_uses", 0)) - 1)
            map_scene.msg = f"狩獵祝福發動：敵人生命值 -{int(ratio * 100)}%"

        c = CombatState()
        c.player_max_hp = run.max_hp
        c.player_hp = run.hp
        if hasattr(c, "gold"):
            c.gold = run.gold
        c.start(combat_enemy)
        c.sigils = [s for s in sigils_all if s.sigil_id in run.equipped_sigils]
        if hasattr(c, "refresh_sigil_state"):
            c.refresh_sigil_state()
        combat_scene = CombatScene(run, c, sigils_all, font, font_big, font_huge)
        scene = "combat"

    def finish_combat_and_return() -> None:
        nonlocal scene, combat_scene, pending_combat_reward_tier
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
            if pending_combat_reward_tier == "normal":
                run.gold += 20
            elif pending_combat_reward_tier == "elite":
                run.gold += 40
            elif node.node_type == NodeType.COMBAT:
                run.gold += 20
            elif node.node_type == NodeType.ELITE:
                run.gold += 40
            elif node.node_type == NodeType.BOSS:
                run.gold += 100
        pending_combat_reward_tier = None
        combat_scene = None
        scene = "map"

    def choose_weighted(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not items:
            return None
        total = sum(max(0, int(item.get("weight", 1))) for item in items)
        if total <= 0:
            return choice(items)
        roll = random() * total
        acc = 0.0
        for item in items:
            acc += max(0, int(item.get("weight", 1)))
            if roll <= acc:
                return item
        return items[-1]

    def choose_start_options() -> List[Dict[str, Any]]:
        if not start_events_all:
            return [{"label": "踏上旅程", "text": "無事發生。", "effects": []}]
        if len(start_events_all) <= 3:
            return list(start_events_all)
        return sample(start_events_all, 3)

    def choose_random_event(node: MapNode) -> Dict[str, Any]:
        game_floor = max(1, node.depth + 1)
        eligible = [
            event_data
            for event_data in events_all
            if int(event_data.get("min_floor", 1)) <= game_floor <= int(event_data.get("max_floor", 99))
        ]
        if not eligible:
            eligible = events_all
        if not eligible:
            return {
                "id": "empty_event",
                "title": "空蕩的房間",
                "description": "你在房間裡停留片刻，沒有發現任何值得帶走的東西。",
                "options": [{"label": "離開", "text": "無事發生。", "effects": []}],
            }
        return choice(eligible)

    def apply_event_effects(effects: List[Dict[str, Any]], messages: List[str]) -> Optional[str]:
        nonlocal pending_sigil_choice_choices
        action: Optional[str] = None

        for effect in effects:
            if not isinstance(effect, dict):
                continue
            effect_type = str(effect.get("type", "nothing"))
            amount = int(effect.get("amount", 0))

            if effect_type == "gain_gold":
                run.gold += amount
                messages.append(f"獲得 {amount} 金幣")

            elif effect_type == "lose_gold":
                lost = min(run.gold, amount)
                run.gold -= lost
                messages.append(f"失去 {lost} 金幣")

            elif effect_type == "gain_hp":
                before = run.hp
                run.hp = min(run.max_hp, run.hp + amount)
                messages.append(f"回復 {run.hp - before} HP")

            elif effect_type == "lose_hp":
                before = run.hp
                run.hp = max(1, run.hp - amount)
                messages.append(f"失去 {before - run.hp} HP")

            elif effect_type == "gain_max_hp":
                run.max_hp += amount
                run.hp += amount
                messages.append(f"最大生命值 +{amount}")

            elif effect_type == "lose_max_hp":
                lost = min(amount, max(0, run.max_hp - 1))
                run.max_hp -= lost
                run.hp = min(run.hp, run.max_hp)
                messages.append(f"最大生命值 -{lost}")

            elif effect_type == "gain_sigil_slot":
                before = run.unlocked_slots
                run.unlocked_slots = min(5, run.unlocked_slots + max(1, amount))
                gained = run.unlocked_slots - before
                messages.append(f"紋章插槽 +{gained}" if gained > 0 else "紋章插槽已達上限")

            elif effect_type == "gain_random_sigil":
                gained_names: List[str] = []
                for _ in range(max(1, amount)):
                    available = [s for s in sigils_all if s.sigil_id not in run.owned_sigils]
                    if not available:
                        run.gold += 30
                        messages.append("已無新紋章，改獲得 30 金幣")
                        break
                    sigil = choice(available)
                    run.owned_sigils.add(sigil.sigil_id)
                    gained_names.append(sigil.name)
                if gained_names:
                    messages.append("獲得紋章：「" + "、".join(gained_names) + "」")

            elif effect_type == "choose_random_sigil":
                available = [s for s in sigils_all if s.sigil_id not in run.owned_sigils]
                count = max(1, amount)
                if not available:
                    run.gold += 30
                    messages.append("已無新紋章，改獲得 30 金幣")
                else:
                    pending_sigil_choice_choices = sample(available, min(count, len(available)))
                    messages.append(f"從 {len(pending_sigil_choice_choices)} 個隨機紋章中選擇 1 個")
                    action = "choose_sigil"

            elif effect_type == "early_enemy_hp_down":
                uses = max(1, amount)
                ratio = max(0.0, min(0.9, float(effect.get("ratio", 0.3))))
                run.early_enemy_hp_down_uses = int(getattr(run, "early_enemy_hp_down_uses", 0)) + uses
                run.early_enemy_hp_down_ratio = ratio
                messages.append(f"接下來 {uses} 場普通戰，敵人生命值 -{int(ratio * 100)}%")

            elif effect_type == "start_normal_combat":
                if normals:
                    messages.append("遭遇敵人")
                    start_combat(choice(normals), "normal")
                    action = "combat"

            elif effect_type == "start_elite_combat":
                if elites:
                    messages.append("遭遇菁英敵人")
                    start_combat(choice(elites), "elite")
                    action = "combat"

            elif effect_type == "open_shop":
                messages.append("遇見商人")
                action = "shop"

            elif effect_type == "random_choice":
                picked = choose_weighted([item for item in effect.get("choices", []) if isinstance(item, dict)])
                if picked is not None:
                    msg = str(picked.get("message", ""))
                    if msg:
                        messages.append(msg)
                    nested_action = apply_event_effects(
                        [e for e in picked.get("effects", []) if isinstance(e, dict)],
                        messages,
                    )
                    if nested_action is not None:
                        action = nested_action

            elif effect_type == "nothing":
                msg = str(effect.get("message", ""))
                if msg:
                    messages.append(msg)

        return action

    def apply_event_option(option: Dict[str, Any]) -> None:
        nonlocal scene, shop_scene, event_scene, sigil_choice_scene, pending_sigil_choice_source, pending_sigil_choice_messages
        messages: List[str] = []
        action = apply_event_effects(
            [effect for effect in option.get("effects", []) if isinstance(effect, dict)],
            messages,
        )
        node = run.map_graph.nodes[run.current_node_id]

        if action == "choose_sigil":
            event_scene = None
            pending_sigil_choice_source = "event"
            pending_sigil_choice_messages = list(messages)
            sigil_choice_scene = SigilChoiceScene(
                run,
                pending_sigil_choice_choices,
                "紋章選擇",
                "房間中的力量凝結成數枚紋章。選擇其中一枚，讓它成為這趟旅程的一部分。",
                font,
                font_big,
            )
            scene = "sigil_choice"
            return

        if action == "shop":
            shop_scene = ShopScene(run, sigils_all, font, font_big)
            event_scene = None
            scene = "shop"
            map_scene.msg = "；".join(messages) if messages else "未知房間：遇見商人"
            return

        if action == "combat":
            event_scene = None
            map_scene.msg = "；".join(messages) if messages else "未知房間：遭遇敵人"
            return

        node.cleared = True
        event_scene = None
        scene = "map"
        map_scene.msg = "事件完成" if not messages else "事件完成：" + "；".join(messages)

    def apply_start_option(option: Dict[str, Any]) -> None:
        nonlocal scene, shop_scene, start_scene, sigil_choice_scene, pending_sigil_choice_source, pending_sigil_choice_messages
        messages: List[str] = []
        action = apply_event_effects(
            [effect for effect in option.get("effects", []) if isinstance(effect, dict)],
            messages,
        )
        node = run.map_graph.nodes[run.current_node_id]

        start_scene = None

        if action == "choose_sigil":
            pending_sigil_choice_source = "start"
            pending_sigil_choice_messages = list(messages)
            sigil_choice_scene = SigilChoiceScene(
                run,
                pending_sigil_choice_choices,
                "起始祝福",
                "鮮血完成了祭禮。現在，從三枚隨機紋章中選擇一枚作為開局力量。",
                font,
                font_big,
            )
            scene = "sigil_choice"
            return

        node.cleared = True

        if action == "shop":
            shop_scene = ShopScene(run, sigils_all, font, font_big)
            scene = "shop"
            map_scene.msg = "起始祝福：" + "；".join(messages) if messages else "起始房間：遇見商人"
            return

        if action == "combat":
            map_scene.msg = "起始祝福：" + "；".join(messages) if messages else "起始房間：遭遇敵人"
            return

        scene = "map"
        map_scene.msg = "旅程開始" if not messages else "旅程開始：" + "；".join(messages)

    def finish_sigil_choice(sigil: Sigil) -> None:
        nonlocal scene, sigil_choice_scene, pending_sigil_choice_source, pending_sigil_choice_messages
        run.owned_sigils.add(sigil.sigil_id)
        node = run.map_graph.nodes[run.current_node_id]
        node.cleared = True

        messages = list(pending_sigil_choice_messages)
        messages.append(f"獲得紋章：「{sigil.name}」")

        source = pending_sigil_choice_source
        sigil_choice_scene = None
        pending_sigil_choice_source = None
        pending_sigil_choice_messages = []
        scene = "map"

        if source == "event":
            map_scene.msg = "事件完成：" + "；".join(messages)
        else:
            map_scene.msg = "旅程開始：" + "；".join(messages)

    def enter_unknown_room(node: MapNode) -> None:
        nonlocal scene, shop_scene, event_scene
        game_floor = max(1, node.depth + 1)
        roll = random()

        if game_floor >= 6:
            if roll < 0.10 and elites:
                map_scene.msg = "未知房間：遭遇菁英敵人！"
                start_combat(choice(elites), "elite")
                return
            if roll < 0.30 and normals:
                map_scene.msg = "未知房間：遭遇敵人！"
                start_combat(choice(normals), "normal")
                return
            if roll < 0.45:
                map_scene.msg = "未知房間：遇見商人"
                shop_scene = ShopScene(run, sigils_all, font, font_big)
                scene = "shop"
                return
        else:
            if roll < 0.25 and normals:
                map_scene.msg = "未知房間：遭遇敵人！"
                start_combat(choice(normals), "normal")
                return
            if roll < 0.40:
                map_scene.msg = "未知房間：遇見商人"
                shop_scene = ShopScene(run, sigils_all, font, font_big)
                scene = "shop"
                return

        event_scene = EventScene(run, choose_random_event(node), font, font_big)
        scene = "event"

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
                if action == "consume":
                    continue
                if action == "training" and training_dummy is not None:
                    start_combat(training_dummy)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not map_scene.show_equip_panel:
                    nid = map_scene.handle_map_click(event.pos, sw, sh)
                    if nid is None:
                        continue
                    run.current_node_id = nid
                    node = run.map_graph.nodes[nid]
                    node_type_name = node.node_type.name

                    if node_type_name == "START":
                        start_scene = StartScene(run, choose_start_options(), font, font_big)
                        scene = "start"

                    elif node_type_name == "CAMP":
                        before = run.hp
                        run.hp = min(run.max_hp, run.hp + 20)
                        node.cleared = True
                        map_scene.msg = f"休息點回復 {run.hp - before} HP"

                    elif node_type_name == "TREASURE":
                        reward = 50
                        run.gold += reward
                        node.cleared = True
                        map_scene.msg = f"寶箱獲得 {reward} 金幣"

                    elif node_type_name == "UNKNOWN":
                        enter_unknown_room(node)

                    elif node_type_name == "SHOP":
                        shop_scene = ShopScene(run, sigils_all, font, font_big)
                        scene = "shop"

                    elif node_type_name == "COMBAT" and normals:
                        start_combat(choice(normals), "normal")

                    elif node_type_name == "ELITE" and elites:
                        start_combat(choice(elites), "elite")

                    elif node_type_name == "BOSS" and bosses:
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

            elif scene == "start":
                if start_scene is None:
                    scene = "map"
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    option = start_scene.handle_click(event.pos)
                    if option is not None:
                        apply_start_option(option)

            elif scene == "event":
                if event_scene is None:
                    scene = "map"
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    option = event_scene.handle_click(event.pos)
                    if option is not None:
                        apply_event_option(option)

            elif scene == "sigil_choice":
                if sigil_choice_scene is None:
                    scene = "map"
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    picked = sigil_choice_scene.handle_click(event.pos)
                    if picked is not None:
                        finish_sigil_choice(picked)

            elif scene == "combat":
                if combat_scene is None:
                    continue
                action = combat_scene.handle_event(event)
                if action == "back_to_map":
                    finish_combat_and_return()

        dt_ms = clock.tick(60)

        if scene == "combat" and combat_scene is not None:
            combat_scene.update(dt_ms)

        if scene == "map":
            map_scene.render(screen, sw, sh)
        elif scene == "shop":
            if shop_scene is None:
                shop_scene = ShopScene(run, sigils_all, font, font_big)
            shop_scene.render(screen, sw, sh)
        elif scene == "start":
            if start_scene is None:
                scene = "map"
                map_scene.render(screen, sw, sh)
            else:
                start_scene.render(screen, sw, sh)

        elif scene == "event":
            if event_scene is None:
                scene = "map"
                map_scene.render(screen, sw, sh)
            else:
                event_scene.render(screen, sw, sh)
        elif scene == "sigil_choice":
            if sigil_choice_scene is None:
                scene = "map"
                map_scene.render(screen, sw, sh)
            else:
                sigil_choice_scene.render(screen, sw, sh)
        elif scene == "combat" and combat_scene is not None:
            combat_scene.render(screen, sw, sh)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
