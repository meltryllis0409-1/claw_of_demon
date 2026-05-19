from __future__ import annotations

import sys
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable
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
SELECTED_CARD_FLOAT_Y = 10
ICON_BASE = 36
ICON_SIZE = ICON_BASE * 2
TOP_ICON_SIZE = ICON_SIZE
TOP_ICON_TOP = 18
TOP_ICON_RIGHT_MARGIN = 30
TOP_ICON_GAP = 12
SPECIAL_MAP_ICON_SCALE = 2.5
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


def _shrink_line_ends(
    start: Tuple[int, int],
    end: Tuple[int, int],
    inset: int = 30,
    start_inset: Optional[int] = None,
    end_inset: Optional[int] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    si = inset if start_inset is None else max(0, int(start_inset))
    ei = inset if end_inset is None else max(0, int(end_inset))
    if dist <= si + ei + 1:
        mid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return mid, mid

    ux = dx / dist
    uy = dy / dist
    return (int(x1 + ux * si), int(y1 + uy * si)), (int(x2 - ux * ei), int(y2 - uy * ei))


def _draw_dashed_path(
    screen: pygame.Surface,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int] = (84, 67, 45),
    width: int = 3,
    dash_len: int = 7,
    gap_len: int = 9,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist <= 0:
        return

    ux = dx / dist
    uy = dy / dist
    step = max(1, dash_len + gap_len)
    t = 0.0

    while t < dist:
        seg_start = t
        seg_end = min(t + dash_len, dist)
        sx = x1 + ux * seg_start
        sy = y1 + uy * seg_start
        ex = x1 + ux * seg_end
        ey = y1 + uy * seg_end

        pygame.draw.line(
            screen,
            color,
            (int(sx), int(sy)),
            (int(ex), int(ey)),
            width,
        )

        t += step


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


SIGIL_RARITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "常見": (80, 210, 110),
    "稀有": (85, 165, 255),
    "史詩": (185, 95, 255),
    "傳說": (248, 196, 65),
}

SIGIL_RARITY_ALIASES: Dict[str, str] = {
    "common": "常見",
    "rare": "稀有",
    "epic": "史詩",
    "legendary": "傳說",
    "normal": "常見",
}


def _sigil_rarity(sigil: Sigil) -> str:
    rarity = getattr(sigil, "rarity", "常見")
    if callable(rarity):
        try:
            rarity = rarity()
        except TypeError:
            rarity = "常見"
    rarity_text = str(rarity or "常見").strip()
    rarity_text = SIGIL_RARITY_ALIASES.get(rarity_text.lower(), rarity_text)
    return rarity_text if rarity_text in SIGIL_RARITY_COLORS else "常見"


def _sigil_name_color(sigil: Sigil) -> Tuple[int, int, int]:
    color = getattr(sigil, "rarity_color", None)
    if callable(color):
        try:
            color = color()
        except TypeError:
            color = None
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        try:
            return (int(color[0]), int(color[1]), int(color[2]))
        except Exception:
            pass
    return SIGIL_RARITY_COLORS[_sigil_rarity(sigil)]


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
        self._btn_equip = Button(pygame.Rect(0, 0, TOP_ICON_SIZE, TOP_ICON_SIZE), "裝備紋章")
        self._btn_training = Button(pygame.Rect(0, 0, TOP_ICON_SIZE, TOP_ICON_SIZE), "進入訓練場")

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
        pad_x = 145
        pad_y = 145
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

    def _node_icon_scale(self, node: MapNode) -> float:
        if node.node_type.name in {"START", "BOSS"}:
            return SPECIAL_MAP_ICON_SCALE
        return 1.0

    def _node_icon_size(self, node: MapNode) -> int:
        return max(1, int(ICON_SIZE * self._node_icon_scale(node)))

    def _node_hit_radius(self, node: MapNode) -> int:
        return max(18, int(18 * self._node_icon_scale(node)))

    def _node_path_inset(self, node: MapNode) -> int:
        if node.node_type.name in {"START", "BOSS"}:
            return self._node_icon_size(node) // 2 + 8
        return 30

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
            r = self._node_hit_radius(node)
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= r ** 2:
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
            r = self._node_hit_radius(node)
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= r ** 2 and nid in clickables:
                return nid
        return None

    def _equip_panel_rect(self, sw: int, sh: int) -> pygame.Rect:
        return pygame.Rect(sw - 360, 80, 330, sh - 120)

    def _layout_top_icon_buttons(self, sw: int) -> None:
        y = TOP_ICON_TOP
        size = TOP_ICON_SIZE
        self._btn_training.rect = pygame.Rect(
            sw - TOP_ICON_RIGHT_MARGIN - size,
            y,
            size,
            size,
        )
        self._btn_equip.rect = pygame.Rect(
            self._btn_training.rect.x - TOP_ICON_GAP - size,
            y,
            size,
            size,
        )

    def _draw_top_icon_button(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        icon_name: str,
        fallback_text: str,
        hovered: bool,
    ) -> None:
        if hovered:
            _draw_glow(screen, rect.center, rect.w, (255, 236, 164), 54)
            _draw_glow(screen, rect.center, max(1, rect.w // 2), (255, 250, 210), 64)

        shadow = pygame.Surface((rect.w + 8, rect.h + 8), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 95), shadow.get_rect(), border_radius=12)
        screen.blit(shadow, (rect.x - 4, rect.y - 4))

        frame = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(frame, (18, 18, 22, 145), frame.get_rect(), border_radius=10)
        screen.blit(frame, rect.topleft)

        icon = _load_image("ui", icon_name, size=(rect.w, rect.h))
        if icon is not None:
            screen.blit(icon, rect.topleft)
        else:
            pygame.draw.rect(screen, (42, 42, 48), rect, border_radius=10)
            _blit_text_outline(
                screen,
                self.font,
                fallback_text,
                rect.center,
                fg=(245, 230, 180),
                outline=(12, 12, 14),
                center=True,
            )

        if hovered:
            pygame.draw.rect(screen, (255, 236, 164), rect.inflate(6, 6), width=3, border_radius=12)
            pygame.draw.rect(screen, (255, 250, 210), rect.inflate(2, 2), width=1, border_radius=10)

    def _draw_top_icon_buttons(self, screen: pygame.Surface, sw: int) -> None:
        self._layout_top_icon_buttons(sw)
        mouse_pos = pygame.mouse.get_pos()
        self._draw_top_icon_button(
            screen,
            self._btn_equip.rect,
            "sigils.png",
            "紋章",
            self._btn_equip.hit(mouse_pos),
        )
        self._draw_top_icon_button(
            screen,
            self._btn_training.rect,
            "dummy.png",
            "訓練",
            self._btn_training.hit(mouse_pos),
        )

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

        title_color = _sigil_name_color(sigil)
        text_y = y + padding_y
        for i, line in enumerate(lines):
            color = title_color if i == 0 else (245, 245, 245)
            surf = self.font.render(line, True, color)
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
        self._layout_top_icon_buttons(sw)
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

        self._draw_top_icon_buttons(screen, sw)

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
                path_start, path_end = _shrink_line_ends(
                    (x1, y1),
                    (x2, y2),
                    start_inset=self._node_path_inset(node),
                    end_inset=self._node_path_inset(n2),
                )
                _draw_dashed_path(
                    screen,
                    path_start,
                    path_end,
                    color=(84, 67, 45),
                    width=3,
                    dash_len=7,
                    gap_len=9,
                )

        for node in g.nodes.values():
            x, y = self._node_screen_pos(node, content_rect)
            scale = self._node_icon_scale(node)
            icon_size = self._node_icon_size(node)

            if node.node_id in clickables:
                _draw_glow(screen, (x, y), int(36 * scale), (255, 230, 120), 80)
            if hovered == node.node_id:
                _draw_glow(screen, (x, y), int(52 * scale), (255, 255, 200), 120)

            icon = _load_image("map_icons", self._icon_name(node.node_type), size=(icon_size, icon_size))
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
                fallback_radius = self._node_hit_radius(node)
                pygame.draw.circle(screen, fallback_colors.get(node.node_type.name, (90, 120, 200)), (x, y), fallback_radius)
                pygame.draw.circle(screen, (30, 30, 30), (x, y), fallback_radius, 2)

            label_offset = icon_size // 2 + 22
            if hovered == node.node_id:
                _blit_text_outline(screen, self.font, self._label(node.node_type), (x, y - label_offset), center=True)
            if node.cleared:
                _blit_text_outline(screen, self.font, "CLEAR!", (x, y + label_offset), center=True)

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
                    _blit_text_outline(
                        screen,
                        self.font,
                        s.name,
                        (r.x + 52, r.y + 8),
                        fg=_sigil_name_color(s),
                        outline=(20, 20, 20),
                    )
                    _blit_text_outline(screen, self.font, mark, (r.x + 52, r.y + 26))
                    y += 60

            if hovered_sigil is not None:
                self._draw_sigil_tooltip(screen, sw, sh, mouse_pos, hovered_sigil)


class ShopScene:
    def __init__(self, run: RunState, all_sigils: List[Sigil], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.all_sigils = all_sigils
        available_sigils = [s for s in all_sigils if s.sigil_id not in self.run.owned_sigils]
        self.shop_sigils = sample(available_sigils, min(3, len(available_sigils))) if available_sigils else []
        self.shop_potions = self._roll_potions()
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.hover_item: Optional[Dict[str, Any]] = None
        self.item_regions: List[Dict[str, Any]] = []
        self.bought_potion_ids: set[str] = set()
        self.btn_leave = Button(pygame.Rect(0, 0, 120, 44), "離開")

        if not hasattr(self.run, "sigil_slot_buy_count"):
            self.run.sigil_slot_buy_count = 0

    def _roll_potions(self) -> List[Dict[str, Any]]:
        potion_pool = [
            {
                "id": "greed_potion",
                "name": "貪婪藥水",
                "desc": "接下來 5 場戰鬥金幣掉落量 +100%，但受到傷害 +20%。",
                "cost": 100,
                "icon": "potion_greed.png",
                "effect_type": "greed",
                "battles": 5,
            },
            {
                "id": "stone_skin_potion",
                "name": "石膚藥劑",
                "desc": "接下來 4 場戰鬥開始時獲得 10 點護盾。",
                "cost": 95,
                "icon": "potion_stoneskin.png",
                "effect_type": "stone_skin",
                "battles": 4,
                "shield": 10,
            },
            {
                "id": "healing_potion",
                "name": "回復藥水",
                "desc": "立即回復 20 HP。",
                "cost": 55,
                "icon": "potion_heal.png",
                "effect_type": "heal",
                "heal": 20,
            },
            {
                "id": "regeneration_potion",
                "name": "再生劑",
                "desc": "接下來 5 回合，每回合回復 10 HP。",
                "cost": 105,
                "icon": "potion_regen.png",
                "effect_type": "regen",
                "turns": 5,
                "heal_per_turn": 10,
            },
            {
                "id": "rage_potion",
                "name": "憤怒藥劑",
                "desc": "接下來 3 場戰鬥內，造成傷害 +50%。",
                "cost": 140,
                "icon": "potion_rage.png",
                "effect_type": "rage",
                "battles": 3,
            },
            {
                "id": "resistance_potion",
                "name": "抗擊藥劑",
                "desc": "接下來 3 場戰鬥內，受到傷害 -25%。",
                "cost": 125,
                "icon": "potion_resistance.png",
                "effect_type": "resistance",
                "battles": 3,
            },
        ]
        return sample(potion_pool, min(3, len(potion_pool)))

    def _slot_cost(self) -> int:
        return 75 * (2 ** int(getattr(self.run, "sigil_slot_buy_count", 0)))

    def _draw_item_glow(self, screen: pygame.Surface, center: Tuple[int, int], hovered: bool, affordable: bool) -> None:
        if hovered:
            _draw_glow(screen, center, 82, (255, 231, 140), 72)
            _draw_glow(screen, center, 54, (255, 248, 190), 86)

    def _draw_price(self, screen: pygame.Surface, center_x: int, y: int, cost: int, affordable: bool) -> None:
        icon = _load_image("items", "gold.png", size=(30, 30))
        text = str(int(cost))
        tw = self.font.size(text)[0]
        total_w = 34 + tw
        x = center_x - total_w // 2
        if icon is not None:
            screen.blit(icon, (x, y))
        else:
            pygame.draw.circle(screen, (222, 157, 38), (x + 15, y + 15), 13)
            pygame.draw.circle(screen, (80, 50, 10), (x + 15, y + 15), 13, 2)
        _blit_text_soft_outline(
            screen,
            self.font,
            text,
            (x + 36, y + 3),
            fg=((246, 174, 48) if affordable else (220, 55, 45)),
            outline=(5, 5, 5),
            outline_alpha=175,
        )

    def _draw_shop_item(
        self,
        screen: pygame.Surface,
        kind: str,
        item_id: str,
        name: str,
        desc: str,
        cost: int,
        center: Tuple[int, int],
        icon: Optional[pygame.Surface],
        fallback_color: Tuple[int, int, int],
        mouse_pos: Tuple[int, int],
        name_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        cx, cy = center
        hit_rect = pygame.Rect(0, 0, 118, 142)
        hit_rect.center = (cx, cy + 14)
        hovered = hit_rect.collidepoint(mouse_pos)
        affordable = self.run.gold >= cost

        self._draw_item_glow(screen, (cx, cy), hovered, affordable)

        if icon is not None:
            icon_rect = icon.get_rect(center=(cx, cy))
            screen.blit(icon, icon_rect)
        else:
            pygame.draw.circle(screen, fallback_color, (cx, cy), 32)
            pygame.draw.circle(screen, (25, 22, 18), (cx, cy), 32, 3)
            pygame.draw.circle(screen, (255, 240, 170), (cx - 10, cy - 12), 7)
            if kind == "slot":
                pygame.draw.rect(screen, (230, 216, 160), pygame.Rect(cx - 22, cy - 28, 44, 56), border_radius=8)
                pygame.draw.rect(screen, (45, 35, 25), pygame.Rect(cx - 22, cy - 28, 44, 56), 3, border_radius=8)
                _blit_text_outline(screen, self.font_big, "+", (cx, cy), fg=(120, 60, 35), outline=(245, 230, 170), center=True)

        if hovered:
            pygame.draw.circle(screen, (255, 226, 128), (cx, cy), 43, 3)

        display_name_color = name_color if name_color is not None else (240, 226, 190)
        _blit_text_outline(screen, self.font, name, (cx, cy + 50), fg=display_name_color, outline=(18, 12, 8), center=True)
        self._draw_price(screen, cx, cy + 78, cost, affordable)

        item_info = {
            "kind": kind,
            "id": item_id,
            "name": name,
            "desc": desc,
            "cost": cost,
            "rect": hit_rect,
            "name_color": display_name_color,
        }
        self.item_regions.append(item_info)
        if hovered:
            self.hover_item = item_info

    def _draw_tooltip(self, screen: pygame.Surface, sw: int, sh: int, mouse_pos: Tuple[int, int]) -> None:
        if self.hover_item is None:
            return

        title = str(self.hover_item.get("name", ""))
        desc = str(self.hover_item.get("desc", ""))
        cost = int(self.hover_item.get("cost", 0))
        lines = [title] + _wrap_text(self.font, desc, 280) + [f"價格：{cost} 金幣"]
        padding_x = 13
        padding_y = 10
        line_h = 24
        tooltip_w = min(330, max(180, max(self.font.size(line)[0] for line in lines) + padding_x * 2))
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
        pygame.draw.rect(tooltip, (0, 0, 0, 220), tooltip.get_rect(), border_radius=8)
        pygame.draw.rect(tooltip, (255, 230, 170, 80), tooltip.get_rect(), width=1, border_radius=8)
        screen.blit(tooltip, (x, y))

        title_color = self.hover_item.get("name_color", (255, 228, 160))
        if not (isinstance(title_color, (tuple, list)) and len(title_color) >= 3):
            title_color = (255, 228, 160)
        ty = y + padding_y
        for i, line in enumerate(lines):
            color = tuple(title_color[:3]) if i == 0 else (245, 245, 245)
            surf = self.font.render(line, True, color)
            screen.blit(surf, (x + padding_x, ty))
            ty += line_h

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = _load_image("backgrounds", "shop_bg.png", size=(sw, sh))
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((45, 38, 32))

        _blit_text_outline(screen, self.font_big, "SHOP", (30, 20), fg=(242, 220, 165), outline=(12, 8, 6))
        _draw_map_hp_bar(screen, self.font, pygame.Rect(30, 58, 240, 28), self.run.hp, self.run.max_hp)
        _draw_map_gold(screen, self.font, 300, 56, self.run.gold)
        _blit_text_outline(
            screen,
            self.font,
            f"紋章插槽 {self.run.unlocked_slots}",
            (410, 60),
            fg=(245, 232, 190),
            outline=(18, 12, 8),
        )

        self.btn_leave.rect = pygame.Rect(sw - 142, 24, 108, 42)
        pygame.draw.rect(screen, (18, 14, 12), self.btn_leave.rect, border_radius=10)
        pygame.draw.rect(screen, (235, 210, 150), self.btn_leave.rect, width=2, border_radius=10)
        leave_text = self.font.render("離開", True, (245, 225, 170))
        screen.blit(leave_text, leave_text.get_rect(center=self.btn_leave.rect.center))

        mouse_pos = pygame.mouse.get_pos()
        self.item_regions = []
        self.hover_item = None

        _blit_text_outline(screen, self.font_big, "紋章", (int(sw * 0.10), int(sh * 0.20)), fg=(242, 220, 165), outline=(12, 8, 6))
        _blit_text_outline(screen, self.font_big, "藥水", (int(sw * 0.10), int(sh * 0.52)), fg=(242, 220, 165), outline=(12, 8, 6))

        sigil_centers = [
            (int(sw * 0.15), int(sh * 0.34)),
            (int(sw * 0.30), int(sh * 0.34)),
            (int(sw * 0.45), int(sh * 0.34)),
        ]
        potion_centers = [
            (int(sw * 0.15), int(sh * 0.66)),
            (int(sw * 0.30), int(sh * 0.66)),
            (int(sw * 0.45), int(sh * 0.66)),
        ]
        slot_center = (int(sw * 0.59), int(sh * 0.50))

        for sigil, center in zip(self.shop_sigils, sigil_centers):
            if sigil.sigil_id in self.run.owned_sigils:
                continue
            icon = _load_image("sigils", f"{sigil.sigil_id}.png", size=(72, 72))
            self._draw_shop_item(
                screen,
                "sigil",
                sigil.sigil_id,
                sigil.name,
                str(getattr(sigil, "desc", "")),
                int(getattr(sigil, "cost", 0)),
                center,
                icon,
                (130, 90, 180),
                mouse_pos,
                name_color=_sigil_name_color(sigil),
            )

        for potion, center in zip(self.shop_potions, potion_centers):
            potion_id = str(potion.get("id", "potion"))
            if potion_id in self.bought_potion_ids:
                continue
            icon_name = str(potion.get("icon", "potion.png"))
            icon = _load_image("items", icon_name, size=(76, 76)) or _load_image("items", "potion.png", size=(76, 76))
            self._draw_shop_item(
                screen,
                "potion",
                potion_id,
                str(potion.get("name", "藥水")),
                str(potion.get("desc", "")),
                int(potion.get("cost", 0)),
                center,
                icon,
                (95, 185, 105),
                mouse_pos,
            )

        slot_cost = self._slot_cost()
        slot_icon = _load_image("items", "sigil_slot.png", size=(96, 116)) or _load_image("items", "slot.png", size=(96, 116))
        self._draw_shop_item(
            screen,
            "slot",
            "sigil_slot",
            "紋章插槽",
            "購買後紋章插槽 +1。此服務可重複購買，每次價格變為上次的 2 倍。",
            slot_cost,
            slot_center,
            slot_icon,
            (185, 120, 55),
            mouse_pos,
        )

        if self.msg:
            _blit_text_outline(screen, self.font, self.msg, (34, sh - 48), fg=(240, 210, 150), outline=(16, 10, 8))

        self._draw_tooltip(screen, sw, sh, mouse_pos)

    def _apply_potion(self, potion: Dict[str, Any]) -> str:
        effect_type = str(potion.get("effect_type", ""))

        if effect_type == "heal":
            heal = max(0, int(potion.get("heal", 0)))
            before = self.run.hp
            self.run.hp = min(self.run.max_hp, self.run.hp + heal)
            return f"回復 {self.run.hp - before} HP"

        if effect_type == "regen":
            turns = max(1, int(potion.get("turns", 5)))
            amount = max(1, int(potion.get("heal_per_turn", 10)))
            self.run.regen_potion_turns_left = int(getattr(self.run, "regen_potion_turns_left", 0)) + turns
            self.run.regen_potion_amount = amount
            return f"再生效果 +{turns} 回合，每回合回復 {amount} HP"

        if effect_type == "greed":
            battles = max(1, int(potion.get("battles", 5)))
            self.run.greed_potion_battles_left = int(getattr(self.run, "greed_potion_battles_left", 0)) + battles
            return f"貪婪效果 +{battles} 場戰鬥"

        if effect_type == "stone_skin":
            battles = max(1, int(potion.get("battles", 4)))
            self.run.stone_skin_potion_battles_left = int(getattr(self.run, "stone_skin_potion_battles_left", 0)) + battles
            return f"石膚效果 +{battles} 場戰鬥"

        if effect_type == "rage":
            battles = max(1, int(potion.get("battles", 3)))
            self.run.rage_potion_battles_left = int(getattr(self.run, "rage_potion_battles_left", 0)) + battles
            return f"憤怒效果 +{battles} 場戰鬥"

        if effect_type == "resistance":
            battles = max(1, int(potion.get("battles", 3)))
            self.run.resistance_potion_battles_left = int(getattr(self.run, "resistance_potion_battles_left", 0)) + battles
            return f"抗擊效果 +{battles} 場戰鬥"

        return "已飲用藥水"

    def handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        if self.btn_leave.hit(pos):
            return "leave"

        for item in self.item_regions:
            rect = item.get("rect")
            if not isinstance(rect, pygame.Rect) or not rect.collidepoint(pos):
                continue

            kind = str(item.get("kind", ""))
            item_id = str(item.get("id", ""))
            cost = int(item.get("cost", 0))

            if self.run.gold < cost:
                self.msg = "金幣不足"
                return None

            if kind == "sigil":
                sigil = next((s for s in self.shop_sigils if s.sigil_id == item_id), None)
                if sigil is None:
                    return None
                if item_id in self.run.owned_sigils:
                    return None
                self.run.gold -= cost
                self.run.owned_sigils.add(item_id)
                self.msg = f"購買紋章：{sigil.name}"
                return None

            if kind == "potion":
                potion = next((p for p in self.shop_potions if str(p.get("id", "")) == item_id), None)
                if potion is None or item_id in self.bought_potion_ids:
                    return None
                self.run.gold -= cost
                self.msg = self._apply_potion(potion)
                self.bought_potion_ids.add(item_id)
                return None

            if kind == "slot":
                self.run.gold -= cost
                self.run.unlocked_slots += 1
                self.run.sigil_slot_buy_count = int(getattr(self.run, "sigil_slot_buy_count", 0)) + 1
                self.msg = f"紋章插槽 +1；下次價格 {self._slot_cost()}G"
                return None

        return None

class OpeningScene:
    def __init__(self, font_big: pygame.font.Font, font_huge: pygame.font.Font) -> None:
        self.font_big = font_big
        self.font_huge = font_huge
        self.start_label = "遊戲開始"

    def _load_opening_background(self, sw: int, sh: int) -> Optional[pygame.Surface]:
        for parts in (("backgrounds", "opening_bg.png"), ("ui", "opening_bg.png"), ("opening_bg.png",)):
            bg = _load_image(*parts, size=(sw, sh))
            if bg is not None:
                return bg
        return None

    def _load_logo(self, sw: int, sh: int) -> Optional[pygame.Surface]:
        for parts in (("ui", "gamelogo.png"), ("backgrounds", "gamelogo.png"), ("gamelogo.png",)):
            raw = _load_image(*parts)
            if raw is None:
                continue
            return _fit_surface_keep_ratio(raw, (min(760, int(sw * 0.78)), min(240, int(sh * 0.32))))
        return None

    def _start_text_rect(self, sw: int, sh: int) -> pygame.Rect:
        text_surf = self.font_big.render(self.start_label, True, (245, 230, 180))
        rect = text_surf.get_rect(center=(sw // 2, int(sh * 0.66)))
        return rect.inflate(44, 30)

    def handle_click(self, pos: Tuple[int, int], sw: int, sh: int) -> bool:
        return self._start_text_rect(sw, sh).collidepoint(pos)

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = self._load_opening_background(sw, sh)
        if bg is not None:
            screen.blit(bg, (0, 0))
        else:
            screen.fill((12, 10, 14))
            _draw_glow(screen, (sw // 2, int(sh * 0.34)), min(sw, sh) // 3, (110, 40, 40), 55)

        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 35))
        screen.blit(dim, (0, 0))

        logo = self._load_logo(sw, sh)
        if logo is not None:
            logo_rect = logo.get_rect(center=(sw // 2, int(sh * 0.28)))
            screen.blit(logo, logo_rect)
        else:
            _blit_text_outline(
                screen,
                self.font_huge,
                "Claw of Demon",
                (sw // 2, int(sh * 0.28)),
                fg=(245, 210, 150),
                outline=(12, 6, 6),
                center=True,
                extra_bold=True,
            )

        mouse_pos = pygame.mouse.get_pos()
        hit_rect = self._start_text_rect(sw, sh)
        text_rect = self.font_big.render(self.start_label, True, (245, 230, 180)).get_rect(center=hit_rect.center)
        hovered = hit_rect.collidepoint(mouse_pos)

        if hovered:
            underline_w = text_rect.w + 58
            underline_h = 9
            underline = pygame.Surface((underline_w, underline_h), pygame.SRCALPHA)
            pygame.draw.rect(underline, (255, 220, 90, 120), underline.get_rect(), border_radius=5)
            screen.blit(underline, (text_rect.centerx - underline_w // 2, text_rect.bottom + 9))

        _blit_text_outline(
            screen,
            self.font_big,
            self.start_label,
            text_rect.center,
            fg=((255, 236, 170) if hovered else (245, 225, 175)),
            outline=(15, 9, 6),
            center=True,
        )


class StartScene:
    def __init__(self, run: RunState, options: List[Dict[str, Any]], font: pygame.font.Font, font_big: pygame.font.Font) -> None:
        self.run = run
        self.options = options
        self.font = font
        self.font_big = font_big
        self.msg = ""
        self.option_buttons: List[Tuple[Dict[str, Any], Button, bool]] = []
        self.elapsed_ms = 0
        self.scroll_duration_ms = 3900
        self.reveal_delay_ms = 1800
        self.fade_duration_ms = 650

    def update(self, dt_ms: int) -> None:
        self.elapsed_ms += max(0, int(dt_ms))

    def _reveal_start_ms(self) -> int:
        return self.scroll_duration_ms + self.reveal_delay_ms

    def _reveal_alpha(self) -> float:
        progress = (self.elapsed_ms - self._reveal_start_ms()) / max(1, self.fade_duration_ms)
        return ease_out_cubic(progress)

    def _load_start_background(self, sw: int, sh: int) -> Optional[pygame.Surface]:
        for filename in ("startroom_bg.png", "stratroom_bg.png", "start_bg.png"):
            raw = _load_image("backgrounds", filename)
            if raw is None:
                continue
            rw, rh = raw.get_size()
            if rw <= 0 or rh <= 0:
                continue
            target_h = max(sh, int(round(rh * (sw / rw))))
            return _load_image("backgrounds", filename, size=(sw, target_h))
        return None

    def _draw_scrolling_background(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = self._load_start_background(sw, sh)
        if bg is None:
            screen.fill((22, 18, 22))
            return

        bg_h = bg.get_height()
        max_scroll = max(0, bg_h - sh)
        progress = ease_out_cubic(self.elapsed_ms / max(1, self.scroll_duration_ms))
        scroll_y = int(round(max_scroll * (1.0 - progress)))
        screen.blit(bg, (0, -scroll_y))

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        self._draw_scrolling_background(screen, sw, sh)

        reveal_alpha = self._reveal_alpha()
        if reveal_alpha <= 0.0:
            self.option_buttons = []
            return

        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)

        _blit_text_outline(
            overlay,
            self.font_big,
            "起始房間",
            (30, 20),
            fg=(245, 230, 190),
            outline=(12, 8, 8),
        )
        _blit_text_outline(
            overlay,
            self.font,
            f"生命值 {self.run.hp}/{self.run.max_hp}   金幣 {self.run.gold}   紋章插槽 {len(self.run.equipped_sigils)}/{self.run.unlocked_slots}",
            (30, 60),
            fg=(245, 245, 245),
            outline=(10, 10, 10),
        )

        title_y = 112
        _blit_text_outline(
            overlay,
            self.font_big,
            "雕像的囈語",
            (70, title_y),
            fg=(255, 214, 150),
            outline=(15, 9, 8),
        )

        description = "雕像的眼角似乎滲出了鮮血。雕像前的木桌上有些幾近燃盡的蠟燭，和一張已經泛黃的羊皮紙。於是你走了向前..."
        y = title_y + 52
        for line in _wrap_text(self.font, description, sw - 140):
            _blit_text_outline(
                overlay,
                self.font,
                line,
                (70, y),
                fg=(245, 240, 225),
                outline=(8, 8, 8),
            )
            y += 28

        self.option_buttons = []
        options = [opt for opt in self.options if isinstance(opt, dict)]
        if not options:
            options = [{"label": "踏上旅程", "text": "無事發生。", "effects": []}]

        option_gap = 14
        option_h = 78
        total_options_h = len(options) * option_h + max(0, len(options) - 1) * option_gap
        option_start_y = max(y + 34, sh - total_options_h - 44)
        option_w = min(sw - 140, 860)
        option_x = (sw - option_w) // 2

        mouse_pos = pygame.mouse.get_pos()
        for idx, option in enumerate(options, start=1):
            enabled = _option_enabled(self.run, option)
            r = pygame.Rect(option_x, option_start_y, option_w, option_h)
            label = str(option.get("label", f"選項 {idx}"))
            text = str(option.get("text", ""))
            b = Button(r, f"{idx}. {label}")

            hovered = r.collidepoint(mouse_pos)
            mask_alpha = 128 if enabled else 92
            option_surf = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(option_surf, (0, 0, 0, mask_alpha), option_surf.get_rect(), border_radius=12)
            border_color = (255, 229, 150, 205) if hovered and enabled else (255, 230, 170, 95)
            if not enabled:
                border_color = (170, 170, 170, 85)
            pygame.draw.rect(option_surf, border_color, option_surf.get_rect(), width=2, border_radius=12)
            overlay.blit(option_surf, r.topleft)

            label_color = (255, 222, 148) if enabled else (170, 170, 170)
            desc_color = (238, 238, 238) if enabled else (145, 145, 145)
            _blit_text_outline(
                overlay,
                self.font,
                f"{idx}. {label}",
                (r.x + 18, r.y + 10),
                fg=label_color,
                outline=(0, 0, 0),
            )

            text_lines = _wrap_text(self.font, text, r.w - 36)
            for line_i, line in enumerate(text_lines[:2]):
                _blit_text_outline(
                    overlay,
                    self.font,
                    line,
                    (r.x + 18, r.y + 40 + line_i * 21),
                    fg=desc_color,
                    outline=(0, 0, 0),
                )

            self.option_buttons.append((option, b, enabled))
            option_start_y += option_h + option_gap

        if self.msg:
            _blit_text_outline(
                overlay,
                self.font,
                self.msg,
                (option_x, sh - 30),
                fg=(255, 115, 100),
                outline=(10, 10, 10),
            )

        overlay.set_alpha(max(0, min(255, int(255 * reveal_alpha))))
        screen.blit(overlay, (0, 0))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        if self.elapsed_ms < self._reveal_start_ms():
            return None
        for option, button, enabled in self.option_buttons:
            if button.hit(pos):
                if not enabled:
                    self.msg = "目前條件不足，無法選擇這個選項。"
                    return None
                return option
        return None


class StartRewardScene:
    def __init__(
        self,
        run: RunState,
        option: Dict[str, Any],
        all_sigils: List[Sigil],
        font: pygame.font.Font,
        font_big: pygame.font.Font,
    ) -> None:
        self.run = run
        self.option = deepcopy(option)
        self.all_sigils = all_sigils
        self.font = font
        self.font_big = font_big
        self.elapsed_ms = 0
        self.damage_delay_ms = 500
        self.damage_done = False
        self.flash_ms = 0
        self.flash_duration_ms = 460
        self.shake_ms = 0
        self.msg = ""
        self.icon_rect = pygame.Rect(0, 0, 96, 96)
        self.continue_rect = pygame.Rect(0, 0, 0, 0)
        self.sigil_buttons: List[Tuple[Sigil, pygame.Rect]] = []
        self.sigil_icon_rects: Dict[str, pygame.Rect] = {}
        self.pickup_fx: Optional[Dict[str, Any]] = None
        self.pickup_duration_ms = 500
        self.finished_option: Optional[Dict[str, Any]] = None
        self.hp_loss = self._direct_hp_loss()
        self.max_hp_loss = self._direct_max_hp_loss()
        self.instant_max_hp_lost = 0
        self.sigil_choice_amount = self._sigil_choice_amount()
        self.sigil_choices = self._roll_sigil_choices()
        self.instant_cost_fx = self._should_play_instant_cost_fx()
        self.bg_morph_duration_ms = 1000
        self.bg_morph_half_ms = 500
        self.fade_from_black_ms = 200 if self.instant_cost_fx else 0
        if self.instant_cost_fx:
            if self.hp_loss > 0:
                self.run.hp = max(1, self.run.hp - self.hp_loss)
            if self.max_hp_loss > 0:
                lost_max_hp = min(self.max_hp_loss, max(0, self.run.max_hp - 1))
                self.instant_max_hp_lost = lost_max_hp
                self.run.max_hp -= lost_max_hp
                self.run.hp = min(self.run.hp, self.run.max_hp)
            self.damage_done = True
            self.flash_ms = self.flash_duration_ms
            self.shake_ms = 260
        self.sigil_choice_description = (
            "你用小刀劃破了手腕，鮮血汩汩流出，在祭壇上沿著鐫刻精美的凹槽流淌，"
            "你發現這些紅色的線條形成了一個六角星的圖騰，似乎在召喚著什麼力量。"
            "當最後一滴血落下，圖騰發出微弱的光芒，三枚紋章緩緩浮現，懸停在空中，等待你的選擇。"
        )

    def _effects(self) -> List[Dict[str, Any]]:
        return [e for e in self.option.get("effects", []) if isinstance(e, dict)]

    def _direct_hp_loss(self) -> int:
        total = 0
        for effect in self._effects():
            if str(effect.get("type", "")) == "lose_hp":
                total += max(0, int(effect.get("amount", 0)))
        return total

    def _sigil_choice_amount(self) -> int:
        amount = 0
        for effect in self._effects():
            if str(effect.get("type", "")) == "choose_random_sigil":
                amount = max(amount, max(1, int(effect.get("amount", 1))))
        return amount

    def _direct_max_hp_loss(self) -> int:
        total = 0
        for effect in self._effects():
            if str(effect.get("type", "")) == "lose_max_hp":
                total += max(0, int(effect.get("amount", 0)))
        return total

    def _is_blood_money_scene(self) -> bool:
        option_id = str(self.option.get("id", ""))
        label = str(self.option.get("label", ""))
        return option_id == "blood_money" or label == "血金契約"

    def _should_play_instant_cost_fx(self) -> bool:
        if self.hp_loss > 0 and self._is_sigil_choice_scene():
            return True
        if self.max_hp_loss > 0 and self._is_blood_money_scene():
            return True
        return False

    def _is_sigil_choice_scene(self) -> bool:
        return self.sigil_choice_amount > 0

    def _roll_sigil_choices(self) -> List[Sigil]:
        if self.sigil_choice_amount <= 0:
            return []
        available = [s for s in self.all_sigils if s.sigil_id not in self.run.owned_sigils]
        if not available:
            return []
        return sample(available, min(self.sigil_choice_amount, len(available)))

    def option_without_direct_hp_loss(self) -> Dict[str, Any]:
        cleaned = deepcopy(self.option)
        removed_types = {"lose_hp"}
        if self.instant_cost_fx:
            removed_types.add("lose_max_hp")
        cleaned["effects"] = [
            e for e in cleaned.get("effects", [])
            if not (isinstance(e, dict) and str(e.get("type", "")) in removed_types)
        ]
        if self.instant_max_hp_lost > 0:
            cleaned["effects"].append({"type": "nothing", "message": f"最大生命值 -{self.instant_max_hp_lost}"})
        return cleaned

    def option_for_picked_sigil(self, sigil: Sigil) -> Dict[str, Any]:
        cleaned = deepcopy(self.option)
        kept_effects: List[Dict[str, Any]] = []
        for effect in cleaned.get("effects", []):
            if not isinstance(effect, dict):
                continue
            effect_type = str(effect.get("type", ""))
            if effect_type in {"lose_hp", "choose_random_sigil"}:
                continue
            kept_effects.append(effect)
        kept_effects.append({
            "type": "gain_specific_sigil",
            "sigil_id": sigil.sigil_id,
            "name": sigil.name,
        })
        cleaned["effects"] = kept_effects
        return cleaned

    def option_for_no_sigil_available(self) -> Dict[str, Any]:
        cleaned = deepcopy(self.option)
        kept_effects: List[Dict[str, Any]] = []
        for effect in cleaned.get("effects", []):
            if not isinstance(effect, dict):
                continue
            effect_type = str(effect.get("type", ""))
            if effect_type in {"lose_hp", "choose_random_sigil"}:
                continue
            kept_effects.append(effect)
        kept_effects.append({"type": "gain_gold", "amount": 30})
        cleaned["effects"] = kept_effects
        return cleaned

    def _icon_filename(self) -> str:
        label_text = f"{self.option.get('label', '')} {self.option.get('text', '')}".lower()
        effects = self._effects()
        effect_types = {str(e.get("type", "")) for e in effects}
        if "gain_gold" in effect_types or "gold" in label_text or "金幣" in label_text:
            return "cash_bag.png"
        if (
            "choose_random_sigil" in effect_types
            or "gain_random_sigil" in effect_types
            or "gain_sigil" in effect_types
            or "sigil" in label_text
            or "紋章" in label_text
            or "文章" in label_text
        ):
            return "sigil.png"
        if "gain_sigil_slot" in effect_types or "插槽" in label_text:
            return "sigil_slot.png"
        if "heal" in effect_types or "hp" in label_text or "生命" in label_text:
            return "heart.png"
        return "claw.png"

    def update(self, dt_ms: int) -> None:
        dt_ms = max(0, int(dt_ms))
        self.elapsed_ms += dt_ms
        self.flash_ms = max(0, self.flash_ms - dt_ms)
        self.shake_ms = max(0, self.shake_ms - dt_ms)
        if self.pickup_fx is not None:
            self.pickup_fx["age"] = int(self.pickup_fx.get("age", 0)) + dt_ms
            if int(self.pickup_fx.get("age", 0)) >= int(self.pickup_fx.get("duration", self.pickup_duration_ms)):
                self.finished_option = self.pickup_fx.get("option")
                self.pickup_fx = None
        if (
            self.hp_loss > 0
            and self.fade_from_black_ms <= 0
            and not self.damage_done
            and self.elapsed_ms >= self.damage_delay_ms
        ):
            self.run.hp = max(1, self.run.hp - self.hp_loss)
            self.damage_done = True
            self.flash_ms = self.flash_duration_ms
            self.shake_ms = 260

    def _shake_offset(self) -> Tuple[int, int]:
        if self.shake_ms <= 0:
            return (0, 0)
        strength = 8
        t = self.elapsed_ms // 24
        x = ((t * 37) % (strength * 2 + 1)) - strength
        y = ((t * 53) % (strength * 2 + 1)) - strength
        return int(x), int(y)

    def _load_background(self, sw: int, sh: int) -> Optional[pygame.Surface]:
        for filename in ("startroom_second_bg.png", "startroom_reward_bg.png", "startroom_bg.png", "start_bg.png"):
            bg = _load_image("backgrounds", filename, size=(sw, sh))
            if bg is not None:
                return bg
        return None

    def _load_background_morph(self, sw: int, sh: int) -> Optional[pygame.Surface]:
        return _load_image("backgrounds", "startroom_second_bg2.png", size=(sw, sh))

    def _background_morph_alpha(self) -> int:
        if self.bg_morph_duration_ms <= 0:
            return 0
        t = max(0, min(self.elapsed_ms, self.bg_morph_duration_ms))
        if t <= self.bg_morph_half_ms:
            progress = t / max(1, self.bg_morph_half_ms)
        else:
            progress = 1.0 - ((t - self.bg_morph_half_ms) / max(1, self.bg_morph_duration_ms - self.bg_morph_half_ms))
        return max(0, min(255, int(255 * clamp_float(progress, 0.0, 1.0))))

    def _load_sigil_choice_icon(self, sigil: Sigil, size: int) -> Optional[pygame.Surface]:
        icon_size = (size, size)
        return (
            _load_image("ui", "sigils.png", size=icon_size)
            or _load_image("ui", "sigil.png", size=icon_size)
        )

    def _start_pickup_fx(
        self,
        option: Dict[str, Any],
        rect: pygame.Rect,
        icon_kind: str,
        sigil: Optional[Sigil] = None,
        filename: Optional[str] = None,
    ) -> None:
        self.pickup_fx = {
            "option": option,
            "rect": rect.copy(),
            "kind": icon_kind,
            "sigil": sigil,
            "filename": filename,
            "age": 0,
            "duration": self.pickup_duration_ms,
        }

    def _is_pickup_animating_sigil(self, sigil: Sigil) -> bool:
        return (
            self.pickup_fx is not None
            and self.pickup_fx.get("kind") == "sigil"
            and self.pickup_fx.get("sigil") is sigil
        )

    def _draw_pickup_fx(self, screen: pygame.Surface) -> None:
        if self.pickup_fx is None:
            return
        rect = self.pickup_fx.get("rect")
        if not isinstance(rect, pygame.Rect):
            return
        duration = max(1, int(self.pickup_fx.get("duration", self.pickup_duration_ms)))
        age = max(0, int(self.pickup_fx.get("age", 0)))
        progress = clamp_float(age / duration, 0.0, 1.0)
        eased = ease_out_cubic(progress)
        alpha = max(0, min(255, int(255 * (1.0 - progress))))
        draw_rect = rect.copy()
        draw_rect.centery = int(rect.centery - rect.h * 0.5 * eased)

        icon: Optional[pygame.Surface] = None
        kind = str(self.pickup_fx.get("kind", ""))
        if kind == "sigil":
            sigil = self.pickup_fx.get("sigil")
            if isinstance(sigil, Sigil):
                icon = self._load_sigil_choice_icon(sigil, rect.w)
        else:
            filename = self.pickup_fx.get("filename")
            if isinstance(filename, str) and filename:
                icon = _load_image("ui", filename, size=(rect.w, rect.h))

        if icon is None:
            icon = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            pygame.draw.circle(icon, (130, 90, 180), (rect.w // 2, rect.h // 2), max(8, rect.w // 2 - 8))
            _blit_text_outline(icon, self.font, "祝福", (rect.w // 2, rect.h // 2), fg=(245, 230, 180), outline=(0, 0, 0), center=True)
        else:
            icon = icon.copy()
        icon.set_alpha(alpha)
        screen.blit(icon, draw_rect.topleft)

    def take_finished_option(self) -> Optional[Dict[str, Any]]:
        option = self.finished_option
        self.finished_option = None
        return option

    def _draw_sigil_choice_tooltip(
        self,
        screen: pygame.Surface,
        sw: int,
        sh: int,
        mouse_pos: Tuple[int, int],
        sigil: Sigil,
    ) -> None:
        title = str(getattr(sigil, "name", "未知紋章"))
        desc = str(getattr(sigil, "desc", "")).strip() or "沒有說明。"
        max_text_w = 300
        lines = [title] + _wrap_text(self.font, desc, max_text_w)
        padding_x = 13
        padding_y = 10
        line_h = 24
        tooltip_w = min(350, max(180, max(self.font.size(line)[0] for line in lines) + padding_x * 2))
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
        pygame.draw.rect(tooltip, (255, 255, 255, 90), rect, width=1, border_radius=8)
        screen.blit(tooltip, (x, y))

        title_color = _sigil_name_color(sigil)
        text_y = y + padding_y
        for i, line in enumerate(lines):
            color = title_color if i == 0 else (245, 245, 245)
            surf = self.font.render(line, True, color)
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

    def _draw_sigil_choice_content(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        content_w = min(860, max(520, sw - 160))
        content_x = (sw - content_w) // 2
        title_y = 76

        title = str(self.option.get("label", "鮮血祭禮"))
        _blit_text_outline(
            screen,
            self.font_big,
            title,
            (sw // 2, title_y + 24),
            fg=(255, 226, 170),
            outline=(8, 6, 6),
            center=True,
        )

        y = title_y + 66
        for line in _wrap_text(self.font, self.sigil_choice_description, content_w - 56)[:4]:
            _blit_text_outline(
                screen,
                self.font,
                line,
                (content_x + 28, y),
                fg=(240, 235, 220),
                outline=(0, 0, 0),
            )
            y += 25

        self.sigil_buttons = []
        self.sigil_icon_rects = {}
        mouse_pos = pygame.mouse.get_pos()
        hovered_sigil: Optional[Sigil] = None

        if not self.sigil_choices:
            self.continue_rect = pygame.Rect((sw - content_w) // 2, y + 42, content_w, 80)
            _blit_text_outline(
                screen,
                self.font,
                "目前沒有可選擇的新紋章。點擊此處繼續，改獲得 30 金幣。",
                self.continue_rect.center,
                fg=(240, 235, 220),
                outline=(0, 0, 0),
                center=True,
            )
            return

        icon_size = 96
        center_x = sw // 2
        icon_y = max(y + 98, int(sh * 0.54))
        horizontal_gap = min(220, max(150, content_w // 4))
        centers = [
            (center_x - horizontal_gap, icon_y),
            (center_x, icon_y),
            (center_x + horizontal_gap, icon_y),
        ]

        for sigil, center in zip(self.sigil_choices[:3], centers):
            icon_rect = pygame.Rect(0, 0, icon_size, icon_size)
            icon_rect.center = center
            hit_rect = icon_rect.inflate(24, 24)
            hovered = hit_rect.collidepoint(mouse_pos)
            self.sigil_buttons.append((sigil, hit_rect))
            self.sigil_icon_rects[sigil.sigil_id] = icon_rect.copy()
            is_picked = self._is_pickup_animating_sigil(sigil)

            if hovered and not is_picked:
                _draw_glow(screen, icon_rect.center, 82, (255, 255, 255), 46)
                _draw_glow(screen, icon_rect.center, 58, (255, 236, 164), 54)
                hovered_sigil = sigil

            if not is_picked:
                icon = self._load_sigil_choice_icon(sigil, icon_size)
                if icon is not None:
                    screen.blit(icon, icon_rect.topleft)
                else:
                    pygame.draw.circle(screen, (130, 90, 180), icon_rect.center, icon_size // 2 - 8)
                    pygame.draw.circle(screen, (255, 230, 170), icon_rect.center, icon_size // 2 - 8, 3)
                    _blit_text_outline(
                        screen,
                        self.font,
                        "紋章",
                        icon_rect.center,
                        fg=(245, 230, 180),
                        outline=(0, 0, 0),
                        center=True,
                    )

        hint_y = min(sh - 58, icon_y + 135)
        _blit_text_outline(
            screen,
            self.font,
            "將滑鼠移到紋章上查看效果，點擊其中一枚帶走。",
            (sw // 2, hint_y),
            fg=(235, 230, 215),
            outline=(0, 0, 0),
            center=True,
        )

        if hovered_sigil is not None:
            self._draw_sigil_choice_tooltip(screen, sw, sh, mouse_pos, hovered_sigil)

    def _draw_icon_reward_content(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        title = str(self.option.get("label", "起始祝福"))
        text = str(self.option.get("text", ""))
        _blit_text_outline(
            screen,
            self.font_big,
            title,
            (sw // 2, int(sh * 0.22)),
            fg=(255, 226, 170),
            outline=(15, 8, 8),
            center=True,
        )
        y = int(sh * 0.29)
        for line in _wrap_text(self.font, text, min(760, sw - 160))[:3]:
            _blit_text_outline(
                screen,
                self.font,
                line,
                (sw // 2, y),
                fg=(240, 235, 220),
                outline=(8, 8, 8),
                center=True,
            )
            y += 27

        icon_size = 96
        self.icon_rect = pygame.Rect(0, 0, icon_size, icon_size)
        self.icon_rect.center = (sw // 2, int(sh * 0.54))
        hovered = self.icon_rect.collidepoint(pygame.mouse.get_pos()) and self.pickup_fx is None
        icon_is_picked = self.pickup_fx is not None and self.pickup_fx.get("kind") == "icon"

        if hovered:
            _draw_glow(screen, self.icon_rect.center, 86, (255, 255, 255), 42)

        if not icon_is_picked:
            icon = _load_image("ui", self._icon_filename(), size=(icon_size, icon_size))
            if icon is not None:
                screen.blit(icon, self.icon_rect.topleft)
            else:
                fallback = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
                pygame.draw.rect(fallback, (20, 20, 24, 180), fallback.get_rect(), border_radius=14)
                pygame.draw.rect(fallback, (255, 255, 255, 95), fallback.get_rect(), width=2, border_radius=14)
                screen.blit(fallback, self.icon_rect.topleft)
                _blit_text_outline(screen, self.font, "祝福", self.icon_rect.center, fg=(245, 230, 180), outline=(0, 0, 0), center=True)

        if hovered:
            pygame.draw.rect(screen, (255, 255, 255), self.icon_rect.inflate(8, 8), width=3, border_radius=14)
            pygame.draw.rect(screen, (255, 255, 255), self.icon_rect.inflate(2, 2), width=1, border_radius=12)

        hint = "點擊圖示繼續"
        if self.hp_loss > 0 and not self.damage_done:
            hint = "承受代價中……"
        _blit_text_outline(
            screen,
            self.font,
            hint,
            (sw // 2, int(sh * 0.72)),
            fg=(235, 235, 235),
            outline=(8, 8, 8),
            center=True,
        )

    def _draw_scene_content(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        bg = self._load_background(sw, sh)
        if bg is not None:
            screen.blit(bg, (0, 0))
            bg2 = self._load_background_morph(sw, sh)
            bg2_alpha = self._background_morph_alpha()
            if bg2 is not None and bg2_alpha > 0:
                bg2 = bg2.copy()
                bg2.set_alpha(bg2_alpha)
                screen.blit(bg2, (0, 0))
        else:
            screen.fill((18, 14, 18))
            _draw_glow(screen, (sw // 2, sh // 2), 260, (95, 35, 35), 50)

        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 45))
        screen.blit(dim, (0, 0))

        if self._is_sigil_choice_scene():
            self._draw_sigil_choice_content(screen, sw, sh)
        else:
            self._draw_icon_reward_content(screen, sw, sh)

        self._draw_pickup_fx(screen)

        if self.flash_ms > 0:
            flash_duration = max(1, int(getattr(self, "flash_duration_ms", 460)))
            pulse = 0.65 + 0.35 * abs(math.sin(self.elapsed_ms * 0.045))
            alpha = max(0, min(190, int(190 * (self.flash_ms / flash_duration) * pulse)))
            flash = pygame.Surface((sw, sh), pygame.SRCALPHA)
            flash.fill((255, 20, 20, alpha))
            screen.blit(flash, (0, 0))

    def _fade_from_black_alpha(self) -> int:
        if self.fade_from_black_ms <= 0:
            return 0
        progress = clamp_float(self.elapsed_ms / max(1, self.fade_from_black_ms), 0.0, 1.0)
        return max(0, min(255, int(255 * (1.0 - progress))))

    def render(self, screen: pygame.Surface, sw: int, sh: int) -> None:
        if self.shake_ms > 0:
            temp = pygame.Surface((sw, sh), pygame.SRCALPHA)
            self._draw_scene_content(temp, sw, sh)
            screen.fill((0, 0, 0))
            screen.blit(temp, self._shake_offset())
        else:
            self._draw_scene_content(screen, sw, sh)

        fade_alpha = self._fade_from_black_alpha()
        if fade_alpha > 0:
            fade = pygame.Surface((sw, sh), pygame.SRCALPHA)
            fade.fill((0, 0, 0, fade_alpha))
            screen.blit(fade, (0, 0))

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        if self._fade_from_black_alpha() > 0 or self.pickup_fx is not None:
            return None
        if self.hp_loss > 0 and not self.damage_done:
            return None

        if self._is_sigil_choice_scene():
            for sigil, rect in self.sigil_buttons:
                if rect.collidepoint(pos):
                    icon_rect = self.sigil_icon_rects.get(sigil.sigil_id, rect.inflate(-24, -24))
                    self._start_pickup_fx(self.option_for_picked_sigil(sigil), icon_rect, "sigil", sigil=sigil)
                    return None
            if not self.sigil_choices and self.continue_rect.collidepoint(pos):
                self.finished_option = self.option_for_no_sigil_available()
                return None
            return None

        if self.icon_rect.collidepoint(pos):
            self._start_pickup_fx(
                self.option_without_direct_hp_loss(),
                self.icon_rect,
                "icon",
                filename=self._icon_filename(),
            )
            return None
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

                _blit_text_outline(
                    screen,
                    self.font_big,
                    f"{idx}. {sigil.name}",
                    (r.x + 76, r.y + 10),
                    fg=_sigil_name_color(sigil),
                    outline=(20, 20, 20),
                )
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
        self.hand_sort_mode: Optional[str] = None
        self._last_hand_signature: Tuple[Tuple[int, str, str], ...] = ()
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

    def _hand_signature(self) -> Tuple[Tuple[int, str, str], ...]:
        return tuple(
            (
                id(card),
                str(getattr(card, "rank", "")),
                str(getattr(getattr(card, "suit", ""), "value", getattr(card, "suit", ""))),
            )
            for card in self.combat.hand
        )

    def _apply_hand_sort(self, clear_selected: bool = True) -> None:
        if self.hand_sort_mode == "value":
            sort_hand_by_value(self.combat.hand)
        elif self.hand_sort_mode == "suit":
            sort_hand_by_suit(self.combat.hand)

        if clear_selected:
            self.selected = []

        self._last_hand_signature = self._hand_signature()

    def _sync_hand_sort(self) -> None:
        current_signature = self._hand_signature()
        if self.hand_sort_mode is None:
            self._last_hand_signature = current_signature
            return

        if current_signature != self._last_hand_signature:
            self._apply_hand_sort(clear_selected=True)

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

        self._sync_hand_sort()

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
        y = int(fx.get("y", 0)) + 44

        surf = pygame.Surface((340, 240), pygame.SRCALPHA)
        white = (255, 255, 255, alpha)
        red = (230, 30, 35, max(0, int(alpha * 0.85)))
        glow = (255, 80, 80, max(0, int(alpha * 0.30)))

        offset = int(progress * 26)
        pygame.draw.line(surf, glow, (42 + offset, 34), (302 + offset, 194), 30)
        pygame.draw.line(surf, red, (52 + offset, 44), (290 + offset, 184), 15)
        pygame.draw.line(surf, white, (64 + offset, 56), (276 + offset, 170), 7)
        pygame.draw.line(surf, (255, 255, 255, max(0, int(alpha * 0.72))), (118 + offset, 45), (294 + offset, 132), 5)

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

        enemy = self.combat.enemy
        combat_scene_bg = None
        if enemy is not None:
            combat_scene_bg = _load_image("combat_scenes", f"{enemy.enemy_id}.png", size=(sw, sh))

        if combat_scene_bg is not None:
            screen.blit(combat_scene_bg, (0, 0))
        else:
            bg = _load_image("backgrounds", "combat_bg.png", size=(sw, sh))
            if bg is not None:
                screen.blit(bg, (0, 0))
            else:
                screen.fill((245, 246, 250))

        if enemy is None:
            if screen is not target_screen:
                target_screen.fill((20, 20, 20))
                target_screen.blit(screen, (shake_x, shake_y))
            return

        portrait_box = pygame.Rect(0, 0, 360, 260)
        portrait_box.midtop = (sw // 2, 18)
        self._enemy_fx_center = portrait_box.center
        self._player_fx_center = (160, 130)

        if combat_scene_bg is None:
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

        if combat_scene_bg is None:
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
        self._draw_hp_bar(
            screen,
            pygame.Rect(30, 60, 280, 24),
            "Enemy HP",
            enemy.hp,
            enemy.max_hp,
            self.enemy_hp_display,
            (190, 45, 45),
            shield=getattr(self.combat, "enemy_shield", getattr(enemy, "shield", 0)),
            block=getattr(self.combat, "enemy_block", getattr(enemy, "block", 0)),
        )
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

        hand_hit_rects = [rr.copy() for rr in hand_rects]
        for i, c in enumerate(self.combat.hand):
            if i >= len(hand_rects):
                break
            awakened = bool(getattr(self.combat, "is_awakened", lambda _card: False)(c))
            card_rect = hand_rects[i].copy()
            if i in self.selected:
                card_rect.y -= SELECTED_CARD_FLOAT_Y
            hand_hit_rects[i] = card_rect.copy()
            draw_card(screen, self.font, c, card_rect, selected=(i in self.selected), awakened=awakened)

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
        self._hand_rects = hand_hit_rects

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
            self.hand_sort_mode = "value"
            self._apply_hand_sort(clear_selected=True)
            self.msg = "已切換為依面值自動排序"
            return None
        if self._btn_sort_s.hit(pos):
            self.hand_sort_mode = "suit"
            self._apply_hand_sort(clear_selected=True)
            self.msg = "已切換為依花色自動排序"
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

    scene = "opening"
    opening_scene = OpeningScene(font_big, font_huge)
    map_scene = MapScene(run, font, font_big)
    shop_scene: Optional[ShopScene] = None
    combat_scene: Optional[CombatScene] = None
    event_scene: Optional[EventScene] = None
    start_scene: Optional[StartScene] = None
    start_reward_scene: Optional[StartRewardScene] = None
    sigil_choice_scene: Optional[SigilChoiceScene] = None
    pending_combat_reward_tier: Optional[str] = None
    pending_sigil_choice_choices: List[Sigil] = []
    pending_sigil_choice_source: Optional[str] = None
    pending_sigil_choice_messages: List[str] = []

    opening_transition_active = False
    opening_transition_elapsed_ms = 0
    opening_transition_fade_out_ms = 400
    opening_transition_hold_ms = 500
    opening_transition_fade_in_ms = 400
    opening_transition_start_scene_created = False

    scene_transition_active = False
    scene_transition_elapsed_ms = 0
    scene_transition_fade_out_ms = 500
    scene_transition_fade_in_ms = 500
    scene_transition_switch_done = False
    scene_transition_action: Optional[Callable[[], None]] = None

    def begin_black_transition(action: Callable[[], None], fade_out_ms: int = 500, fade_in_ms: int = 500) -> None:
        nonlocal scene_transition_active, scene_transition_elapsed_ms
        nonlocal scene_transition_fade_out_ms, scene_transition_fade_in_ms
        nonlocal scene_transition_switch_done, scene_transition_action
        scene_transition_active = True
        scene_transition_elapsed_ms = 0
        scene_transition_fade_out_ms = max(1, int(fade_out_ms))
        scene_transition_fade_in_ms = max(1, int(fade_in_ms))
        scene_transition_switch_done = False
        scene_transition_action = action

    def apply_potion_battle_buffs(c: CombatState, combat_enemy: Enemy) -> None:
        if getattr(combat_enemy, "enemy_id", "") == "training_dummy":
            return

        if int(getattr(run, "greed_potion_battles_left", 0)) > 0:
            run.greed_potion_battles_left = max(0, int(getattr(run, "greed_potion_battles_left", 0)) - 1)
            c.potion_gold_bonus_percent += 100
            c.potion_damage_taken_bonus_percent += 20
            c.log.append(f"貪婪藥水作用中：金幣掉落 +100%，受到傷害 +20%。剩餘 {run.greed_potion_battles_left} 場。")

        if int(getattr(run, "stone_skin_potion_battles_left", 0)) > 0:
            run.stone_skin_potion_battles_left = max(0, int(getattr(run, "stone_skin_potion_battles_left", 0)) - 1)
            if hasattr(c, "gain_shield"):
                c.gain_shield(10)
            c.log.append(f"石膚藥劑作用中：戰鬥開始獲得 10 點護盾。剩餘 {run.stone_skin_potion_battles_left} 場。")

        if int(getattr(run, "rage_potion_battles_left", 0)) > 0:
            run.rage_potion_battles_left = max(0, int(getattr(run, "rage_potion_battles_left", 0)) - 1)
            c.potion_damage_bonus_percent += 50
            c.log.append(f"憤怒藥劑作用中：造成傷害 +50%。剩餘 {run.rage_potion_battles_left} 場。")

        if int(getattr(run, "resistance_potion_battles_left", 0)) > 0:
            run.resistance_potion_battles_left = max(0, int(getattr(run, "resistance_potion_battles_left", 0)) - 1)
            c.potion_damage_taken_reduction_percent += 25
            c.log.append(f"抗擊藥劑作用中：受到傷害 -25%。剩餘 {run.resistance_potion_battles_left} 場。")

        if int(getattr(run, "regen_potion_turns_left", 0)) > 0:
            c.potion_regen_turns_left = int(getattr(run, "regen_potion_turns_left", 0))
            c.potion_regen_amount = max(1, int(getattr(run, "regen_potion_amount", 10)))
            run.regen_potion_turns_left = 0
            c.log.append(f"再生劑作用中：剩餘 {c.potion_regen_turns_left} 回合，每回合回復 {c.potion_regen_amount} HP。")

    def start_combat(enemy: Enemy, reward_tier: Optional[str] = None) -> None:
        nonlocal pending_combat_reward_tier
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

        def enter_combat_scene() -> None:
            nonlocal combat_scene, scene
            c = CombatState()
            c.player_max_hp = run.max_hp
            c.player_hp = run.hp
            if hasattr(c, "gold"):
                c.gold = run.gold
            c.start(combat_enemy)
            apply_potion_battle_buffs(c, combat_enemy)
            c.sigils = [s for s in sigils_all if s.sigil_id in run.equipped_sigils]
            if hasattr(c, "refresh_sigil_state"):
                c.refresh_sigil_state()
            combat_scene = CombatScene(run, c, sigils_all, font, font_big, font_huge)
            scene = "combat"

        begin_black_transition(enter_combat_scene)

    def finish_combat_and_return() -> None:
        nonlocal scene, combat_scene, pending_combat_reward_tier
        if combat_scene is None:
            return
        enemy = combat_scene.combat.enemy
        run.hp = combat_scene.combat.player_hp
        if hasattr(combat_scene.combat, "gold"):
            run.gold = combat_scene.combat.gold
        if hasattr(combat_scene.combat, "potion_regen_turns_left"):
            run.regen_potion_turns_left = int(getattr(combat_scene.combat, "potion_regen_turns_left", 0))
            run.regen_potion_amount = int(getattr(combat_scene.combat, "potion_regen_amount", getattr(run, "regen_potion_amount", 10)))

        node = run.map_graph.nodes[run.current_node_id]
        is_training = enemy is not None and getattr(enemy, "enemy_id", "") == "training_dummy"
        if (not is_training) and enemy and enemy.hp <= 0:
            node.cleared = True
            base_reward = 0
            if pending_combat_reward_tier == "normal":
                base_reward = 20
            elif pending_combat_reward_tier == "elite":
                base_reward = 40
            elif node.node_type == NodeType.COMBAT:
                base_reward = 20
            elif node.node_type == NodeType.ELITE:
                base_reward = 40
            elif node.node_type == NodeType.BOSS:
                base_reward = 100

            reward = base_reward
            gold_bonus_percent = max(0, int(getattr(combat_scene.combat, "potion_gold_bonus_percent", 0)))
            if base_reward > 0 and gold_bonus_percent > 0:
                reward += int(round(base_reward * gold_bonus_percent / 100.0))
            run.gold += reward
            if reward > 0:
                if gold_bonus_percent > 0:
                    map_scene.msg = f"戰鬥勝利：獲得 {reward} 金幣（貪婪藥水加成）。"
                else:
                    map_scene.msg = f"戰鬥勝利：獲得 {reward} 金幣。"
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

            elif effect_type == "gain_specific_sigil":
                sigil_id = str(effect.get("sigil_id", ""))
                sigil = next((s for s in sigils_all if s.sigil_id == sigil_id), None)
                if sigil is None:
                    messages.append("紋章力量消散了")
                elif sigil.sigil_id in run.owned_sigils:
                    run.gold += 30
                    messages.append(f"已擁有紋章「{sigil.name}」，改獲得 30 金幣")
                else:
                    run.owned_sigils.add(sigil.sigil_id)
                    messages.append(f"獲得紋章：「{sigil.name}」")

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
        nonlocal scene, shop_scene, start_scene, start_reward_scene, sigil_choice_scene, pending_sigil_choice_source, pending_sigil_choice_messages
        messages: List[str] = []
        action = apply_event_effects(
            [effect for effect in option.get("effects", []) if isinstance(effect, dict)],
            messages,
        )
        node = run.map_graph.nodes[run.current_node_id]

        start_scene = None
        start_reward_scene = None

        if action == "choose_sigil":
            pending_sigil_choice_source = "start"
            pending_sigil_choice_messages = list(messages)
            sigil_choice_scene = SigilChoiceScene(
                run,
                pending_sigil_choice_choices,
                "起始祝福",
                "你用小刀劃破了手腕，鮮血汩汩流出，在祭壇上沿著鐫刻精美的凹槽流淌，你發現這些紅色的線條形成了一個六角星的圖騰，似乎在召喚著什麼力量。當最後一滴血落下，圖騰發出微弱的光芒，三枚紋章緩緩浮現，懸停在空中，等待你的選擇。",
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

            if opening_transition_active or scene_transition_active:
                continue

            if scene == "opening":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if opening_scene.handle_click(event.pos, sw, sh):
                        opening_transition_active = True
                        opening_transition_elapsed_ms = 0
                        opening_transition_start_scene_created = False

            elif scene == "map":
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
                        start_reward_scene = StartRewardScene(run, option, sigils_all, font, font_big)
                        start_scene = None
                        scene = "start_reward"

            elif scene == "start_reward":
                if start_reward_scene is None:
                    scene = "map"
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    start_reward_scene.handle_click(event.pos)

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
                    begin_black_transition(finish_combat_and_return)

        dt_ms = clock.tick(60)

        if opening_transition_active:
            opening_transition_elapsed_ms += dt_ms
            opening_transition_switch_ms = opening_transition_fade_out_ms + opening_transition_hold_ms
            opening_transition_total_ms = opening_transition_switch_ms + opening_transition_fade_in_ms

            if (
                not opening_transition_start_scene_created
                and opening_transition_elapsed_ms >= opening_transition_switch_ms
            ):
                start_scene = StartScene(run, choose_start_options(), font, font_big)
                scene = "start"
                opening_transition_start_scene_created = True

            if opening_transition_elapsed_ms >= opening_transition_total_ms:
                if not opening_transition_start_scene_created:
                    start_scene = StartScene(run, choose_start_options(), font, font_big)
                    scene = "start"
                    opening_transition_start_scene_created = True
                opening_transition_active = False

        if scene_transition_active:
            scene_transition_elapsed_ms += dt_ms
            scene_transition_total_ms = scene_transition_fade_out_ms + scene_transition_fade_in_ms
            if (
                not scene_transition_switch_done
                and scene_transition_elapsed_ms >= scene_transition_fade_out_ms
            ):
                if scene_transition_action is not None:
                    scene_transition_action()
                scene_transition_switch_done = True

            if scene_transition_elapsed_ms >= scene_transition_total_ms:
                scene_transition_active = False
                scene_transition_action = None

        if scene == "combat" and combat_scene is not None:
            combat_scene.update(dt_ms)
        if scene == "start" and start_scene is not None and not opening_transition_active:
            start_scene.update(dt_ms)
        if scene == "start_reward" and start_reward_scene is not None:
            start_reward_scene.update(dt_ms)
            picked_option = start_reward_scene.take_finished_option()
            if picked_option is not None and not scene_transition_active:
                def finish_start_reward(option: Dict[str, Any] = picked_option) -> None:
                    nonlocal start_reward_scene
                    start_reward_scene = None
                    apply_start_option(option)

                begin_black_transition(finish_start_reward)

        if scene == "opening":
            opening_scene.render(screen, sw, sh)
        elif scene == "map":
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

        elif scene == "start_reward":
            if start_reward_scene is None:
                scene = "map"
                map_scene.render(screen, sw, sh)
            else:
                start_reward_scene.render(screen, sw, sh)

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

        if opening_transition_active:
            opening_transition_switch_ms = opening_transition_fade_out_ms + opening_transition_hold_ms
            opening_transition_total_ms = opening_transition_switch_ms + opening_transition_fade_in_ms
            if opening_transition_elapsed_ms < opening_transition_fade_out_ms:
                fade_progress = opening_transition_elapsed_ms / max(1, opening_transition_fade_out_ms)
                fade_alpha = int(255 * clamp_float(fade_progress, 0.0, 1.0))
            elif opening_transition_elapsed_ms < opening_transition_switch_ms:
                fade_alpha = 255
            else:
                fade_progress = (
                    opening_transition_elapsed_ms - opening_transition_switch_ms
                ) / max(1, opening_transition_fade_in_ms)
                fade_alpha = int(255 * (1.0 - clamp_float(fade_progress, 0.0, 1.0)))

            if fade_alpha > 0:
                fade = pygame.Surface((sw, sh), pygame.SRCALPHA)
                fade.fill((0, 0, 0, max(0, min(255, fade_alpha))))
                screen.blit(fade, (0, 0))

        if scene_transition_active:
            if scene_transition_elapsed_ms < scene_transition_fade_out_ms:
                fade_progress = scene_transition_elapsed_ms / max(1, scene_transition_fade_out_ms)
                fade_alpha = int(255 * clamp_float(fade_progress, 0.0, 1.0))
            else:
                fade_progress = (scene_transition_elapsed_ms - scene_transition_fade_out_ms) / max(1, scene_transition_fade_in_ms)
                fade_alpha = int(255 * (1.0 - clamp_float(fade_progress, 0.0, 1.0)))

            if fade_alpha > 0:
                fade = pygame.Surface((sw, sh), pygame.SRCALPHA)
                fade.fill((0, 0, 0, max(0, min(255, fade_alpha))))
                screen.blit(fade, (0, 0))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
