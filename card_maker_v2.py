from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image


CARD_SIZE = (300, 420)
SUIT_CODES = ("A", "B", "C", "D")

RANK_ORDERS = {
    "2_to_A": ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"),
    "A_to_K": ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"),
}

RANK_BOX = (76, 82)
SUIT_BOX = (95, 95)

RANK_LEFT = 18
SUIT_RIGHT = 14
BOTTOM_MARGIN = 14


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates: Iterable[Path] = (start, *start.parents)
    for path in candidates:
        if (path / "game" / "ui" / "assets").exists():
            return path
    return Path.cwd().resolve()


def load_rgba(
    path: Path,
    remove_near_black: bool = False,
    remove_near_white: bool = False,
) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    if remove_near_black or remove_near_white:
        px = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]
                if not a:
                    continue
                if remove_near_black and max(r, g, b) <= 18:
                    px[x, y] = (r, g, b, 0)
                elif remove_near_white and min(r, g, b) >= 245:
                    px[x, y] = (r, g, b, 0)
    return img


def fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    new_size = (round(src_w * scale), round(src_h * scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    left = (resized.width - dst_w) // 2
    top = (resized.height - dst_h) // 2
    return resized.crop((left, top, left + dst_w, top + dst_h))


def fit_inside(img: Image.Image, box_size: tuple[int, int]) -> Image.Image:
    box_w, box_h = box_size
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def paste_alpha(base: Image.Image, overlay: Image.Image, xy: tuple[int, int]) -> None:
    base.alpha_composite(overlay, xy)


def compose_card(
    spla_path: Path,
    num_path: Path,
    suit_path: Path,
    output_path: Path,
    remove_near_black: bool = False,
    remove_near_white: bool = False,
) -> None:
    card = fit_cover(load_rgba(spla_path), CARD_SIZE)

    num = fit_inside(load_rgba(num_path, remove_near_black, remove_near_white), RANK_BOX)
    suit = fit_inside(load_rgba(suit_path, remove_near_black), SUIT_BOX)

    num_x = RANK_LEFT
    num_y = CARD_SIZE[1] - BOTTOM_MARGIN - num.height

    suit_x = CARD_SIZE[0] - SUIT_RIGHT - suit.width
    suit_y = CARD_SIZE[1] - BOTTOM_MARGIN - suit.height

    paste_alpha(card, num, (num_x, num_y))
    paste_alpha(card, suit, (suit_x, suit_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--rank-order", choices=RANK_ORDERS.keys(), default="2_to_A")
    parser.add_argument("--remove-near-black", action="store_true")
    parser.add_argument("--remove-near-white", action="store_true")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    project_root = args.project_root.resolve() if args.project_root else find_project_root(script_root)

    src_dir = project_root / "game" / "ui" / "assets" / "card_making"
    out_dir = project_root / "game" / "ui" / "assets" / "cards"

    if not src_dir.exists():
        raise FileNotFoundError(f"找不到素材資料夾：{src_dir}")

    rank_labels = RANK_ORDERS[args.rank_order]
    made = 0
    skipped: list[str] = []

    for idx, rank_label in enumerate(rank_labels, start=1):
        spla_path = src_dir / f"SPLA{idx}.png"
        num_path = src_dir / f"NUM{idx}.png"

        if not spla_path.exists():
            skipped.append(f"缺少 {spla_path.name}")
            continue
        if not num_path.exists():
            skipped.append(f"缺少 {num_path.name}")
            continue

        for suit_code in SUIT_CODES:
            suit_path = src_dir / f"{suit_code}.png"
            if not suit_path.exists():
                skipped.append(f"缺少 {suit_path.name}")
                continue

            output_path = out_dir / f"{rank_label}_{suit_code}.png"
            compose_card(
                spla_path=spla_path,
                num_path=num_path,
                suit_path=suit_path,
                output_path=output_path,
                remove_near_black=args.remove_near_black,
                remove_near_white=args.remove_near_white,
            )
            made += 1

    print(f"完成輸出 {made} 張卡牌 → {out_dir}")
    if skipped:
        print("略過項目：")
        for item in sorted(set(skipped)):
            print(f"- {item}")


if __name__ == "__main__":
    main()
