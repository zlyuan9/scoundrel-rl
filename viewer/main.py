"""
Retro RPG–style viewer for Scoundrel. Run: python -m viewer
Requires: pip install -e ".[gui]"
"""

from __future__ import annotations

import math
import random
import sys
from typing import Literal

try:
    import pygame
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pygame is required for the viewer. Install with: pip install -e \".[gui]\""
    ) from e

from scoundrel.engine import Engine
from scoundrel.card import Card

# --- Retro palette (muted dungeon / CRT-adjacent) ---
C_BG = (22, 18, 32)
C_PANEL = (42, 36, 58)
C_PANEL_DARK = (28, 24, 40)
C_BORDER = (201, 162, 39)
C_BORDER_DIM = (120, 98, 48)
C_TEXT = (232, 224, 208)
C_TEXT_DIM = (160, 150, 130)
C_ACCENT = (120, 200, 140)
C_DANGER = (200, 90, 90)
C_MONSTER = (140, 170, 120)
C_WEAPON = (110, 180, 220)
C_POTION = (220, 130, 130)
C_GOLD = (212, 175, 55)

Phase = Literal["turn", "room", "over"]

W, H = 960, 640
FPS = 60


def infer_phase(engine: Engine) -> Phase:
    """Derive UI phase from engine state (matches interactive play flow)."""
    if engine.game_won or engine.game_lost:
        return "over"
    if engine._resolves_left > 0:
        return "room"
    return "turn"

# Layout (keeps text inside bordered panels)
MARGIN = 24
TITLE_TOP = 16
TITLE_H = 72
STATS_Y = TITLE_TOP + TITLE_H + 6
STATS_H = 44
ROOM_Y = STATS_Y + STATS_H + 10
ROOM_H = 260
MSG_GAP = 10
MSG_Y = ROOM_Y + ROOM_H + MSG_GAP
MSG_H = 100
BTN_TOP = 530


def _wrap_lines(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    """Split text into lines that fit within max_width when rendered with font."""
    text = text.strip()
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        trial = f"{current} {w}"
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def _blit_wrapped(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
    max_width: int,
    line_spacing: int | None = None,
) -> int:
    """Draw wrapped lines; returns total height used."""
    ls = line_spacing if line_spacing is not None else font.get_linesize()
    lines = _wrap_lines(font, text, max_width)
    yy = y
    for line in lines:
        surf = font.render(line, True, color)
        screen.blit(surf, (x, yy))
        yy += ls
    return yy - y


def rank_label(value: int) -> str:
    if value <= 10:
        return str(value)
    return {11: "J", 12: "Q", 13: "K", 14: "A"}[value]


def suit_symbol(suit: str) -> str:
    return {"spades": "\u2660", "clubs": "\u2663", "diamonds": "\u2666", "hearts": "\u2665"}.get(
        suit, "?"
    )


def card_type_label(card: Card) -> str:
    if card.type == "monster":
        return "FOE"
    if card.type == "weapon":
        return "WEA"
    return "HP"


def card_colors(card: Card) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if card.type == "monster":
        return (C_MONSTER, C_PANEL_DARK)
    if card.type == "weapon":
        return (C_WEAPON, C_PANEL_DARK)
    return (C_POTION, C_PANEL_DARK)


def room_card_row_start_y(room_r: pygame.Rect, font: pygame.font.Font, deck_count: int) -> int:
    """Y position where the four card slots begin (below ROOM / dungeon header)."""
    lab = font.render("ROOM", True, C_ACCENT)
    deck_t = font.render(f"Dungeon cards: {deck_count}", True, C_TEXT)
    y1 = room_r.y + 8
    deck_w = deck_t.get_width()
    right_x = room_r.right - deck_w - 14
    if right_x >= room_r.x + 12 + lab.get_width() + 16:
        return y1 + font.get_linesize() + 12
    return y1 + 2 * font.get_linesize() + 18


class GameView:
    def __init__(self, *, surface: pygame.Surface | None = None) -> None:
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Courier New", 36, bold=True)
        self.font = pygame.font.SysFont("Courier New", 18)
        self.font_small = pygame.font.SysFont("Courier New", 15)
        self.font_big = pygame.font.SysFont("Courier New", 22, bold=True)

        self.engine: Engine | None = None
        self.phase: Phase = "turn"
        self.message = "Welcome, adventurer."
        self.log: list[str] = []

        if surface is not None:
            if surface.get_size() != (W, H):
                raise ValueError(f"surface must be {W}x{H}, got {surface.get_size()}")
            self.screen = surface
            self._owns_window = False
        else:
            pygame.display.set_caption("Scoundrel — Dungeon")
            self.screen = pygame.display.set_mode((W, H))
            self._owns_window = True

    def render_engine(
        self,
        engine: Engine,
        *,
        phase: Phase | None = None,
        message: str = "",
        log: list[str] | None = None,
    ) -> None:
        """Draw the given engine state to ``self.screen`` (same layout as the interactive viewer)."""
        self.engine = engine
        self.phase = phase if phase is not None else infer_phase(engine)
        self.message = message or "—"
        self.log = (log or [])[-5:]
        skip_r = pygame.Rect(80, 530, 300, 48)
        play_r = pygame.Rect(W - 380, 530, 300, 48)
        self.screen.fill(C_BG)
        for y in range(0, H, 4):
            pygame.draw.line(self.screen, (12, 10, 18), (0, y), (W, y), 1)
        self.draw_ui(skip_r, play_r)

    def log_push(self, text: str) -> None:
        self.log.append(text)
        self.log = self.log[-5:]
        self.message = text

    def new_game(self) -> None:
        random.seed()
        self.engine = Engine()
        self.phase = "turn"
        self.log_push("A new dungeon is shuffled.")
        if self.engine.start_turn():
            self.phase = "over"
            self.log_push("The dungeon was empty. Victory?")

    def draw_frame_border(self, rect: pygame.Rect, bright: bool = True) -> None:
        c = C_BORDER if bright else C_BORDER_DIM
        pygame.draw.rect(self.screen, c, rect, 3)
        # corner ticks (retro panel feel)
        t = 8
        for x, y in (
            (rect.left, rect.top),
            (rect.right - 1, rect.top),
            (rect.left, rect.bottom - 1),
            (rect.right - 1, rect.bottom - 1),
        ):
            pygame.draw.line(self.screen, c, (x, y), (x + t, y), 2)
            pygame.draw.line(self.screen, c, (x, y), (x, y + t), 2)

    def draw_hp_bar(self, x: int, y: int, w: int, h: int, current: int, max_hp: int) -> None:
        pygame.draw.rect(self.screen, C_PANEL_DARK, (x, y, w, h))
        frac = max(0.0, min(1.0, current / max_hp))
        fill_w = int((w - 4) * frac)
        pygame.draw.rect(self.screen, (60, 120, 70), (x + 2, y + 2, fill_w, h - 4))
        pygame.draw.rect(self.screen, C_BORDER_DIM, (x, y, w, h), 2)
        t = self.font.render(f"HP {current}/{max_hp}", True, C_TEXT)
        self.screen.blit(t, (x + 10, y + (h - t.get_height()) // 2))

    def draw_weapon(self, x: int, y: int, w: int, h: int) -> None:
        assert self.engine is not None
        pygame.draw.rect(self.screen, C_PANEL_DARK, (x, y, w, h))
        pygame.draw.rect(self.screen, C_BORDER_DIM, (x, y, w, h), 2)
        wcard, last = self.engine.weapon
        if wcard is None:
            txt = "Weapon: (bare hands)"
            col = C_TEXT_DIM
        else:
            ls = "\u221e" if math.isinf(last) else str(int(last))
            txt = f"Weapon: {rank_label(wcard.value)}{suit_symbol(wcard.suit)}  chain≤{ls}"
            col = C_WEAPON
        pad_x = 10
        max_w = max(40, w - 2 * pad_x)
        lines = _wrap_lines(self.font, txt, max_w)
        line_h = self.font.get_linesize()
        total_h = len(lines) * line_h
        y0 = y + max(0, (h - total_h) // 2)
        yy = y0
        for line in lines:
            surf = self.font.render(line, True, col)
            self.screen.blit(surf, (x + pad_x, yy))
            yy += line_h

    def draw_card_slot(
        self, rect: pygame.Rect, card: Card | None, index: int, highlight: bool
    ) -> None:
        if card is None:
            pygame.draw.rect(self.screen, C_PANEL_DARK, rect)
            pygame.draw.rect(self.screen, C_BORDER_DIM, rect, 2)
            t = self.font.render(f"[{index}] —", True, C_TEXT_DIM)
            self.screen.blit(t, (rect.x + 12, rect.centery - t.get_height() // 2))
            return

        fg, bg = card_colors(card)
        pygame.draw.rect(self.screen, bg, rect)
        if highlight:
            pygame.draw.rect(self.screen, C_GOLD, rect, 4)
        else:
            pygame.draw.rect(self.screen, C_BORDER_DIM, rect, 2)

        type_l = card_type_label(card)
        t1 = self.font_big.render(type_l, True, fg)
        t2 = self.font.render(
            f"{rank_label(card.value)} {suit_symbol(card.suit)}", True, C_TEXT
        )
        self.screen.blit(t1, (rect.x + 10, rect.y + 10))
        self.screen.blit(t2, (rect.x + 10, rect.y + 40))

    def draw_ui(self, skip_rect: pygame.Rect, play_rect: pygame.Rect) -> None:
        assert self.engine is not None
        cw = W - 2 * MARGIN

        # Title — vertically centered block inside panel
        title_r = pygame.Rect(MARGIN, TITLE_TOP, cw, TITLE_H)
        pygame.draw.rect(self.screen, C_PANEL, title_r)
        self.draw_frame_border(title_r)
        title = self.font_title.render("SCOUNDREL", True, C_GOLD)
        sub = self.font_small.render("— solo dungeon —", True, C_TEXT_DIM)
        block_h = title.get_height() + 6 + sub.get_height()
        ty = title_r.y + (title_r.h - block_h) // 2
        self.screen.blit(title, (title_r.centerx - title.get_width() // 2, ty))
        self.screen.blit(
            sub,
            (title_r.centerx - sub.get_width() // 2, ty + title.get_height() + 6),
        )

        # Stats row (weapon strip spans from x=320 to right margin)
        self.draw_hp_bar(MARGIN, STATS_Y, 280, STATS_H, self.engine.health, 20)
        weapon_w = W - MARGIN - 320
        self.draw_weapon(320, STATS_Y, weapon_w, STATS_H)

        # Room panel — header row keeps labels inside; cards start below
        room_r = pygame.Rect(MARGIN, ROOM_Y, cw, ROOM_H)
        pygame.draw.rect(self.screen, C_PANEL, room_r)
        self.draw_frame_border(room_r, bright=True)
        lab = self.font.render("ROOM", True, C_ACCENT)
        deck_t = self.font.render(f"Dungeon cards: {len(self.engine.deck)}", True, C_TEXT)
        y1 = room_r.y + 8
        deck_w = deck_t.get_width()
        right_x = room_r.right - deck_w - 14
        self.screen.blit(lab, (room_r.x + 12, y1))
        if right_x >= room_r.x + 12 + lab.get_width() + 16:
            self.screen.blit(deck_t, (right_x, y1))
        else:
            self.screen.blit(deck_t, (room_r.x + 12, y1 + self.font.get_linesize() + 4))

        slot_w, slot_h = 200, 180
        gap = 18
        start_x = room_r.x + (room_r.w - (4 * slot_w + 3 * gap)) // 2
        sy = room_card_row_start_y(room_r, self.font, len(self.engine.deck))
        for i in range(4):
            r = pygame.Rect(start_x + i * (slot_w + gap), sy, slot_w, slot_h)
            card = self.engine.hand[i]
            highlight = self.phase == "room" and card is not None
            self.draw_card_slot(r, card, i, highlight)

        # Message / log — wrapped to panel width; clip so text stays inside border
        msg_r = pygame.Rect(MARGIN, MSG_Y, cw, MSG_H)
        pygame.draw.rect(self.screen, C_PANEL_DARK, msg_r)
        pygame.draw.rect(self.screen, C_BORDER_DIM, msg_r, 2)
        inner_w = msg_r.w - 24
        clip_inner = msg_r.inflate(-8, -8)
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(clip_inner)
        try:
            my = msg_r.y + 10
            my += _blit_wrapped(
                self.screen, self.font, self.message, C_TEXT, msg_r.x + 12, my, inner_w
            )
            my += 6
            for old in self.log[-3:]:
                if old == self.message:
                    continue
                my += _blit_wrapped(
                    self.screen, self.font_small, old, C_TEXT_DIM, msg_r.x + 12, my, inner_w
                )
                my += 4
        finally:
            self.screen.set_clip(prev_clip)

        # Buttons (turn phase) — text centered in each button rect
        if self.phase == "turn":
            pygame.draw.rect(self.screen, C_PANEL, skip_rect)
            pygame.draw.rect(self.screen, C_BORDER, skip_rect, 2)
            pygame.draw.rect(self.screen, C_PANEL, play_rect)
            pygame.draw.rect(self.screen, C_BORDER, play_rect, 2)
            s = self.font_big.render("SKIP ROOM", True, C_TEXT)
            p = self.font_big.render("ENTER ROOM", True, C_ACCENT)
            self.screen.blit(
                s,
                (
                    skip_rect.centerx - s.get_width() // 2,
                    skip_rect.centery - s.get_height() // 2,
                ),
            )
            self.screen.blit(
                p,
                (
                    play_rect.centerx - p.get_width() // 2,
                    play_rect.centery - p.get_height() // 2,
                ),
            )

        if self.phase == "room":
            hint_txt = f"Choose {self.engine._resolves_left} more card(s) to resolve."
            _blit_wrapped(self.screen, self.font, hint_txt, C_GOLD, MARGIN, BTN_TOP - 28, cw)

        # Footer — wrapped; anchor block so it sits above the bottom edge
        foot_str = (
            "[R] new game   [ESC] quit   "
            "Rules: resolve 3 of 4; fourth stays for next room."
        )
        foot_ls = self.font_small.get_linesize() + 2
        foot_lines = _wrap_lines(self.font_small, foot_str, cw)
        foot_h = len(foot_lines) * foot_ls
        _blit_wrapped(
            self.screen,
            self.font_small,
            foot_str,
            C_TEXT_DIM,
            MARGIN,
            H - foot_h - MARGIN // 2,
            cw,
            line_spacing=foot_ls,
        )

        if self.phase == "over":
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((10, 8, 16, 210))
            self.screen.blit(ov, (0, 0))
            outcome = "VICTORY" if self.engine and self.engine.game_won else "DEFEAT"
            col = C_ACCENT if outcome == "VICTORY" else C_DANGER
            big = self.font_title.render(outcome, True, col)
            self.screen.blit(big, (W // 2 - big.get_width() // 2, H // 2 - 40))
            small = self.font.render("Press [R] to play again", True, C_TEXT)
            self.screen.blit(small, (W // 2 - small.get_width() // 2, H // 2 + 20))

    def handle_click_turn(self, pos: tuple[int, int], skip_r: pygame.Rect, play_r: pygame.Rect) -> None:
        assert self.engine is not None
        if skip_r.collidepoint(pos):
            if not self.engine.try_skip():
                self.log_push("You cannot skip two rooms in a row. Face the room!")
                self.engine.begin_play_room()
                self.phase = "room"
            else:
                self.log_push("You avoided the room (cards sent to bottom).")
                if self.engine.start_turn():
                    self.phase = "over"
                    self.log_push("The dungeon is cleared. Victory!")
        elif play_r.collidepoint(pos):
            self.engine.begin_play_room()
            self.phase = "room"
            self.log_push("You enter the room…")

    def handle_click_room(self, pos: tuple[int, int]) -> None:
        assert self.engine is not None
        cw = W - 2 * MARGIN
        room_r = pygame.Rect(MARGIN, ROOM_Y, cw, ROOM_H)
        slot_w, slot_h = 200, 180
        gap = 18
        start_x = room_r.x + (room_r.w - (4 * slot_w + 3 * gap)) // 2
        sy = room_card_row_start_y(room_r, self.font, len(self.engine.deck))
        for i in range(4):
            r = pygame.Rect(start_x + i * (slot_w + gap), sy, slot_w, slot_h)
            if r.collidepoint(pos):
                if self.engine.hand[i] is None:
                    self.log_push("That slot is empty.")
                    return
                result = self.engine.resolve_in_room(i)
                if result == "lost":
                    self.phase = "over"
                    self.log_push("You have fallen. Game over.")
                elif result == "won":
                    self.phase = "over"
                    self.log_push("The dungeon is cleared. Victory!")
                elif result == "room_done":
                    self.phase = "turn"
                    self.log_push("Room cleared. The last card remains as foundation.")
                    if self.engine.start_turn():
                        self.phase = "over"
                        self.log_push("The dungeon is cleared. Victory!")
                else:
                    self.log_push(f"Resolved. ({3 - self.engine._resolves_left}/3)")
                return

    def run(self) -> None:
        self.new_game()
        skip_r = pygame.Rect(80, 530, 300, 48)
        play_r = pygame.Rect(W - 380, 530, 300, 48)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_r:
                        self.new_game()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.phase == "over":
                        continue
                    if self.phase == "turn":
                        self.handle_click_turn(event.pos, skip_r, play_r)
                    elif self.phase == "room":
                        self.handle_click_room(event.pos)

            assert self.engine is not None
            self.render_engine(self.engine, phase=self.phase)

            pygame.display.flip()
            self.clock.tick(FPS)


def main() -> None:
    GameView().run()


if __name__ == "__main__":
    main()
