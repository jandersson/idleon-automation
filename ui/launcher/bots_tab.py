"""Bots tab: one card per minigame (Start/Stop toggle, live status,
persisted options, template coverage, hoops predictor cards) and the
color-coded log pane.

Structure (2026-07-11 refactor): each minigame gets a BotRow component
that owns its widgets and subscribes to the shell's event bus
(bot_started / bot_exited / tick). Mutual exclusion is visible — while a
bot runs, every other row's toggle is disabled instead of failing in the
log. The old tries strip moved to the shell header (visible on all tabs).
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from ui.launcher import config, process, state, theme

# Predictor "cards" — one per kind in the launcher's hoops row.
# Type colors are arbitrary but consistent (each model gets one).
PREDICTOR_CARD_SPECS = [
    ("gp",             "🧠", "GP",     "#8b7bf0"),
    ("make_prob",      "🎲", "P(make)", "#e2566c"),
    ("knn",            "🎯", "KNN",    "#4aa3df"),
    ("bivariate",      "📐", "Biv",    "#34c3b5"),
    ("trajectory_knn", "📍", "T-KNN",  "#ff9f43"),
    ("trajectory_gp",  "🌐", "T-GP",   "#d8a24a"),
    ("trajectory_rf",  "🌲", "T-RF",   "#4cc77a"),
]

# Log-source tag palette for non-bot sources (bots use their own color).
_LOG_FALLBACK_COLORS = [theme.INFO, theme.WARN, "#c792ea", "#89ddff"]


def build(parent: ttk.Frame, app) -> None:
    app.bot_rows: dict[str, BotRow] = {}

    rows_holder = ttk.Frame(parent)
    rows_holder.pack(fill="x", padx=theme.PAD, pady=(theme.PAD, 0))
    for mg in config.MINIGAMES:
        row = BotRow(rows_holder, app, mg)
        app.bot_rows[mg["name"]] = row

    _build_log_pane(parent, app)

    app.on("bot_started", lambda name: _on_bot_started(app, name))
    app.on("bot_exited", lambda name, code: _on_bot_exited(app, name, code))
    app.on("tick", lambda: _on_tick(app))


class BotRow:
    """One minigame's card: accent stripe, name, toggle, status, options."""

    def __init__(self, parent: ttk.Frame, app, mg: dict):
        self.app = app
        self.mg = mg
        self.name = mg["name"]
        self.started_at: float | None = None
        color = theme.bot_color(self.name)

        outer = tk.Frame(parent, bg=theme.SURFACE, highlightthickness=1,
                         highlightbackground=theme.BORDER)
        outer.pack(fill="x", pady=theme.PAD_XS)
        tk.Frame(outer, bg=color, width=4).pack(side="left", fill="y")
        body = ttk.Frame(outer, style="Card.TFrame",
                         padding=(theme.PAD, theme.PAD_S))
        body.pack(side="left", fill="both", expand=True)

        top = ttk.Frame(body, style="Card.TFrame")
        top.pack(fill="x")
        tk.Label(top, text=f"{mg.get('emoji', '')} {self.name.capitalize()}",
                 fg=color, bg=theme.SURFACE, font=theme.FONT_HEAD,
                 width=11, anchor="w").pack(side="left")
        self.toggle_btn = ttk.Button(top, text="▶ Start", width=9,
                                     style="Start.TButton", command=self.toggle)
        self.toggle_btn.pack(side="left", padx=(0, theme.PAD_L))
        self.status = ttk.Label(top, text="● stopped", style="Card.TLabel",
                                foreground=theme.MUTED, font=theme.FONT_SMALL)
        self.status.pack(side="left")

        # Options right-aligned, persisted across launcher restarts.
        for opt in reversed(mg.get("bot_options", [])):
            initial = state.get_option(app.ui_state, self.name, opt["env"],
                                       opt["default"])
            if initial not in opt["values"]:
                initial = opt["default"]
            var = tk.StringVar(value=initial)
            var.trace_add("write", lambda *_a, o=opt, v=var: state.set_option(
                app.ui_state, self.name, o["env"], v.get()))
            app.bot_option_vars[(self.name, opt["env"])] = var
            ttk.Combobox(top, textvariable=var, state="readonly",
                         values=opt["values"],
                         width=max(len(v) for v in opt["values"]) + 2,
                         ).pack(side="right", padx=(0, theme.PAD))
            ttk.Label(top, text=opt["label"], style="CardMuted.TLabel",
                      ).pack(side="right", padx=(theme.PAD_S, theme.PAD_S))

        templates_dir = (config.PROJECT_ROOT / "minigames" / self.name
                         / "assets" / "digit_templates")
        if templates_dir.exists():
            build_templates_row(body, app, self.name)

        # Hoops gets its Pokemon-style predictor stat cards (#21).
        if self.name == "hoops":
            build_predictor_cards(body, app)
            refresh_hoops_stats(app)
            pred_var = app.bot_option_vars.get((self.name, "HOOPS_PREDICTOR_KIND"))
            if pred_var is not None:
                pred_var.trace_add(
                    "write", lambda *_a: highlight_active_predictor(app))
            highlight_active_predictor(app)
            app.on("bot_exited", lambda name, code: (
                refresh_hoops_stats(app) if name == "hoops" else None))

    def toggle(self) -> None:
        if self.app.processes.get(self.name) is None:
            process.start_bot(self.app, self.mg)
        else:
            process.stop_bot(self.app, self.mg)

    def set_running(self, running: bool) -> None:
        if running:
            self.started_at = time.time()
            self.toggle_btn.config(text="■ Stop", style="Stop.TButton",
                                   state="normal")
            self.status.config(text="● running", foreground=theme.SUCCESS)
        else:
            self.started_at = None
            self.toggle_btn.config(text="▶ Start", style="Start.TButton")
            self.status.config(text="● stopped", foreground=theme.MUTED)

    def tick(self) -> None:
        if self.started_at is not None:
            m, s = divmod(int(time.time() - self.started_at), 60)
            self.status.config(text=f"● running · {m}:{s:02d}")


def _on_bot_started(app, name: str) -> None:
    for row_name, row in app.bot_rows.items():
        if row_name == name:
            row.set_running(True)
        else:
            # Mutual exclusion made visible: one mouse, one bot.
            row.toggle_btn.config(state="disabled")


def _on_bot_exited(app, name: str, _code: int) -> None:
    for row_name, row in app.bot_rows.items():
        if row_name == name:
            row.set_running(False)
        row.toggle_btn.config(state="normal")


def _on_tick(app) -> None:
    for row in app.bot_rows.values():
        row.tick()


# ---- log pane ---------------------------------------------------------------

def _build_log_pane(parent: ttk.Frame, app) -> None:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=theme.PAD, pady=theme.PAD)

    header = ttk.Frame(frame)
    header.pack(fill="x", pady=(0, theme.PAD_XS))
    ttk.Label(header, text="Log", style="Muted.TLabel",
              font=theme.FONT_BOLD).pack(side="left")
    ttk.Button(header, text="Clear", style="Ghost.TButton",
               command=lambda: _clear_log(app)).pack(side="right")
    app.log_autoscroll = tk.BooleanVar(value=True)
    ttk.Checkbutton(header, text="autoscroll", variable=app.log_autoscroll,
                    ).pack(side="right", padx=theme.PAD)

    app.log_text = tk.Text(frame, height=10, wrap="none", state="disabled",
                           bg=theme.LOG_BG, fg=theme.TEXT,
                           insertbackground=theme.TEXT, font=theme.MONO,
                           relief="flat", borderwidth=0, padx=6, pady=4)
    scrollbar = ttk.Scrollbar(frame, command=app.log_text.yview)
    app.log_text.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    app.log_text.pack(side="left", fill="both", expand=True)
    app._log_tags: dict[str, str] = {}


def _clear_log(app) -> None:
    app.log_text.config(state="normal")
    app.log_text.delete("1.0", "end")
    app.log_text.config(state="disabled")


def log_tag_for(app, source: str) -> str | None:
    """Text tag (lazily configured) coloring a log line's source prefix.
    Bots get their identity color; other sources cycle a small palette."""
    if not source:
        return None
    tags = getattr(app, "_log_tags", None)
    if tags is None:
        return None
    if source not in tags:
        if source in theme.BOT_COLORS or source in app.processes:
            color = theme.bot_color(source)
        else:
            color = _LOG_FALLBACK_COLORS[len(tags) % len(_LOG_FALLBACK_COLORS)]
        tag = f"src_{source}"
        app.log_text.tag_configure(tag, foreground=color)
        tags[source] = tag
    return tags[source]


# ---- score-template coverage row --------------------------------------------

def build_templates_row(parent: ttk.Frame, app, game_name: str) -> None:
    """Score-template coverage: N/10 status plus (only while incomplete)
    the two ways to add digits — a DB re-scan and a capture-from-game with
    a user-entered current score. Hidden controls reappear on the next
    launcher start if a set drops below 10/10."""
    row = ttk.Frame(parent, style="Card.TFrame")
    row.pack(fill="x", pady=(theme.PAD_S, 0))
    ttk.Label(row, text="score templates", style="CardMuted.TLabel").pack(side="left")
    label = ttk.Label(row, text="—", style="CardMuted.TLabel")
    label.pack(side="left", padx=(theme.PAD_S, theme.PAD))
    app.template_status_labels[game_name] = label
    _captured, missing = template_status(game_name)
    if missing:
        # "Refresh from DB" only supports games whose DB stores OCR'd
        # per-shot scores + post_throw monitor frames (darts/hoops); gate
        # on the bootstrap script's own GAMES so the two can't drift.
        from scripts.bootstrap_digit_templates import GAMES as _DB_REFRESH_GAMES
        if game_name in _DB_REFRESH_GAMES:
            ttk.Button(row, text="Refresh from DB", style="Ghost.TButton",
                       command=lambda g=game_name: refresh_templates(app, g),
                       ).pack(side="left")
        ttk.Label(row, text="capture from game — score now:",
                  style="CardMuted.TLabel").pack(side="left", padx=(theme.PAD_L, theme.PAD_S))
        score_var = tk.StringVar()
        ttk.Entry(row, textvariable=score_var, width=6).pack(side="left")
        ttk.Button(row, text="Capture", style="Ghost.TButton",
                   command=lambda g=game_name, v=score_var:
                   capture_templates_from_game(app, g, v),
                   ).pack(side="left", padx=(theme.PAD_S, 0))
    update_template_label(app, game_name)


def template_status(game_name: str) -> tuple[set[int], set[int]]:
    """(captured_digits, missing_digits) for the game's digit_templates."""
    template_dir = (config.PROJECT_ROOT / "minigames" / game_name
                    / "assets" / "digit_templates")
    captured: set[int] = set()
    if template_dir.exists():
        for d in range(10):
            if (template_dir / f"{d}.png").exists():
                captured.add(d)
    return captured, set(range(10)) - captured


def update_template_label(app, game_name: str) -> None:
    label = app.template_status_labels.get(game_name)
    if label is None:
        return
    captured, missing = template_status(game_name)
    if not missing:
        label.config(text="10/10 ✓", foreground=theme.SUCCESS)
    else:
        missing_str = ", ".join(str(d) for d in sorted(missing))
        label.config(
            text=f"{len(captured)}/10 — missing {missing_str}",
            foreground=theme.WARN if len(captured) >= 7 else theme.DANGER)


def capture_templates_from_game(app, game_name: str, score_var: tk.StringVar) -> None:
    """Grab the live game frame, crop the score region, and save any new
    digit templates assuming the visible score equals the entered value.
    Skips save when the component count mismatches the score's digits
    (stale region pick / wrong screen)."""
    raw = score_var.get().strip()
    try:
        score = int(raw)
        if score < 0:
            raise ValueError("negative score")
    except ValueError:
        app.emit("log", f"[templates:{game_name}] capture: '{raw}' is not a "
                        f"non-negative int\n")
        return
    try:
        import cv2
        from common.capture import grab_region
        from common.regions import get_region
        from common.score_template_ocr import extract_digit_components, save_template
        from common.window import get_bounds, WindowNotFoundError

        mod = __import__(f"minigames.{game_name}.main", fromlist=["WINDOW_TITLE"])
        try:
            left, top, width, height = get_bounds(mod.WINDOW_TITLE)
        except WindowNotFoundError as e:
            app.emit("log", f"[templates:{game_name}] capture: {e}\n")
            return
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        game_dir = config.PROJECT_ROOT / "minigames" / game_name
        region = get_region(game_dir, "score", width, height)
        if region is None:
            app.emit("log", f"[templates:{game_name}] capture: no score region "
                            f"picked (run '{game_name}-pick-score-region')\n")
            return
        crop = bgr[region["top"]:region["top"] + region["height"],
                   region["left"]:region["left"] + region["width"]]
        components = extract_digit_components(crop)
        score_str = str(score)
        if len(components) != len(score_str):
            app.emit("log", f"[templates:{game_name}] capture: {len(components)} "
                            f"components vs {len(score_str)} digits — is the "
                            f"game visible and the region right?\n")
            return
        template_dir = game_dir / "assets" / "digit_templates"
        saved, skipped = [], []
        for (patch, _box), digit_char in zip(components, score_str):
            digit = int(digit_char)
            if (template_dir / f"{digit}.png").exists():
                skipped.append(digit)
                continue
            save_template(template_dir, digit, patch)
            saved.append(digit)
        if saved:
            app.emit("log", f"[templates:{game_name}] capture: saved "
                            f"{', '.join(map(str, saved))}\n")
        if skipped:
            app.emit("log", f"[templates:{game_name}] capture: skipped "
                            f"{', '.join(map(str, skipped))} (already present)\n")
        score_var.set("")
    except Exception as e:
        app.emit("log", f"[templates:{game_name}] capture failed: {e!r}\n")
    update_template_label(app, game_name)


def refresh_templates(app, game_name: str) -> None:
    """Run the digit bootstrap for this game on a background thread,
    routing its stdout into the log, then refresh the row."""
    import contextlib
    import io
    import threading

    def worker() -> None:
        app.emit("log", f"[templates:{game_name}] refreshing...\n")
        try:
            from scripts.bootstrap_digit_templates import bootstrap_game
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bootstrap_game(game_name)
            for line in buf.getvalue().splitlines():
                app.emit("log", f"[templates:{game_name}] {line}\n")
        except Exception as e:
            app.emit("log", f"[templates:{game_name}] failed: {e!r}\n")
        app.root.after(0, lambda: update_template_label(app, game_name))

    threading.Thread(target=worker, daemon=True).start()


# ---- hoops predictor cards ---------------------------------------------------

def build_predictor_cards(parent: ttk.Frame, app) -> None:
    """Horizontal row of stat cards, one per predictor kind."""
    row = ttk.Frame(parent, style="Card.TFrame")
    row.pack(fill="x", pady=(theme.PAD_S, 0))
    for kind, emoji, label, color in PREDICTOR_CARD_SPECS:
        # tk.Frame (not ttk) for per-card background + colored outline.
        card = tk.Frame(row, relief="flat", borderwidth=0, padx=8, pady=5,
                        highlightthickness=2, highlightbackground=theme.BORDER,
                        bg=theme.SURFACE_ALT)
        card.pack(side="left", padx=3)
        tk.Label(card, text=f"{emoji} {label}", fg=color, bg=theme.SURFACE_ALT,
                 font=("Segoe UI", 9, "bold")).pack()
        lv = tk.Label(card, text="Lv 0", fg=theme.MUTED, bg=theme.SURFACE_ALT,
                      font=("Segoe UI", 8))
        lv.pack()
        hp = ttk.Progressbar(card, length=80, mode="determinate", maximum=100)
        hp.pack(pady=(2, 1))
        rate = tk.Label(card, text="0/0", fg=theme.MUTED, bg=theme.SURFACE_ALT,
                        font=("Consolas", 8))
        rate.pack()
        app.predictor_cards[kind] = {"card": card, "lv": lv, "hp": hp,
                                     "rate": rate, "color": color}


def refresh_hoops_stats(app) -> None:
    """Pull per-predictor stats from shots.db and update each card.
    Cheap GROUP BY; runs on launcher start and when a hoops session ends."""
    if not app.predictor_cards:
        return
    import sqlite3
    db_path = config.MINIGAMES_DIR / "hoops" / "assets" / "shots.db"
    per_kind: dict[str, tuple[int, int, int]] = {}
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            rows = con.execute(
                "SELECT predictor_kind, COUNT(DISTINCT session_started), "
                "COUNT(*), SUM(made) "
                "FROM shots WHERE direction='up' AND predictor_kind IS NOT NULL "
                "GROUP BY predictor_kind"
            ).fetchall()
            con.close()
            for kind, sessions, shots, makes in rows:
                per_kind[kind] = (int(sessions), int(shots), int(makes or 0))
        except sqlite3.Error:
            pass  # leave cards in their default state
    for kind, widgets in app.predictor_cards.items():
        sessions, shots, makes = per_kind.get(kind, (0, 0, 0))
        rate_pct = int(round(100 * makes / shots)) if shots else 0
        widgets["lv"].config(text=f"Lv {sessions}")
        widgets["hp"]["value"] = rate_pct
        if shots:
            widgets["rate"].config(text=f"{makes}/{shots}  {rate_pct}%",
                                   fg=theme.TEXT)
        else:
            widgets["rate"].config(text="—", fg=theme.MUTED)


def highlight_active_predictor(app) -> None:
    """Outline the card matching the Predictor dropdown selection."""
    if not app.predictor_cards:
        return
    var = app.bot_option_vars.get(("hoops", "HOOPS_PREDICTOR_KIND"))
    active = var.get() if var is not None else ""
    for kind, widgets in app.predictor_cards.items():
        widgets["card"].config(
            highlightbackground=widgets["color"] if kind == active else theme.BORDER,
            highlightthickness=2)
