"""Bots tab: per-minigame Start/Stop, options, score-template coverage,
the shared tries strip + hoops cooldown countdown, and the hoops predictor
stat cards. Also builds the log pane that lives at the bottom of this tab
(the app shell's poll loop appends to app.log_text).

Functions take the Launcher app as their first argument; widget refs and
state are stored back on the app so the shell's poll loop and the process
machinery can reach them.
"""
import tkinter as tk
from tkinter import ttk

from common import tries_counter
from common.hoops_cooldown_observer import record_observation as _record_hoops_cooldown_observation
from common.idleon_save import read_minigame_plays_shared, read_hoops_cooldown
from ui.launcher import config, process, theme

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


def build(parent: ttk.Frame, app) -> None:
    # Top strip: shared "global tries" counter (chopping + catching share
    # one tries pool in-game; user updates manually since it isn't easily
    # OCR-able — only visible on a pre-game popup).
    build_tries_strip(parent, app).grid(
        row=0, column=0, sticky="ew", padx=8, pady=(8, 0)
    )

    for i, mg in enumerate(config.MINIGAMES):
        label = f"{mg.get('emoji', '')} {mg['name'].capitalize()}".strip()
        frame = ttk.LabelFrame(parent, text=label, padding=6)
        frame.grid(row=i + 1, column=0, sticky="ew", padx=8, pady=4)

        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Button(top, text="▶ Start", width=9, style="Start.TButton",
                   command=lambda m=mg: process.start_bot(app, m)).pack(side="left")
        ttk.Button(top, text="■ Stop", width=9, style="Stop.TButton",
                   command=lambda m=mg: process.stop_bot(app, m)).pack(side="left", padx=(6, 12))
        status = ttk.Label(top, text="stopped", style="Status.TLabel")
        status.pack(side="left")
        app.status_labels[mg["name"]] = status

        for opt in mg.get("bot_options", []):
            var = tk.StringVar(value=opt["default"])
            app.bot_option_vars[(mg["name"], opt["env"])] = var
            ttk.Label(top, text=opt["label"] + ":").pack(side="left", padx=(12, 4))
            ttk.Combobox(
                top, textvariable=var, state="readonly",
                values=opt["values"], width=max(len(v) for v in opt["values"]) + 2,
            ).pack(side="left")

        # Score-template coverage row — auto-appears for games that
        # have a digit_templates/ directory under assets. Lets the
        # user see which digit glyphs are still missing from the
        # template library and re-run the bootstrap on demand
        # (catches new digits from recent sessions).
        templates_dir = config.PROJECT_ROOT / "minigames" / mg["name"] / "assets" / "digit_templates"
        if templates_dir.exists():
            build_templates_row(frame, app, mg["name"])

        # Hoops gets a Pokemon-style stat-card row, one per predictor
        # — see GitHub issue #21. Each card shows Lv (sessions),
        # HP-bar (make rate), and shots/makes. Active predictor
        # (matching the dropdown) is highlighted. Other minigames
        # don't have a comparable signal yet.
        if mg["name"] == "hoops":
            build_predictor_cards(frame, app)
            refresh_hoops_stats(app)
            # Re-highlight the active card whenever the dropdown changes.
            pred_var = app.bot_option_vars.get((mg["name"], "HOOPS_PREDICTOR_KIND"))
            if pred_var is not None:
                pred_var.trace_add("write", lambda *_a: highlight_active_predictor(app))
            highlight_active_predictor(app)

    log_frame = ttk.LabelFrame(parent, text="Log", padding=4)
    log_frame.grid(row=len(config.MINIGAMES) + 1, column=0, sticky="nsew", padx=8, pady=(4, 8))
    app.log_text = tk.Text(log_frame, height=10, wrap="none", state="disabled",
                           bg=theme.LOG_BG, fg=theme.TEXT, insertbackground=theme.TEXT,
                           font=theme.MONO, relief="flat", borderwidth=0,
                           padx=6, pady=4)
    scrollbar = ttk.Scrollbar(log_frame, command=app.log_text.yview)
    app.log_text.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    app.log_text.pack(side="left", fill="both", expand=True)

    parent.columnconfigure(0, weight=1)
    # Log row is len(MINIGAMES) + 1 (tries strip is row 0).
    parent.rowconfigure(len(config.MINIGAMES) + 1, weight=1)


def build_tries_strip(parent: ttk.Frame, app) -> ttk.Frame:
    """Strip showing the per-character minigame plays remaining
    (chopping + catching + mining share this counter in-game) plus
    the next-playable hoops countdown. Auto-reads from the local
    Idleon save on launcher open; Refresh re-reads on demand."""
    strip = ttk.Frame(parent)
    ttk.Label(strip, text="🪓 / 🪰 / ⛏️ Tries:").pack(side="left")

    app.tries_label = ttk.Label(
        strip, text="—", font=("Segoe UI", 10, "bold"), foreground=theme.ACCENT,
    )
    app.tries_label.pack(side="left", padx=(6, 12))

    ttk.Button(strip, text="Refresh", width=8,
               command=lambda: refresh_tries(app)).pack(side="left")

    # 🚧 = save-flush timing makes this unreliable; see GitHub #22.
    ttk.Label(strip, text="🏀 Cooldown 🚧:").pack(side="left", padx=(16, 4))
    app.hoops_label = ttk.Label(
        strip, text="—", font=("Segoe UI", 10, "bold"), foreground=theme.ACCENT,
    )
    app.hoops_label.pack(side="left")

    # Read once at startup so values are visible without a manual click.
    refresh_tries(app, silent=True)
    # Tick the hoops countdown every second so it stays live between
    # save flushes.
    tick_hoops_display(app)
    return strip


def build_templates_row(parent: ttk.Frame, app, game_name: str) -> None:
    """Build the score-template coverage row for a game.

    Always shows the N/10 status (a complete library reads "all 10
    captured ✓"). While digits are still missing, also shows two ways to
    add them (hidden once complete — see the build-time check below):
      - Refresh templates: scan existing monitor frames + manual
        labels for digits the OCR already could read
      - Capture from current game: grab the live game window, crop
        the score region, and extract digit templates assuming the
        current visible score equals what the user typed in the
        adjacent field. Use this to add missing digits without
        waiting for the bot to encounter them organically (e.g.
        digit 0 only appears at session start, before any throw is
        logged to the DB).
    """
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(4, 0))
    ttk.Label(row, text="📋 Score templates:").pack(side="left")
    label = ttk.Label(row, text="—")
    label.pack(side="left", padx=(4, 8))
    app.template_status_labels[game_name] = label
    # The Refresh/Capture controls only matter while digits are still
    # missing. A complete library (10/10 — hoops, darts) shows just the
    # "all 10 captured ✓" status; the buttons would be no-ops there (and the
    # blob-guard removed the only reason to re-run on a complete set). Mining
    # (5/10) and any future incomplete game still get the full controls.
    # Evaluated at build time; if a set later drops below 10/10 the controls
    # reappear on the next launcher start.
    _captured, missing = template_status(game_name)
    if missing:
        ttk.Button(
            row, text="Refresh from DB", width=15,
            command=lambda g=game_name: refresh_templates(app, g),
        ).pack(side="left")
        ttk.Label(row, text="  Capture from game — score now:").pack(side="left", padx=(12, 4))
        score_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=score_var, width=6)
        entry.pack(side="left")
        ttk.Button(
            row, text="Capture", width=8,
            command=lambda g=game_name, v=score_var: capture_templates_from_game(app, g, v),
        ).pack(side="left", padx=(4, 0))
    update_template_label(app, game_name)


def template_status(game_name: str) -> tuple[set[int], set[int]]:
    """Return (captured_digits, missing_digits) for the game's
    digit_templates directory."""
    template_dir = config.PROJECT_ROOT / "minigames" / game_name / "assets" / "digit_templates"
    captured: set[int] = set()
    if template_dir.exists():
        for d in range(10):
            if (template_dir / f"{d}.png").exists():
                captured.add(d)
    return captured, set(range(10)) - captured


def update_template_label(app, game_name: str) -> None:
    """Refresh the displayed status text for one game's templates row."""
    label = app.template_status_labels.get(game_name)
    if label is None:
        return
    captured, missing = template_status(game_name)
    if not missing:
        text = "all 10 captured ✓"
        color = theme.SUCCESS
    else:
        missing_str = ", ".join(str(d) for d in sorted(missing))
        text = f"{len(captured)}/10 captured — missing {missing_str}"
        color = theme.WARN if len(captured) >= 7 else theme.DANGER
    label.config(text=text, foreground=color)


def capture_templates_from_game(app, game_name: str, score_var: tk.StringVar) -> None:
    """Grab the live game frame, crop the score region, extract digit
    components, and save any new templates assuming the digits line up
    with the user-entered score. Done on the main thread (fast — single
    grab + a few mss/cv2 ops). Skips save if the component count doesn't
    match the score's digit count (which would mean a stale region pick
    or the wrong game on screen)."""
    raw = score_var.get().strip()
    try:
        score = int(raw)
        if score < 0:
            raise ValueError("negative score")
    except ValueError:
        app.log_queue.put(
            f"[templates:{game_name}] capture: '{raw}' is not a non-negative int\n"
        )
        return
    try:
        import cv2
        from common.capture import grab_region
        from common.regions import get_region
        from common.score_template_ocr import extract_digit_components, save_template
        from common.window import get_bounds, WindowNotFoundError

        # All bots share the same window title constant — defined on
        # each minigame's main module. Import lazily to avoid module-
        # load-time penalty when the launcher starts.
        mod = __import__(f"minigames.{game_name}.main", fromlist=["WINDOW_TITLE"])
        window_title = mod.WINDOW_TITLE
        try:
            left, top, width, height = get_bounds(window_title)
        except WindowNotFoundError as e:
            app.log_queue.put(f"[templates:{game_name}] capture: {e}\n")
            return
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        game_dir = config.PROJECT_ROOT / "minigames" / game_name
        region = get_region(game_dir, "score", width, height)
        if region is None:
            app.log_queue.put(
                f"[templates:{game_name}] capture: no score region picked "
                f"(run '{game_name}-pick-score-region' first)\n"
            )
            return
        crop = bgr[
            region["top"]:region["top"] + region["height"],
            region["left"]:region["left"] + region["width"],
        ]
        components = extract_digit_components(crop)
        score_str = str(score)
        if len(components) != len(score_str):
            app.log_queue.put(
                f"[templates:{game_name}] capture: extracted {len(components)} "
                f"digit components but score '{score_str}' has {len(score_str)} digits — "
                f"is the game visible and score region picked correctly?\n"
            )
            return
        template_dir = game_dir / "assets" / "digit_templates"
        saved: list[int] = []
        skipped: list[int] = []
        for (patch, _box), digit_char in zip(components, score_str):
            digit = int(digit_char)
            target = template_dir / f"{digit}.png"
            if target.exists():
                skipped.append(digit)
                continue
            save_template(template_dir, digit, patch)
            saved.append(digit)
        if saved:
            app.log_queue.put(
                f"[templates:{game_name}] capture: saved "
                f"{', '.join(str(d) for d in saved)}\n"
            )
        if skipped:
            app.log_queue.put(
                f"[templates:{game_name}] capture: skipped "
                f"{', '.join(str(d) for d in skipped)} (already present)\n"
            )
        score_var.set("")
    except Exception as e:
        app.log_queue.put(f"[templates:{game_name}] capture failed: {e!r}\n")
    update_template_label(app, game_name)


def refresh_templates(app, game_name: str) -> None:
    """Run the bootstrap script for this game in a background thread,
    route its stdout into the launcher log, then refresh the row."""
    import contextlib
    import io
    import threading

    def worker() -> None:
        app.log_queue.put(f"[templates:{game_name}] refreshing...\n")
        try:
            from scripts.bootstrap_digit_templates import bootstrap_game
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bootstrap_game(game_name)
            for line in buf.getvalue().splitlines():
                app.log_queue.put(f"[templates:{game_name}] {line}\n")
        except Exception as e:
            app.log_queue.put(f"[templates:{game_name}] failed: {e!r}\n")
        # UI must update on the Tk main thread.
        app.root.after(0, lambda: update_template_label(app, game_name))

    threading.Thread(target=worker, daemon=True).start()


def build_predictor_cards(parent: ttk.Frame, app) -> None:
    """Build a horizontal row of stat cards, one per predictor kind."""
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(6, 0))
    for kind, emoji, label, color in PREDICTOR_CARD_SPECS:
        # tk.Frame (not ttk) so we can set a per-card background and
        # a coloured border via highlightbackground for the "active"
        # outline. ttk.Frame styling is more painful for this.
        card = tk.Frame(
            row, relief="flat", borderwidth=0, padx=8, pady=5,
            highlightthickness=2, highlightbackground=theme.BORDER,
            bg=theme.CARD_BG,
        )
        card.pack(side="left", padx=3)
        name = tk.Label(
            card, text=f"{emoji} {label}", fg=color, bg=theme.CARD_BG,
            font=("Segoe UI", 9, "bold"),
        )
        name.pack()
        lv = tk.Label(card, text="Lv 0", fg=theme.MUTED, bg=theme.CARD_BG,
                      font=("Segoe UI", 8))
        lv.pack()
        hp = ttk.Progressbar(card, length=80, mode="determinate", maximum=100)
        hp.pack(pady=(2, 1))
        rate = tk.Label(card, text="0/0", fg=theme.MUTED, bg=theme.CARD_BG,
                        font=("Consolas", 8))
        rate.pack()
        app.predictor_cards[kind] = {
            "card": card, "lv": lv, "hp": hp, "rate": rate,
            "color": color,
        }


def refresh_hoops_stats(app) -> None:
    """Pull per-predictor stats from shots.db and update each card.
    Cheap query (one GROUP BY over a few-hundred-row table); called
    on launcher start and whenever a hoops session ends."""
    if not app.predictor_cards:
        return
    import sqlite3
    db_path = config.MINIGAMES_DIR / "hoops" / "assets" / "shots.db"
    per_kind: dict[str, tuple[int, int, int]] = {}  # kind -> (sessions, shots, makes)
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
    """Outline the predictor card that matches the launcher's
    Predictor dropdown selection."""
    if not app.predictor_cards:
        return
    active = ""
    var = app.bot_option_vars.get(("hoops", "HOOPS_PREDICTOR_KIND"))
    if var is not None:
        active = var.get()
    for kind, widgets in app.predictor_cards.items():
        if kind == active:
            widgets["card"].config(highlightbackground=widgets["color"],
                                   highlightthickness=2)
        else:
            widgets["card"].config(highlightbackground=theme.BORDER,
                                   highlightthickness=2)


def save_mtime() -> float:
    """Newest mtime among the LevelDB save files, or 0 if unreachable."""
    import os, glob
    save_dir = os.path.expandvars(r"%APPDATA%\legends-of-idleon\Local Storage\leveldb")
    try:
        files = glob.glob(os.path.join(save_dir, "*.log")) + glob.glob(os.path.join(save_dir, "*.ldb"))
        return max((os.path.getmtime(p) for p in files), default=0.0)
    except OSError:
        return 0.0


def tick_hoops_display(app) -> None:
    """Update the hoops cooldown label, extrapolating from the last
    save-read snapshot. Re-arms itself once a second.

    We can't predict the cooldown accurately ahead of a save flush
    (the duration appears to escalate with consecutive plays via
    OLA[424] and we don't have the formula yet), so we surface
    save-staleness as a "(save Xm old)" suffix and let the user
    force a flush via character switch when the displayed value
    matters.
    """
    import time
    mtime = save_mtime()
    if mtime > app.last_save_mtime:
        app.last_save_mtime = mtime
        refresh_tries(app, silent=True)
    snap = app.hoops_cooldown_snapshot
    if snap is None:
        app.hoops_label.config(text="—")
    else:
        save_at, stored_cd = snap
        now = time.time()
        remaining = stored_cd - (now - save_at)
        save_age = now - save_at
        stale = save_age > 30
        if remaining <= 0:
            # Don't claim "ready" off a stale snapshot — the user
            # may have played in-game since the last flush, in
            # which case there's an active cooldown we can't see.
            # Verified 2026-05-10 14:49: in-game showed 219s while
            # launcher said "ready (save 1m32s old)" — misleading.
            base = "?" if stale else "ready"
        else:
            m, s = divmod(int(remaining) + 1, 60)
            base = f"{m}:{s:02d}"
        if stale:
            am, asec = divmod(int(save_age), 60)
            age_str = f"{am}m{asec:02d}s" if am else f"{asec}s"
            base += f"  (save {age_str} old)"
        app.hoops_label.config(text=base)
    app.root.after(1000, lambda: tick_hoops_display(app))


def refresh_tries(app, silent: bool = False) -> None:
    import time
    cd = read_hoops_cooldown()
    if cd is not None:
        # Anchor the snapshot to the save mtime, not the read time:
        # the cd value reflects the cooldown *at save time*, so
        # extrapolating from save_mtime keeps the math exact.
        anchor = save_mtime() or time.time()
        app.hoops_cooldown_snapshot = (anchor, cd)
        # Append one row to the cooldown-observation JSONL each
        # time the save flushes — feeds the formula-derivation
        # work in issue #22. Cheap (one extra load_save on
        # mtime-change ticks only) and gitignored output.
        try:
            _record_hoops_cooldown_observation()
        except Exception as e:  # never let observation logging block the UI
            if not silent:
                app.enqueue_log(f"[cooldown-obs] skipped: {e}\n")
    plays = read_minigame_plays_shared()
    if plays is None:
        app.tries_label.config(text="(save not found)")
        if not silent:
            app.enqueue_log("[tries] couldn't read save (Idleon not installed, or plyvel missing)\n")
        return
    app.tries_label.config(text=f"{plays}")
    # Persist for any external consumer.
    tries_counter.write(plays)
    if not silent:
        app.enqueue_log(f"[tries] refreshed: {plays}\n")
