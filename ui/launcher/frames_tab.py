"""Frames tab: image inspector.

Pick a minigame, then any directory under its assets/ that contains PNGs —
see them stacked, with sibling .txt files shown above. Works for the
hoops/darts per-shot monitor folders, capture dumps, calibration outputs.
"""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from ui.launcher import config, theme


def build(parent: ttk.Frame, app) -> None:
    controls = ttk.Frame(parent, padding=6)
    controls.pack(fill="x")

    ttk.Label(controls, text="Minigame:").pack(side="left")
    app.frame_minigame = tk.StringVar(value=config.MINIGAMES[0]["name"])
    mg_picker = ttk.Combobox(
        controls, textvariable=app.frame_minigame, state="readonly",
        values=[mg["name"] for mg in config.MINIGAMES],
        width=12,
    )
    mg_picker.pack(side="left", padx=(4, 12))
    mg_picker.bind("<<ComboboxSelected>>", lambda _e: refresh_frames_list(app))

    ttk.Button(controls, text="Refresh", command=lambda: refresh_frames_list(app)).pack(side="left")
    app.frame_status_label = ttk.Label(controls, text="", style="Muted.TLabel")
    app.frame_status_label.pack(side="left", padx=12)

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    # Left: list of directories under assets/ that contain PNGs.
    list_frame = ttk.LabelFrame(body, text="Folders with images", padding=4)
    list_frame.pack(side="left", fill="y", padx=(0, 6))
    list_scroll = ttk.Scrollbar(list_frame)
    list_scroll.pack(side="right", fill="y")
    app.frame_listbox = tk.Listbox(list_frame, width=32, height=20,
                                   yscrollcommand=list_scroll.set,
                                   font=theme.MONO, bg=theme.SURFACE, fg=theme.TEXT,
                                   selectbackground=theme.ACCENT_DIM,
                                   selectforeground=theme.TEXT,
                                   relief="flat", borderwidth=0,
                                   highlightthickness=0)
    app.frame_listbox.pack(side="left", fill="y")
    list_scroll.config(command=app.frame_listbox.yview)
    app.frame_listbox.bind("<<ListboxSelect>>", lambda _e: show_selected_frames(app))

    # Right: scrollable image canvas.
    viewer_frame = ttk.LabelFrame(body, text="Images", padding=4)
    viewer_frame.pack(side="left", fill="both", expand=True)

    app.frame_canvas = tk.Canvas(viewer_frame, bg=theme.CANVAS_BG, highlightthickness=0)
    app.frame_canvas.pack(side="left", fill="both", expand=True)
    canvas_scroll = ttk.Scrollbar(viewer_frame, orient="vertical",
                                  command=app.frame_canvas.yview)
    canvas_scroll.pack(side="right", fill="y")
    app.frame_canvas.config(yscrollcommand=canvas_scroll.set)

    app.frame_inner = ttk.Frame(app.frame_canvas)
    app.frame_canvas.create_window((0, 0), window=app.frame_inner, anchor="nw")
    app.frame_inner.bind(
        "<Configure>",
        lambda _e: app.frame_canvas.configure(scrollregion=app.frame_canvas.bbox("all")),
    )
    app.frame_canvas.bind_all(
        "<MouseWheel>",
        lambda e: app.frame_canvas.yview_scroll(-int(e.delta / 120), "units"),
    )

    refresh_frames_list(app)


def refresh_frames_list(app) -> None:
    app.frame_listbox.delete(0, "end")
    app.frame_dirs.clear()
    mg = app.frame_minigame.get()
    assets = config.MINIGAMES_DIR / mg / "assets"
    if not assets.exists():
        app.frame_status_label.config(text=f"no assets/ dir for {mg}")
        return

    # Find every directory under assets/ (including assets/ itself) that
    # has at least one PNG directly inside it. Display path is relative
    # to assets/ for compactness; "." represents assets/ root.
    candidates: list[Path] = []
    for d in [assets, *(p for p in assets.rglob("*") if p.is_dir())]:
        if any(p.suffix.lower() == ".png" for p in d.iterdir() if p.is_file()):
            candidates.append(d)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for d in candidates:
        rel = "." if d == assets else str(d.relative_to(assets)).replace("\\", "/")
        app.frame_listbox.insert("end", rel)
        app.frame_dirs[rel] = d
    app.frame_status_label.config(
        text=f"{len(candidates)} folder(s) with PNGs under {mg}/assets/"
    )


def show_selected_frames(app) -> None:
    sel = app.frame_listbox.curselection()
    if not sel:
        return
    rel = app.frame_listbox.get(sel[0])
    folder = app.frame_dirs.get(rel)
    if folder is None or not folder.exists():
        return

    for child in app.frame_inner.winfo_children():
        child.destroy()
    app.frame_images.clear()

    # Show any sibling text files (meta.txt, notes) above the images.
    for txt in sorted(folder.glob("*.txt")):
        try:
            content = txt.read_text(errors="replace")
        except Exception:
            continue
        t = tk.Text(app.frame_inner, height=min(8, max(2, content.count("\n") + 1)),
                    wrap="word", bg=theme.SURFACE, fg=theme.TEXT, font=theme.MONO,
                    relief="flat", borderwidth=0, padx=6, pady=4)
        t.insert("1.0", f"{txt.name}\n{content}")
        t.config(state="disabled")
        t.pack(fill="x", padx=4, pady=(0, 6))

    # Stable ordering: a few well-known names first (pre/post for hoops),
    # then everything else sorted by name.
    png_files = sorted(folder.glob("*.png"))
    priority = ["pre_shot.png", "post_shot.png"]
    ordered = (
        [p for name in priority for p in png_files if p.name == name]
        + [p for p in png_files if p.name not in priority]
    )
    for path in ordered:
        try:
            img = Image.open(path)
        except Exception as e:
            ttk.Label(app.frame_inner, text=f"{path.name}: {e}",
                      foreground=theme.DANGER).pack(anchor="w")
            continue
        max_w = 720
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        app.frame_images.append(photo)
        row = ttk.Frame(app.frame_inner)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=path.name, width=20,
                  foreground=theme.INFO).pack(side="left", anchor="n")
        tk.Label(row, image=photo, bd=0, bg=theme.BG).pack(side="left")
