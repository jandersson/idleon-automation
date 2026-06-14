"""Dark theme for the launcher: a palette + ttk style setup.

The original launcher mixed default (light) ttk widgets with hand-darkened
tk widgets (the log pane, predictor cards). This unifies everything on one
dark palette so the whole window is cohesive. The palette constants are the
single source of truth — tab modules import them for their plain-tk widgets
(Text/Listbox/Canvas/tk.Frame) since those can't be styled via ttk.

apply_theme() switches to the 'clam' base theme (the only stock ttk theme
that lets you recolor every element) and configures each widget class. Call
it once on the root before building any widgets.
"""
from tkinter import ttk

# --- Palette ---------------------------------------------------------------
BG = "#1e1f26"          # window background
SURFACE = "#262833"     # cards, labelframes, fields
SURFACE_ALT = "#31343f"  # buttons, headings, raised chrome
BORDER = "#3a3d4a"
ACCENT = "#8b7bf0"      # primary accent (violet — matches the GP card)
ACCENT_DIM = "#5a4fa8"
TEXT = "#e7e8ec"
MUTED = "#9aa0ad"
SUCCESS = "#4cc77a"
DANGER = "#e2566c"
WARN = "#d8a24a"
INFO = "#7fb6ff"        # informational labels (e.g. frame filenames)
LOG_BG = "#15161c"      # log pane / query editor (darkest surface)
CANVAS_BG = "#202129"   # image viewer backdrop
CARD_BG = SURFACE       # predictor stat cards

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_HEAD = ("Segoe UI", 11, "bold")
MONO = ("Consolas", 9)
MONO_SMALL = ("Consolas", 8)


def apply_theme(root) -> ttk.Style:
    """Apply the dark theme to `root` and return the configured Style.

    Configures every ttk widget class the launcher uses plus the popup
    listbox of comboboxes (an X-resource, not a ttk element). Also defines
    Start/Stop accent button styles."""
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)

    style.configure(".", background=BG, foreground=TEXT, font=FONT,
                    fieldbackground=SURFACE, bordercolor=BORDER,
                    focuscolor=ACCENT, troughcolor=SURFACE_ALT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Status.TLabel", foreground=MUTED, font=FONT_BOLD)

    style.configure("TLabelframe", background=BG, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                    font=FONT_BOLD)

    style.configure("TButton", background=SURFACE_ALT, foreground=TEXT,
                    borderwidth=0, focusthickness=0, padding=(11, 6),
                    font=FONT)
    style.map("TButton",
              background=[("disabled", SURFACE), ("pressed", ACCENT_DIM),
                          ("active", BORDER)],
              foreground=[("disabled", MUTED)])

    # Start = green, Stop = red — the two actions that matter at a glance.
    style.configure("Start.TButton", background="#2c6e49", foreground="#eafff1")
    style.map("Start.TButton", background=[("active", SUCCESS), ("pressed", "#23583a")])
    style.configure("Stop.TButton", background="#7a3340", foreground="#ffeef1")
    style.map("Stop.TButton", background=[("active", DANGER), ("pressed", "#5f2832")])

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(6, 6, 6, 0))
    style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED,
                    padding=(18, 9), font=FONT_BOLD, borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACCENT), ("active", TEXT)])

    style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE_ALT,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
                    selectbackground=ACCENT_DIM, selectforeground=TEXT, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)],
              foreground=[("readonly", TEXT)])
    # Combobox dropdown list is a Tk Listbox styled via the option DB.
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_DIM)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)

    style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT,
                    bordercolor=BORDER, insertcolor=TEXT, padding=4)

    style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE,
                    foreground=TEXT, borderwidth=0, rowheight=22)
    style.configure("Treeview.Heading", background=SURFACE_ALT, foreground=ACCENT,
                    font=FONT_BOLD, borderwidth=0, padding=4)
    style.map("Treeview", background=[("selected", ACCENT_DIM)],
              foreground=[("selected", TEXT)])
    style.map("Treeview.Heading", background=[("active", BORDER)])

    style.configure("Vertical.TScrollbar", background=SURFACE_ALT, troughcolor=BG,
                    borderwidth=0, arrowcolor=MUTED)
    style.configure("Horizontal.TScrollbar", background=SURFACE_ALT, troughcolor=BG,
                    borderwidth=0, arrowcolor=MUTED)
    style.map("Vertical.TScrollbar", background=[("active", BORDER)])
    style.map("Horizontal.TScrollbar", background=[("active", BORDER)])

    style.configure("TProgressbar", background=SUCCESS, troughcolor=SURFACE_ALT,
                    borderwidth=0, thickness=10)

    return style
