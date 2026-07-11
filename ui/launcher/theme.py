"""Dark theme for the launcher: palette, type scale, and ttk styles.

The palette constants are the single source of truth — tab modules import
them for their plain-tk widgets (Text/Listbox/Canvas/tk.Frame) since those
can't be styled via ttk.

apply_theme() switches to the 'clam' base theme (the only stock ttk theme
that lets you recolor every element) and configures each widget class. Call
it once on the root before building any widgets.

Layout rhythm: use the PAD_* constants instead of ad-hoc pixel values so
spacing stays consistent across tabs.
"""
from pathlib import Path
from tkinter import ttk

# --- Palette ---------------------------------------------------------------
BG = "#16171d"          # window background (deepest)
SURFACE = "#1f2129"     # cards, labelframes, fields
SURFACE_ALT = "#2a2d38"  # buttons, headings, raised chrome
BORDER = "#363a47"
ACCENT = "#8b7bf0"      # primary accent (violet)
ACCENT_DIM = "#5a4fa8"
TEXT = "#e7e8ec"
MUTED = "#8f95a3"
SUCCESS = "#4cc77a"
DANGER = "#e2566c"
WARN = "#d8a24a"
INFO = "#7fb6ff"        # informational labels (e.g. frame filenames)
LOG_BG = "#101116"      # log pane / query editor (darkest surface)
CANVAS_BG = "#1b1c23"   # image viewer backdrop
CARD_BG = SURFACE       # stat cards
HEADER_BG = "#1a1b22"   # top header bar

# Per-bot accent colors — match the repo's GitHub label colors so the same
# hue identifies a minigame everywhere. Fallback ACCENT for new bots.
BOT_COLORS = {
    "chopping": "#3fa46a",
    "hoops": "#f97316",
    "darts": "#f4a261",
    "catching": "#1abc9c",
    "mining": "#b08968",
    "fishing": "#4a90d9",
}


def bot_color(name: str) -> str:
    return BOT_COLORS.get(name, ACCENT)


# --- Type scale -------------------------------------------------------------
FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_HEAD = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI Semibold", 13)
MONO = ("Consolas", 9)
MONO_SMALL = ("Consolas", 8)

# --- Spacing rhythm ----------------------------------------------------------
PAD_XS = 2
PAD_S = 4
PAD = 8
PAD_L = 12
PAD_XL = 16

_HERE = Path(__file__).parent
ICON_ICO = _HERE / "assets" / "idleon.ico"


def apply_theme(root) -> ttk.Style:
    """Apply the dark theme to `root` and return the configured Style.

    Configures every ttk widget class the launcher uses plus the popup
    listbox of comboboxes (an X-resource, not a ttk element). Also defines
    the accent button styles and the window icon."""
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    if ICON_ICO.exists():
        try:
            root.iconbitmap(str(ICON_ICO))
        except Exception:
            pass  # icon is cosmetic — never block startup on it

    style.configure(".", background=BG, foreground=TEXT, font=FONT,
                    fieldbackground=SURFACE, bordercolor=BORDER,
                    focuscolor=ACCENT, troughcolor=SURFACE_ALT)
    style.configure("TFrame", background=BG)
    style.configure("Header.TFrame", background=HEADER_BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Small.TLabel", foreground=MUTED, font=FONT_SMALL)
    style.configure("Status.TLabel", foreground=MUTED, font=FONT_BOLD)
    style.configure("Header.TLabel", background=HEADER_BG, foreground=TEXT)
    style.configure("HeaderTitle.TLabel", background=HEADER_BG,
                    foreground=TEXT, font=FONT_TITLE)
    style.configure("HeaderMuted.TLabel", background=HEADER_BG,
                    foreground=MUTED, font=FONT_SMALL)
    style.configure("HeaderValue.TLabel", background=HEADER_BG,
                    foreground=ACCENT, font=FONT_BOLD)
    style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED,
                    font=FONT_SMALL)

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
    style.configure("Ghost.TButton", background=SURFACE, padding=(8, 3),
                    font=FONT_SMALL)
    style.map("Ghost.TButton",
              background=[("disabled", SURFACE), ("pressed", ACCENT_DIM),
                          ("active", SURFACE_ALT)],
              foreground=[("disabled", BORDER)])

    # Start = green, Stop = red — the two actions that matter at a glance.
    style.configure("Start.TButton", background="#2c6e49", foreground="#eafff1",
                    font=FONT_BOLD, padding=(14, 7))
    style.map("Start.TButton",
              background=[("disabled", SURFACE), ("active", SUCCESS),
                          ("pressed", "#23583a")],
              foreground=[("disabled", MUTED)])
    style.configure("Stop.TButton", background="#7a3340", foreground="#ffeef1",
                    font=FONT_BOLD, padding=(14, 7))
    style.map("Stop.TButton",
              background=[("active", DANGER), ("pressed", "#5f2832")])

    style.configure("TNotebook", background=BG, borderwidth=0,
                    tabmargins=(PAD, PAD_S, PAD, 0))
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                    padding=(14, 8), font=FONT_BOLD, borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", SURFACE)],
              foreground=[("selected", ACCENT), ("active", TEXT)])

    style.configure("TCombobox", fieldbackground=SURFACE_ALT, background=SURFACE_ALT,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
                    selectbackground=ACCENT_DIM, selectforeground=TEXT, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE_ALT)],
              foreground=[("readonly", TEXT)])
    # Combobox dropdown list is a Tk Listbox styled via the option DB.
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_DIM)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)

    style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT,
                    bordercolor=BORDER, insertcolor=TEXT, padding=4)
    style.configure("TCheckbutton", background=BG, foreground=MUTED,
                    font=FONT_SMALL, focuscolor=BG)
    style.map("TCheckbutton", background=[("active", BG)],
              foreground=[("active", TEXT)])

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
