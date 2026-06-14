"""Static launcher config: project paths and the minigame registry.

Kept separate so tab modules can import the registry without pulling in
the Tk app shell. Bot definition shape is unchanged from the original
single-file launcher (issue #24 out-of-scope: don't touch MINIGAMES).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MINIGAMES_DIR = PROJECT_ROOT / "minigames"

MINIGAMES = [
    {
        "name": "chopping",
        "emoji": "🪓",
        "bot": "chopping",
        "setup": [
            ("Observe (you play)", "chopping-observe"),
            ("Pick bar", "chopping-pick-bar-region"),
            ("Pick leaf", "chopping-pick-leaf-region"),
            ("Pick button", "chopping-pick-button-region"),
            ("Pick game over", "chopping-pick-game-over"),
            ("Calibrate", "chopping-calibrate"),
        ],
    },
    {
        "name": "hoops",
        "emoji": "🏀",
        "bot": "hoops",
        "setup": [
            ("Observe (you play)", "hoops-observe"),
            ("Capture", "hoops-capture"),
            ("Debug match", "hoops-debug"),
            ("Ball calibrate", "hoops-ball-calibrate"),
            ("Score calibrate", "hoops-score-calibrate"),
            ("Pick score", "hoops-pick-score-region"),
            ("Pick game over", "hoops-pick-game-over"),
            ("Pick lives", "hoops-pick-lives-region"),
        ],
        "bot_options": [
            {
                "label": "Predictor",
                "env": "HOOPS_PREDICTOR_KIND",
                "values": ["make_prob", "gp", "knn", "bivariate", "trajectory_knn", "trajectory_gp", "trajectory_rf"],
                "default": "make_prob",
            },
        ],
    },
    {
        "name": "darts",
        "emoji": "🎯",
        "bot": "darts",
        "setup": [
            ("Observe (you play)", "darts-observe"),
            ("Capture", "darts-capture"),
            ("Pick release", "darts-pick-release"),
            ("Auto-crop release", "darts-auto-crop-release"),
            ("Pick wind", "darts-pick-wind-region"),
            ("Watch wind", "darts-watch-wind"),
            ("Pick score", "darts-pick-score-region"),
        ],
    },
    {
        "name": "catching",
        "emoji": "🪰",
        "bot": "catching",
        "setup": [
            ("Observe (you play)", "catching-observe"),
            ("Capture", "catching-capture"),
            ("Pick play region", "catching-pick-play-region"),
            ("Extract fly", "catching-extract-fly"),
        ],
    },
    {
        "name": "mining",
        "emoji": "⛏️",
        "bot": "mining",
        "setup": [
            ("Observe (you play)", "mining-observe"),
            ("Capture", "mining-capture"),
            ("Trace (record attempt)", "mining-trace"),
            ("Pick play region", "mining-pick-play-region"),
            ("Pick start button", "mining-pick-start-button"),
            ("Render overlay video", "mining-render-overlay"),
        ],
    },
]
