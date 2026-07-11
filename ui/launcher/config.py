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
            ("Pick score", "chopping-pick-score-region"),
            ("Pick game over", "chopping-pick-game-over"),
            ("Calibrate", "chopping-calibrate"),
        ],
        "bot_options": [
            {
                # Writes the (leaf+bar) composite the detector saw to
                # assets/captures/botrun_<stamp>/ (gitignored) — ~2 Hz
                # heartbeat + one frame per chop. Sets
                # CHOPPING_SAVE_FRAMES, which main.py reads (the GUI
                # can't pass --save-frames). Default on: attempts are
                # scarce (shared 5/day pool), the crops are tiny, and
                # the 2026-07-06 chop-9 stall had to be diagnosed
                # blind because nothing visual persisted.
                "label": "Save frames",
                "env": "CHOPPING_SAVE_FRAMES",
                "values": ["on", "off"],
                "default": "on",
            },
            {
                # Aiming mode (2026-07-06 refactor): 'plan' schedules
                # clicks for the time-domain center of an upcoming
                # zone crossing (fires through the ~600 px/s speed
                # saturation where the reactive gate starves); 'gate'
                # is the legacy in-zone time-to-red gate, kept as the
                # validated fallback. Sets CHOPPING_AIM.
                "label": "Aim",
                "env": "CHOPPING_AIM",
                "values": ["plan", "gate"],
                "default": "plan",
            },
            {
                # Find the overlay visually each session (the bar
                # anchors above the player, so its position changes
                # per environment; regions.json only fits the map it
                # was picked in). 'off' = legacy regions.json. Sets
                # CHOPPING_AUTOREGIONS.
                "label": "Auto regions",
                "env": "CHOPPING_AUTOREGIONS",
                "values": ["on", "off"],
                "default": "on",
            },
            {
                # While waiting for the bar, template-match the 'Play
                # Game' prompt and click it (capped at 2/session) —
                # zero idle time before chop 1, which matters since
                # the speed ramp accumulates with round age. CLICKING
                # PLAY CONSUMES a shared daily attempt. Sets
                # CHOPPING_AUTO_PLAY.
                "label": "Auto play",
                "env": "CHOPPING_AUTO_PLAY",
                "values": ["on", "off"],
                "default": "on",
            },
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
            ("Pick score region", "catching-pick-score-region"),
            ("Extract fly", "catching-extract-fly"),
            ("Capture digits", "catching-capture-digits"),
            ("Fit dynamics", "catching-fit-dynamics"),
        ],
        "bot_options": [
            {
                # Saves what the detectors saw to assets/captures/botrun_<stamp>/
                # (gitignored) for offline diagnosis. Sets CATCHING_SAVE_FRAMES,
                # which main.py reads (the GUI can't pass --save-frames).
                "label": "Save frames",
                "env": "CATCHING_SAVE_FRAMES",
                "values": ["on", "off"],
                "default": "on",
            },
            {
                # Dense per-poll trajectory CSV to assets/traces/ (gitignored)
                # for fitting the flap model (#60). Sets CATCHING_TRACE. Turn on
                # for a calibration run, then `catching-fit-dynamics`.
                "label": "Trace",
                "env": "CATCHING_TRACE",
                "values": ["on", "off"],
                "default": "off",
            },
            {
                # Use the fitted predictive flap timer (needs assets/
                # dynamics.json from Fit dynamics). Sets CATCHING_USE_MODEL;
                # main.py falls back to the hand-tuned timing if unfit.
                "label": "Use model",
                "env": "CATCHING_USE_MODEL",
                "values": ["on", "off"],
                "default": "off",
            },
            {
                # The #61 phase-locked flap planner (setup->launch alignment,
                # side-dodges, live speed tracking) — supersedes "Use model"
                # when on (implies it). Sets CATCHING_PLANNER; opt-in until
                # live-validated (sim: 3.7x baseline threads, 45% full-stream
                # survival vs 0).
                "label": "Planner",
                "env": "CATCHING_PLANNER",
                "values": ["on", "off"],
                "default": "off",
            },
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
        "bot_options": [
            {
                # Writes every frame to assets/captures/botrun_<stamp>/
                # (gitignored) for offline analysis — death diagnosis, slam
                # timing, ore characterisation. Sets MINING_SAVE_FRAMES,
                # which main.py reads (the GUI can't pass --save-frames).
                # Default on while the slam policy is still being validated:
                # runs are short (~15-20s) so the disk cost is trivial, and
                # every attempt is scarce calibration data.
                "label": "Save frames",
                "env": "MINING_SAVE_FRAMES",
                "values": ["on", "off"],
                "default": "on",
            },
        ],
    },
    {
        "name": "fishing",
        "emoji": "🎣",
        "bot": "fishing",
        "setup": [
            ("Observe (you play)", "fishing-observe"),
            ("Capture", "fishing-capture"),
            ("Pick play region", "fishing-pick-play-region"),
            ("Calibrate", "fishing-calibrate"),
        ],
        "bot_options": [
            {
                # Saves each cast's landing-poll frames to assets/captures/
                # botrun_<stamp>/ (gitignored), named with the find_lure x —
                # to debug landing detection. Sets FISHING_SAVE_FRAMES.
                "label": "Save frames",
                "env": "FISHING_SAVE_FRAMES",
                "values": ["on", "off"],
                "default": "off",
            },
        ],
    },
]
