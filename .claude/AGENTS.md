# AGENTS.md

Agent conventions for this repo. The full project guide — architecture,
per-minigame patterns, the commit/push and testing workflow — is in
[`../CLAUDE.md`](../CLAUDE.md). This file adds the cross-cutting rule for work an
agent *discovers* but doesn't finish in the current task.

## File an issue for every refinement you find

While working you will surface things worth doing that aren't part of the task at
hand: an accuracy refinement, a deferred edge case, an approximation a better
approach could replace, a follow-up a fix just unblocked, or a "looks off"
anomaly in behaviour or data. **Do not leave it only in chat, a code comment, or
a docs caveat — open a GitHub issue so it's discoverable for later (or immediate)
work.** Bias to over-file: a redundant issue is cheap, a lost refinement is not.

What to file:
- An approximation / heuristic a more accurate method could replace.
- A deferred case, fallback, or TODO a change leaves behind.
- A follow-up the just-completed work makes possible.
- A behaviour or data anomaly worth a dedicated look (file it; don't bury it).

How:
- `gh issue create --title "<scope>: <concise refinement>" --label "<area>"`,
  matching the existing label scheme (`gh label list`, e.g. `minigame:fishing`).
- In the body: the context, the concrete refinement, why it's deferred (or why it
  should be done now), and a short acceptance sketch.
- Link related issues (`Refs #NN`) and the docs/code that describe it, and add a
  pointer **back** from those docs/code to the issue number — traceable both ways.

When to just do it instead: if the refinement is small and in scope, do it now —
the issue is for work that's genuinely deferred or needs its own discovery pass.

Example: the converged multi-catch kind-attribution refinement (#66) — found while
fixing the fishing score-delta handling, deferred because the catch *kind* drives
no decision yet, filed with a plan and cross-linked from
`docs/fishing_minigame.md`.
