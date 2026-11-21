"""No-login push-up leaderboard (SQLite, stdlib only).

Stores ONLY: username, clean reps, total reps, avg form score, grade, timestamp. Never videos.
Ranking: clean reps desc -> avg form score desc -> earliest submission. Best entry per username
(case-insensitive).

DB path: env LEADERBOARD_DB, default <repo>/data/leaderboard.db, falls back to a temp dir when the
default location isn't writable (e.g. read-only container image).
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "leaderboard.db"

USERNAME_RE = re.compile(r"^[A-Za-z0-9_ .\-]{3,20}$")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
BAD_WORDS = (
    "fuck", "shit", "bitch", "cunt", "dick", "pussy", "asshole", "bastard", "slut", "whore",
    "nigger", "nigga", "faggot", "retard", "nazi", "hitler", "porn", "penis", "vagina",
)

CONSENT_TEXT = "This is me, and I agree to my username and score being shown on the public leaderboard"
PRIVACY_TEXT = "We don't keep your video — only your username and score."


# ------------------------------------------------------------------------------------------ DB path
def _writable_dir(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".probe_{os.getpid()}"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def db_path() -> Path:
    """Resolve the SQLite path (env override -> repo data/ -> temp dir)."""
    env = os.environ.get("LEADERBOARD_DB")
    p = Path(env) if env else DEFAULT_DB
    if _writable_dir(p.parent):
        return p
    fb = Path(tempfile.gettempdir()) / "formsim"
    fb.mkdir(parents=True, exist_ok=True)
    return fb / p.name


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=10)
    con.execute(
        "CREATE TABLE IF NOT EXISTS scores ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " username TEXT NOT NULL, username_key TEXT NOT NULL,"
        " clean_reps INTEGER NOT NULL, total_reps INTEGER NOT NULL,"
        " avg_score REAL, grade TEXT, ts REAL NOT NULL)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_scores_key ON scores(username_key)")
    return con


# ------------------------------------------------------------------------------------------ validation
def normalize(name: str) -> str:
    """Lower-case, undo common leetspeak, drop separators (for bad-word checks)."""
    s = str(name or "").lower().translate(LEET)
    return re.sub(r"[^a-z]", "", s)


def contains_bad_word(name: str) -> bool:
    n = normalize(name)
    return any(w in n for w in BAD_WORDS)


def validate_username(name: Any) -> tuple[bool, str]:
    """Return (True, cleaned_name) or (False, reason)."""
    s = re.sub(r"\s+", " ", str(name or "")).strip()
    if len(s) < 3:
        return False, "Username must be at least 3 characters."
    if len(s) > 20:
        return False, "Username must be at most 20 characters."
    if not USERNAME_RE.match(s):
        return False, "Use only letters, numbers, spaces, and _ . -"
    if not re.search(r"[A-Za-z0-9]", s):
        return False, "Username needs at least one letter or number."
    if contains_bad_word(s):
        return False, "Please choose a different username."
    return True, s


# Leeway: a couple of tracking glitches (baggy clothes, someone walking past) can look like a "cut".
# Only clearly edited montages (more cuts than this) are refused.
MAX_CUTS = 3
_CUT_RE = re.compile(r"detected\s+(\d+)\s+camera cut", re.I)


def count_cuts(analysis: dict | None) -> int:
    """Number of camera cuts / person switches analysis reported (0 if none).
    Only the explicit 'Detected N camera cut(s)' warning counts — NOT the 'partial rep cut off
    (camera cut, lost tracking or clip end)' note, which is harmless."""
    n = 0
    for w in (analysis or {}).get("warnings") or []:
        m = _CUT_RE.search(str(w))
        if m:
            n = max(n, int(m.group(1)))
    return n


def can_submit(analysis: dict | None, consent: bool, is_demo: bool = False,
               simulated: bool = False) -> tuple[bool, str]:
    """Anti-cheat gate. Returns (ok, reason-if-not-ok)."""
    if is_demo:
        return False, "Demo clips can't be submitted — upload your own set."
    if simulated:
        return False, "Turn off the simulated fault (sidebar) to submit a real score."
    if not analysis:
        return False, "Analyze a video first."
    if not analysis.get("confidence_ok", False):
        return False, ("We couldn't see you clearly enough to verify the reps — film from the side with your "
                       "whole body in frame.")
    if count_cuts(analysis) > MAX_CUTS:
        return False, "This clip looks like an edited montage (many camera cuts) — submit one continuous set."
    s = analysis.get("summary") or {}
    if not int(s.get("rep_count") or 0):
        return False, "No complete push-up reps detected."
    if not consent:
        return False, "Tick the consent box to appear on the leaderboard."
    return True, ""


# ------------------------------------------------------------------------------------------ ranking
def _sort_key(r: dict) -> tuple:
    avg = r.get("avg_score")
    return (-int(r.get("clean_reps") or 0), -(float(avg) if avg is not None else -1.0), float(r.get("ts") or 0))


def rank_rows(rows: list[dict], n: int | None = 20) -> list[dict]:
    """Best entry per username (case-insensitive), ranked; adds 'rank'. Pure function."""
    best: dict[str, dict] = {}
    for r in rows:
        k = str(r.get("username", "")).strip().lower()
        if k not in best or _sort_key(r) < _sort_key(best[k]):
            best[k] = r
    out = [dict(r) for r in sorted(best.values(), key=_sort_key)]
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out[:n] if n else out


def _all_rows(path: str | Path | None = None) -> list[dict]:
    con = _connect(path)
    try:
        cur = con.execute("SELECT id, username, clean_reps, total_reps, avg_score, grade, ts FROM scores")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def submit(username: str, clean_reps: int, total_reps: int, avg_score: float | None, grade: str | None,
           path: str | Path | None = None) -> int:
    """Insert a score (validated). Returns the row id. Raises ValueError on a bad username."""
    ok, name = validate_username(username)
    if not ok:
        raise ValueError(name)
    clean_reps, total_reps = max(0, int(clean_reps or 0)), max(0, int(total_reps or 0))
    clean_reps = min(clean_reps, total_reps)
    avg = None if avg_score is None else round(max(0.0, min(100.0, float(avg_score))), 1)
    g = (str(grade or "")[:2]) or None
    con = _connect(path)
    try:
        cur = con.execute(
            "INSERT INTO scores (username, username_key, clean_reps, total_reps, avg_score, grade, ts) "
            "VALUES (?,?,?,?,?,?,?)", (name, name.lower(), clean_reps, total_reps, avg, g, time.time()))
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


def top(n: int = 20, path: str | Path | None = None) -> list[dict]:
    return rank_rows(_all_rows(path), n)


def rank_of(username: str, path: str | Path | None = None) -> tuple[int | None, int]:
    """(rank of this username's best entry or None, number of ranked users)."""
    ranked = rank_rows(_all_rows(path), None)
    k = str(username or "").strip().lower()
    for r in ranked:
        if str(r["username"]).lower() == k:
            return r["rank"], len(ranked)
    return None, len(ranked)
