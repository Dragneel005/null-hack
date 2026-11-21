"""Leaderboard tests. Run: .venv/bin/python tests/test_leaderboard.py"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import leaderboard as lb  # noqa: E402


def test_username_validation() -> None:
    assert lb.validate_username("  Suhail_99 ") == (True, "Suhail_99")
    assert lb.validate_username("a.b-c d")[0]
    assert not lb.validate_username("ab")[0]
    assert not lb.validate_username("x" * 21)[0]
    assert not lb.validate_username("<script>")[0]
    assert not lb.validate_username("name'; DROP")[0]
    assert not lb.validate_username("...")[0]
    assert not lb.validate_username(None)[0]


def test_bad_words() -> None:
    for bad in ("fuck", "FuCkEr99", "sh1t_head", "b1tch", "a$$hole", "5h17", "nazi.boy", "f.u.c.k"):
        assert lb.contains_bad_word(bad), bad
        assert not lb.validate_username(bad + "xx")[0] or len(bad + "xx") > 20, bad
    for ok in ("pushup_king", "Anna", "Helsinki_2026", "Matti"):
        assert not lb.contains_bad_word(ok), ok


def test_ranking_tiebreak_and_best_per_user() -> None:
    rows = [
        {"username": "alice", "clean_reps": 10, "total_reps": 12, "avg_score": 80, "ts": 3},
        {"username": "Alice", "clean_reps": 12, "total_reps": 12, "avg_score": 70, "ts": 4},  # alice's best
        {"username": "bob", "clean_reps": 12, "total_reps": 15, "avg_score": 90, "ts": 5},    # same clean, better score
        {"username": "carl", "clean_reps": 12, "total_reps": 12, "avg_score": 70, "ts": 1},   # ties alice, earlier
        {"username": "dina", "clean_reps": 2, "total_reps": 30, "avg_score": None, "ts": 0},  # half reps don't help
    ]
    r = lb.rank_rows(rows)
    assert [x["username"] for x in r] == ["bob", "carl", "Alice", "dina"], [x["username"] for x in r]
    assert [x["rank"] for x in r] == [1, 2, 3, 4]
    assert len(lb.rank_rows(rows, 2)) == 2


def test_gating() -> None:
    good = {"confidence_ok": True, "summary": {"rep_count": 5},
            "warnings": ["1 partial rep(s) were cut off (camera cut, lost tracking or clip end) and not counted."]}
    assert lb.can_submit(good, consent=True) == (True, "")
    assert not lb.can_submit(good, consent=False)[0]
    assert not lb.can_submit(good, consent=True, is_demo=True)[0]
    assert not lb.can_submit(good, consent=True, simulated=True)[0]
    assert not lb.can_submit(None, consent=True)[0]
    assert not lb.can_submit({**good, "confidence_ok": False}, consent=True)[0]
    assert not lb.can_submit({**good, "summary": {"rep_count": 0}}, consent=True)[0]
    # the harmless "partial rep cut off (camera cut, ...)" note must NOT block (it did before)
    assert lb.can_submit(good, consent=True)[0]
    # leeway: a couple of tracking glitches that look like cuts are fine (e.g. baggy clothes) ...
    few = {**good, "warnings": ["Detected 2 camera cut(s) / person switch(es); rep tracking was reset at each."]}
    assert lb.count_cuts(few) == 2 and lb.can_submit(few, consent=True)[0]
    # ... but an edited montage with many cuts is refused
    cut = {**good, "warnings": ["Detected 5 camera cut(s) / person switch(es); rep tracking was reset at each."]}
    assert not lb.can_submit(cut, consent=True)[0]


def test_db_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "sub" / "lb.db"
        lb.submit("Alice", 5, 6, 81.234, "B", path=db)
        time.sleep(0.01)
        lb.submit("bob", 7, 7, 90, "A", path=db)
        lb.submit("ALICE", 9, 8, 120, "A", path=db)  # clean capped at total, score clamped
        top = lb.top(20, path=db)
        assert [t["username"] for t in top] == ["ALICE", "bob"], top
        assert top[0]["clean_reps"] == 8 and top[0]["avg_score"] == 100.0
        assert lb.rank_of("alice", path=db) == (1, 2)
        assert lb.rank_of("nobody", path=db) == (None, 2)
        try:
            lb.submit("sh1t", 1, 1, 50, "C", path=db)
            raise AssertionError("bad username accepted")
        except ValueError:
            pass
        # stored columns: no video / path columns at all
        con = lb._connect(db)
        cols = [c[1] for c in con.execute("PRAGMA table_info(scores)")]
        con.close()
        assert set(cols) == {"id", "username", "username_key", "clean_reps", "total_reps", "avg_score", "grade", "ts"}


def test_db_path_env() -> None:
    import os
    with tempfile.TemporaryDirectory() as d:
        os.environ["LEADERBOARD_DB"] = str(Path(d) / "x" / "env.db")
        try:
            assert lb.db_path() == Path(d) / "x" / "env.db"
        finally:
            del os.environ["LEADERBOARD_DB"]
    os.environ["LEADERBOARD_DB"] = "/proc/nope/forbidden.db"   # unwritable -> temp fallback
    try:
        p = lb.db_path()
        assert str(p).startswith(tempfile.gettempdir()), p
    finally:
        del os.environ["LEADERBOARD_DB"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL OK")
