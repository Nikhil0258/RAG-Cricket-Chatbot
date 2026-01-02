# ============================================================================
# STATS TOOL - PRODUCTION SAFE (SCORECARD-BASED SCHEMA)
# ============================================================================

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORECARDS_DIR = PROJECT_ROOT / "final_json_scorecards"

if not SCORECARDS_DIR.exists():
    raise FileNotFoundError(f"Scorecards directory not found at {SCORECARDS_DIR}")

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_scorecards(directory: Path) -> List[Dict]:
    matches = []
    files = [f for f in os.listdir(directory) if f.endswith(".json")]

    print(f"📂 Loading scorecards from: {directory}")

    for file in files:
        try:
            with open(directory / file, "r", encoding="utf-8") as f:
                data = json.load(f)
                matches.append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    print(f"✓ Loaded {len(matches)} matches")
    return matches


scorecards = load_scorecards(SCORECARDS_DIR)

# ---------------------------------------------------------------------------
# CORE MATCH FILTER
# ---------------------------------------------------------------------------

def _filter_matches(
    player: str,
    year: Optional[int] = None,
    opponent: Optional[str] = None,
    match_ids: Optional[List[str]] = None
) -> List[Dict]:

    results = []
    target_player = player.lower().strip()
    target_opponent = opponent.lower().strip() if opponent else ""

    for sc in scorecards:
        mi = sc.get("match_info", {})
        match_id = str(mi.get("match_id", ""))

        if match_ids and match_id not in map(str, match_ids):
            continue

        start_date = mi.get("dates", {}).get("start")
        if not start_date:
            continue

        match_year = int(start_date.split("-")[0])
        if year and match_year != year:
            continue

        teams = mi.get("teams", {})
        team_names = [
            teams.get("team1", {}).get("name", "").lower(),
            teams.get("team2", {}).get("name", "").lower()
        ]

        if target_opponent and target_opponent not in team_names:
            continue

        # Player presence check (batting OR bowling)
        found = False

        for row in sc.get("batting_scorecard", []):
            if target_player in row.get("batsman", "").lower():
                found = True
                break

        if not found:
            for row in sc.get("bowling_scorecard", []):
                if target_player in row.get("bowler", "").lower():
                    found = True
                    break

        if found:
            results.append(sc)

    return results

# ---------------------------------------------------------------------------
# PLAYER EXTRACTION (SCORECARD BASED)
# ---------------------------------------------------------------------------

def _player_batting(sc: Dict, player: str) -> List[Dict]:
    p = player.lower().strip()
    data = []

    for row in sc.get("batting_scorecard", []):
        if p in row.get("batsman", "").lower():
            data.append({
                "match_id": sc["match_info"]["match_id"],
                "runs": safe_int(row.get("runs")),
                "balls": safe_int(row.get("balls")),
                "fours": safe_int(row.get("fours")),
                "sixes": safe_int(row.get("sixes")),
                "dismissed": bool(row.get("isOut", False))
            })

    return data


def _player_bowling(sc: Dict, player: str) -> List[Dict]:
    p = player.lower().strip()
    data = []

    for row in sc.get("bowling_scorecard", []):
        if p in row.get("bowler id", "").lower():
            overs = safe_float(row.get("overs"))
            balls = int(overs * 6)

            data.append({
                "match_id": sc["match_info"]["match_id"],
                "wickets": safe_int(row.get("wickets")),
                "runs_conceded": safe_int(row.get("conceded")),
                "balls": balls
            })

    return data

# ---------------------------------------------------------------------------
# PUBLIC METRICS
# ---------------------------------------------------------------------------

def total_runs_in_series(player, year=None, opponent=None, match_ids=None) -> int:
    total = 0
    for sc in _filter_matches(player, year, opponent, match_ids):
        for inns in _player_batting(sc, player):
            total += inns["runs"]
    return total


def individual_runs_per_match(player, year=None, opponent=None, match_ids=None) -> List[Dict]:
    results = []

    for sc in _filter_matches(player, year, opponent, match_ids):
        runs = sum(i["runs"] for i in _player_batting(sc, player))
        if runs > 0:
            results.append({
                "match_id": sc["match_info"]["match_id"],
                "runs": runs,
                "player": player
            })

    return results


def batting_avg_sr(player, year=None, opponent=None, match_ids=None) -> Dict:
    runs = balls = outs = 0

    for sc in _filter_matches(player, year, opponent, match_ids):
        for inns in _player_batting(sc, player):
            runs += inns["runs"]
            balls += inns["balls"]
            outs += int(inns["dismissed"])

    avg = runs / outs if outs else runs
    sr = (runs / balls) * 100 if balls else 0.0

    return {
        "runs": runs,
        "balls": balls,
        "average": round(avg, 2),
        "strike_rate": round(sr, 2),
        "dismissals": outs
    }


def total_wickets(player, year=None, opponent=None, match_ids=None) -> int:
    total = 0
    for sc in _filter_matches(player, year, opponent, match_ids):
        for b in _player_bowling(sc, player):
            total += b["wickets"]
    return total


def bowling_economy_sr(player, year=None, opponent=None, match_ids=None) -> Dict:
    wickets = runs = balls = 0

    for sc in _filter_matches(player, year, opponent, match_ids):
        for b in _player_bowling(sc, player):
            wickets += b["wickets"]
            runs += b["runs_conceded"]
            balls += b["balls"]

    overs = balls / 6 if balls else 0
    economy = runs / overs if overs else 0
    sr = balls / wickets if wickets else 0
    avg = runs / wickets if wickets else 0

    return {
        "wickets": wickets,
        "runs_conceded": runs,
        "balls": balls,
        "economy": round(economy, 2),
        "strike_rate": round(sr, 2),
        "average": round(avg, 2)
    }


def boundaries(player, year=None, opponent=None, match_ids=None) -> Dict:
    fours = sixes = 0

    for sc in _filter_matches(player, year, opponent, match_ids):
        for inns in _player_batting(sc, player):
            fours += inns["fours"]
            sixes += inns["sixes"]

    return {
        "fours": fours,
        "sixes": sixes,
        "total_boundary_runs": (fours * 4) + (sixes * 6)
    }
'''if __name__ == "__main__":
    print("Total runs:", total_runs_in_series("Virat Kohli"))
    print("Batting AVG & SR:", batting_avg_sr("Virat Kohli"))
    print("\n--- Year Filter Test ---")
    print("2019 runs:", total_runs_in_series("Virat Kohli", year=2019))
    print("2020 runs:", total_runs_in_series("Virat Kohli", year=2020))
    print("\n--- Opponent Filter Test ---")
    print("vs Australia:", total_runs_in_series("Virat Kohli", opponent="australia"))
    print("vs England:", total_runs_in_series("Virat Kohli", opponent="england"))
    print("\n--- Runs Per Match ---")
    for r in individual_runs_per_match("Virat Kohli"):
        print(r)
    print("\n--- Bowling Stats ---")
    print("Total wickets:", total_wickets("Jasprit Bumrah"))
    print("Bowling SR & Econ:", bowling_economy_sr("Jasprit Bumrah"))
    print("\n--- Boundary Stats ---")
    print(boundaries("Virat Kohli"))
    print("\n--- Match ID Filter ---")
    print(total_runs_in_series("Virat Kohli", match_ids=["1187685"]))'''
