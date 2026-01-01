# ============================================================================
# STATS TOOL - PRODUCTION SAFE VERSION
# ============================================================================

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing JSON scorecards
DEFAULT_PATH = PROJECT_ROOT / "final_json_scorecards"
if not DEFAULT_PATH.exists():
    raise FileNotFoundError(
        f"Scorecards directory not found at {DEFAULT_PATH}"
    )

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_scorecards(directory_path: str = DEFAULT_PATH) -> List[Dict]:
    """
    Load and aggregate all JSON scorecard files from a directory.
    """
    all_matches = []

    # 1. Validation: Check if directory exists
    if not os.path.exists(directory_path):
        # Fallback: check if the folder is in the current working directory
        local_path = os.path.join(os.getcwd(), "scorecards_json")
        if os.path.exists(local_path):
            directory_path = local_path
        else:
            print(f"⚠️ Warning: Scorecards directory not found at {directory_path}")
            return []

    print(f"📂 Loading scorecards from: {directory_path}...")

    # 2. Iteration: Loop through all files in the folder
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        
        if not files:
            print("⚠️ Warning: No .json files found in the directory.")
            return []

        for filename in files:
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Handle both single-match objects (Dict) and lists of matches (List)
                    if isinstance(data, list):
                        all_matches.extend(data)
                    elif isinstance(data, dict):
                        all_matches.append(data)
                        
            except json.JSONDecodeError:
                print(f"❌ Error: {filename} is not valid JSON. Skipping.")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

        print(f"✓ Stats Tool successfully loaded {len(all_matches)} matches from {len(files)} files.")
        return all_matches

    except Exception as e:
        print(f"❌ Critical Error accessing directory: {e}")
        return []

# Initialize global data
scorecards = load_scorecards()

# ---------------------------------------------------------------------------
# CORE FILTERS (PRIVATE)
# ---------------------------------------------------------------------------

def _filter_matches(
    player: str,
    year: Optional[int],
    opponent: Optional[str],
    match_ids: Optional[List[str]] = None
) -> List[Dict]:
    """
    Filter matches based on year, opponent, and explicit match IDs.
    """
    results = []
    
    # Normalize inputs
    target_player = player.strip().lower() if player else ""
    target_opponent = opponent.strip().lower() if opponent else ""

    for sc in scorecards:
        # Safety check: Ensure sc is a dictionary and has match_info
        if not isinstance(sc, dict) or "match_info" not in sc:
            continue

        # 1. ID Filter (for Hybrid RAG queries)
        # Handle cases where match_id might be int or string
        current_id = str(sc["match_info"].get("match_id", ""))
        
        if match_ids and current_id not in [str(m) for m in match_ids]:
            continue

        # 2. Year Filter
        match_date = sc["match_info"]["dates"]["start"]
        match_year = int(match_date.split("-")[0]) if match_date else 0
        if year and match_year != year:
            continue

        # 3. Opponent Filter
        teams = sc["match_info"]["teams"]
        team_names = [teams["team1"]["name"].lower(), teams["team2"]["name"].lower()]
        if target_opponent and target_opponent not in team_names:
            continue

        # 4. Player Presence Check (Optimization)
        player_found = False
        for innings in sc.get("innings", []):
            for bat in innings.get("batting", []):
                if target_player in bat["player_name"].lower():
                    player_found = True
                    break
            if not player_found:
                for bowl in innings.get("bowling", []):
                    if target_player in bowl["player_name"].lower():
                        player_found = True
                        break
            if player_found: 
                break
        
        if player_found:
            results.append(sc)

    return results


def _player_innings(sc: Dict, player: str) -> List[Dict]:
    """Extract batting performance for a specific match."""
    innings_data = []
    target_player = player.lower().strip()

    for innings in sc.get("innings", []):
        for bat in innings.get("batting", []):
            if target_player in bat["player_name"].lower():
                innings_data.append({
                    "match_id": sc["match_info"]["match_id"],
                    "innings": innings.get("innings_number"),
                    "runs": int(bat.get("runs", 0)),
                    "balls": int(bat.get("balls", 0)),
                    "fours": int(bat.get("fours", 0)),
                    "sixes": int(bat.get("sixes", 0)),
                    "dismissed": bat.get("dismissed", False)
                })
    return innings_data


def _player_bowling(sc: Dict, player: str) -> List[Dict]:
    """Extract bowling performance for a specific match."""
    bowling_data = []
    target_player = player.lower().strip()

    for innings in sc.get("innings", []):
        for bowl in innings.get("bowling", []):
            if target_player in bowl["player_name"].lower():
                # Handle cases where 'overs' exists but 'balls' doesn't
                balls_bowled = int(bowl.get("balls", 0))
                if balls_bowled == 0 and "overs" in bowl:
                    balls_bowled = int(float(bowl["overs"]) * 6)

                bowling_data.append({
                    "match_id": sc["match_info"]["match_id"],
                    "innings": innings.get("innings_number"),
                    "wickets": int(bowl.get("wickets", 0)),
                    "runs_conceded": int(bowl.get("runs", 0)),
                    "balls": balls_bowled
                })

    return bowling_data


# ---------------------------------------------------------------------------
# PUBLIC METRICS
# ---------------------------------------------------------------------------

def total_runs_in_series(player, year=None, opponent=None, match_ids=None) -> int:
    matches = _filter_matches(player, year, opponent, match_ids)
    total = 0
    for sc in matches:
        for inns in _player_innings(sc, player):
            total += inns["runs"]
    return total

def individual_runs_per_match(player, year=None, opponent=None, match_ids=None) -> List[Dict]:
    matches = _filter_matches(player, year, opponent, match_ids)
    result = []
    
    for sc in matches:
        match_runs = 0
        played = False
        for inns in _player_innings(sc, player):
            match_runs += inns["runs"]
            played = True
        
        if played:
            result.append({
                "match_id": sc["match_info"]["match_id"],
                "runs": match_runs,
                "player": player
            })
    return result

def total_wickets(player, year=None, opponent=None, match_ids=None) -> int:
    matches = _filter_matches(player, year, opponent, match_ids)
    total = 0
    for sc in matches:
        for bowl in _player_bowling(sc, player):
            total += bowl["wickets"]
    return total

def wickets_per_match(player, year=None, opponent=None, match_ids=None) -> List[Dict]:
    matches = _filter_matches(player, year, opponent, match_ids)
    result = []

    for sc in matches:
        match_wickets = 0
        match_runs = 0
        played = False
        
        for bowl in _player_bowling(sc, player):
            match_wickets += bowl["wickets"]
            match_runs += bowl["runs_conceded"]
            played = True
            
        if played:
            result.append({
                "match_id": sc["match_info"]["match_id"],
                "wickets": match_wickets,
                "runs_conceded": match_runs,
                "player": player
            })
    return result

def batting_avg_sr(player, year=None, opponent=None, match_ids=None) -> Dict:
    matches = _filter_matches(player, year, opponent, match_ids)
    runs = 0
    balls = 0
    dismissals = 0

    for sc in matches:
        for inns in _player_innings(sc, player):
            runs += inns["runs"]
            balls += inns["balls"]
            if inns["dismissed"]:
                dismissals += 1

    avg = (runs / dismissals) if dismissals > 0 else (runs if runs > 0 else 0.0)
    sr = (runs / balls * 100) if balls > 0 else 0.0

    return {
        "runs": runs,
        "balls": balls,
        "average": round(avg, 2),
        "strike_rate": round(sr, 2),
        "dismissals": dismissals
    }

def bowling_economy_sr(player, year=None, opponent=None, match_ids=None) -> Dict:
    matches = _filter_matches(player, year, opponent, match_ids)
    wickets = 0
    runs_conceded = 0
    balls = 0

    for sc in matches:
        for bowl in _player_bowling(sc, player):
            wickets += bowl["wickets"]
            runs_conceded += bowl["runs_conceded"]
            balls += bowl["balls"]

    overs = balls / 6
    economy = (runs_conceded / overs) if overs > 0 else 0.0
    bowl_sr = (balls / wickets) if wickets > 0 else 0.0
    average = (runs_conceded / wickets) if wickets > 0 else 0.0

    return {
        "wickets": wickets,
        "runs_conceded": runs_conceded,
        "balls": balls,
        "economy": round(economy, 2),
        "strike_rate": round(bowl_sr, 2),
        "average": round(average, 2)
    }

def boundaries(player, year=None, opponent=None, match_ids=None) -> Dict:
    matches = _filter_matches(player, year, opponent, match_ids)
    fours = 0
    sixes = 0

    for sc in matches:
        for inns in _player_innings(sc, player):
            fours += inns["fours"]
            sixes += inns["sixes"]

    return {
        "fours": fours,
        "sixes": sixes,
        "total_boundary_runs": (fours * 4) + (sixes * 6)
    }