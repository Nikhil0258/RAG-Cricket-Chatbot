# %%
#Importing Required Libraries
import pandas as pd
import json
from pathlib import Path
import ast
import os

# %%
#Defining Paths and Constants
TEAM_NAME = 'India'
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
dataset_path = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/Dataset"
new_folder_path = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/"

# %%
#Creating new folders
final_json_scorecards = Path(new_folder_path+"final_json_scorecards")
final_json_scorecards.mkdir(exist_ok = True)
final_match_summaries = Path(new_folder_path+"final_match_summaries")
final_match_summaries.mkdir(exist_ok = True)

# %%
#Reading Datasets/CSV Files
matches = pd.read_csv(dataset_path+"/test_Matches_Data.csv")
batting = pd.read_csv(dataset_path+"/test_Batting_Card.csv")
bowling = pd.read_csv(dataset_path+"/test_Bowling_Card.csv")
partnership = pd.read_csv(dataset_path+"/test_Partnership_Card.csv")
fow = pd.read_csv(dataset_path+"/test_Fow_Card.csv")
playerInfo = pd.read_csv(dataset_path+"/players_info.csv")

# %%
matches['Match Start Date'] = pd.to_datetime(matches['Match Start Date'],errors='coerce')
matches['Match End Date'] = pd.to_datetime(matches['Match End Date'],errors = 'coerce')

# %%
final_dataset = matches[
    ((matches['Team1 Name'] == TEAM_NAME)|(matches['Team2 Name'] == TEAM_NAME)) 
    &
    ((matches['Match Start Date'] >= '2020-01-01') & (matches['Match Start Date'] <= '2024-12-31'))
]

# %%
player_id_to_name = dict(zip(playerInfo["player_id"], playerInfo["player_name"]))

# %%
def map_player_id(value):
    if pd.isna(value):
        return None
    try:
        return player_id_to_name.get(int(value), str(value))
    except:
        return str(value)

# %%
def parse_fielders(value):
    if pd.isna(value) or value in ["[]", "", None]:
        return []
    try:
        ids = ast.literal_eval(value)
        return [map_player_id(pid) for pid in ids]
    except:
        return []

# %%
def parse_player_list(value):
    if pd.isna(value) or value in ["[]", "", None]:
        return []
    try:
        ids = ast.literal_eval(value)
        return [map_player_id(pid) for pid in ids]
    except:
        # fallback if already comma-separated names
        return [v.strip() for v in str(value).split(",")]

# %%
import numpy as np

def clean_nan(obj):
    """
    Recursively converts objects into JSON-serializable types:
    - numpy.int64 → int
    - numpy.float64 → float
    - NaN → None
    """
    # Handle numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)

    # Handle Python NaN
    if isinstance(obj, float) and pd.isna(obj):
        return None

    # Handle dict
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}

    # Handle list
    if isinstance(obj, list):
        return [clean_nan(i) for i in obj]

    return obj


# %%
batting["batsman"] = batting["batsman"].apply(map_player_id)
batting["bowler"] = batting["bowler"].apply(map_player_id)
batting["fielders"] = batting["fielders"].apply(parse_fielders)
bowling["bowler id"] = bowling["bowler id"].apply(map_player_id)
partnership["player1"] = partnership["player1"].apply(map_player_id)
partnership["player2"] = partnership["player2"].apply(map_player_id)
fow["player"] = fow["player"].apply(map_player_id)

# %%
for match_id in matches["Match ID"].unique():

    match_row = matches[matches["Match ID"] == match_id].iloc[0]

    innings_summary = []

    if not pd.isna(match_row["Innings1 Team1 Runs Scored"]):
        innings_summary.append({
            "team": match_row["Team1 Name"],
            "innings": 1,
            "runs": match_row["Innings1 Team1 Runs Scored"],
            "wickets": match_row["Innings1 Team1 Wickets Fell"],
            "extras": match_row["Innings1 Team1 Extras Rec"]
        })

    if not pd.isna(match_row["Innings2 Team1 Runs Scored"]):
        innings_summary.append({
            "team": match_row["Team1 Name"],
            "innings": 2,
            "runs": match_row["Innings2 Team1 Runs Scored"],
            "wickets": match_row["Innings2 Team1 Wickets Fell"],
            "extras": match_row["Innings2 Team1 Extras Rec"]
        })

    if not pd.isna(match_row["Innings1 Team2 Runs Scored"]):
        innings_summary.append({
            "team": match_row["Team2 Name"],
            "innings": 1,
            "runs": match_row["Innings1 Team2 Runs Scored"],
            "wickets": match_row["Innings1 Team2 Wickets Fell"],
            "extras": match_row["Innings1 Team2 Extras Rec"]
        })

    if not pd.isna(match_row["Innings2 Team2 Runs Scored"]):
        innings_summary.append({
            "team": match_row["Team2 Name"],
            "innings": 2,
            "runs": match_row["Innings2 Team2 Runs Scored"],
            "wickets": match_row["Innings2 Team2 Wickets Fell"],
            "extras": match_row["Innings2 Team2 Extras Rec"]
        })

    match_info = {
        "match_id": int(match_id),
        "match_name": match_row["Match Name"],
        "series": match_row["Series Name"],
        "format": match_row["Match Format"],

        "dates": {
            "start": (
                match_row["Match Start Date"].strftime("%Y-%m-%d")
                if not pd.isna(match_row["Match Start Date"])
                else None
            ),
            "end": (
                match_row["Match End Date"].strftime("%Y-%m-%d")
                if not pd.isna(match_row["Match End Date"])
                else None
            )
},

        "venue": {
            "stadium": match_row["Match Venue (Stadium)"],
            "city": match_row["Match Venue (City)"],
            "country": match_row["Match Venue (Country)"]
        },

        "teams": {
            "team1": {
                "name": match_row["Team1 Name"],
                "captain": map_player_id(match_row["Team1 Captain"]),
                "playing_xi": parse_player_list(match_row["Team1 Playing 11"])
            },
            "team2": {
                "name": match_row["Team2 Name"],
                "captain": map_player_id(match_row["Team2 Captain"]),
                "playing_xi": parse_player_list(match_row["Team2 Playing 11"])
            }
        },

        "toss": {
            "winner": match_row["Toss Winner"],
            "decision": match_row["Toss Winner Choice"]
        },

        "officials": {
            "umpires": [match_row["Umpire 1"], match_row["Umpire 2"]],
            "match_referee": match_row["Match Referee"]
        },

        "result": {
            "winner": match_row["Match Winner"],
            "result_text": match_row["Match Result Text"],
            "player_of_match": map_player_id(match_row["MOM Player"])
        },

        "innings_summary": innings_summary,

        "debut_players": parse_player_list(match_row["Debut Players"])
    }

    scorecard = {
        "match_info": match_info,
        "batting_scorecard": batting[batting["Match ID"] == match_id].to_dict("records"),
        "bowling_scorecard": bowling[bowling["Match ID"] == match_id].to_dict("records"),
        "partnerships": partnership[partnership["Match ID"] == match_id].to_dict("records"),
        "fall_of_wickets": fow[fow["Match ID"] == match_id].to_dict("records")
    }

    scorecard = clean_nan(scorecard)

    with open(final_json_scorecards / f"match_{match_id}.json", "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

# %%
def get_match_innings(team_name, team_innings, team1, team2):
    if team_name == team1:
        if team_innings == 1:
            return 1
        elif team_innings == 2:
            return 3
    elif team_name == team2:
        if team_innings == 1:
            return 2
        elif team_innings == 2:
            return 4
    return None

# %%
json_files_present = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/final_json_scorecards"

# %%
for file in os.listdir(json_files_present):
    if file.startswith('match_') and file.endswith('.json'):
        with open(os.path.join(json_files_present,file), 'r') as f:
            match_data = json.load(f)
        summary_lines = []
        match_name = match_data['match_info']['match_name']
        series_name = match_data['match_info']['series']
        match_start_date = match_data['match_info']['dates']['start']
        match_end_date = match_data['match_info']['dates']['end']
        venue_stadium, venue_city,venue_country = match_data['match_info']['venue']['stadium'], match_data['match_info']['venue']['city'], match_data['match_info']['venue']['country']
        team1_name, team2_name = match_data['match_info']['teams']['team1']['name'], match_data['match_info']['teams']['team2']['name']
        team1_captain, team2_captain = match_data['match_info']['teams']['team1']['captain'], match_data['match_info']['teams']['team2']['captain']
        debutants = match_data['match_info']['debut_players']
        result = match_data['match_info']['result']['result_text']
        POTM = match_data['match_info']['result']['player_of_match']
        toss_winner = match_data['match_info']['toss']['winner']
        toss_elected = match_data['match_info']['toss']['decision']
        team1 = match_data['match_info']['teams']['team1']['name']
        team2 = match_data['match_info']['teams']['team2']['name']
        summary_lines.append(
            f"In the series {series_name}, the match between {team1_name} and {team2_name} "
            f"was played at {venue_stadium}, {venue_city}, {venue_country} "
            f"from {match_start_date} to {match_end_date}. "
            f"{team1_name} were captained by {team1_captain}, while {team2_name} "
            f"were led by {team2_captain}."
        )
        summary_lines.append(
        f"{toss_winner} won the toss and elected to {toss_elected} first. "
        f"Debutants in this match were: {', '.join(debutants)}."
        )
        summary_lines.append("\nInnings Summary:")
        for info in match_data['match_info']['innings_summary']:
            team_name = info['team']
            team_innings = info['innings']
            total_runs = info['runs']
            total_wickets = info['wickets']

            match_innings = get_match_innings(
                team_name,
                team_innings,
                team1,
                team2
            )

            # Best batter
            best_batsman_runs = 0.0
            best_batsman_name = None

            for value in match_data['batting_scorecard']:
                if (
                    value['team'] == team_name
                    and value['innings'] == match_innings
                    and value['runs'] is not None
                ):
                    if value['runs'] > best_batsman_runs:
                        best_batsman_runs = value['runs']
                        best_batsman_name = value['batsman']

            # Best bowler
            best_bowler_wickets = 0
            best_bowler_name = None

            for value in match_data['bowling_scorecard']:
                if (
                    value['team'] != team_name
                    and value['innings'] == match_innings
                    and value['wickets'] is not None
                ):
                    if value['wickets'] > best_bowler_wickets:
                        best_bowler_wickets = value['wickets']
                        best_bowler_name = value['bowler id']

            # Final innings sentence
            batter_text = (
                f"{best_batsman_name} ({best_batsman_runs})"
                if best_batsman_name else
                "no significant batting contribution"
            )

            bowler_text = (
                f"{best_bowler_name} ({best_bowler_wickets} wickets)"
                if best_bowler_name else
                "no significant bowling contribution"
            )

            summary_lines.append(
                f"{team_name} {team_innings} innings: "
                f"{total_runs}/{total_wickets}. "
                f"Top scorer was {batter_text}, "
                f"while {bowler_text} led the bowling."
            )
        summary_lines.append(
            f"\nFinal Result: {result}. "
            f"{POTM} was named Player of the Match."
        )
        match_id = match_data['match_info']['match_id']
        summary_text = "\n".join(summary_lines)
        file_path = os.path.join(
            final_match_summaries,
            f"match_{match_id}_summary.txt"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Summary saved to {file_path}")



