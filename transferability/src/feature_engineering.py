# coding: utf-8

"""
Transferability project
Feature engineering for the first test
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


SUMMER_WINDOW_START_MONTH = 6
SUMMER_WINDOW_START_DAY = 1
SUMMER_WINDOW_END_MONTH = 9
SUMMER_WINDOW_END_DAY = 15
EUROPEAN_SEASON_PATTERN = r"^\d{2}/\d{2}$"
TEAM_CATEGORY_PATTERN = re.compile(
    r"\b(u-?\d{1,2}|b|ii|iii|iv|youth|yth|reserve|reserves|primavera|mestalla)\b",
    flags=re.IGNORECASE,
)


def has_team_category_marker(team_name):
    """
    Detect whether a team name looks like a youth, reserve or academy side.

    Parameters
    ----------
    team_name : str
        Raw team name.

    Returns
    -------
    bool
        True when the team looks like a category side.
    """
    if pd.isna(team_name):
        return False
    return TEAM_CATEGORY_PATTERN.search(str(team_name)) is not None


def normalize_team_family(team_name):
    """
    Build a simplified club-family name to detect internal category moves.

    Parameters
    ----------
    team_name : str
        Raw team name.

    Returns
    -------
    str
        Simplified family name.
    """
    if pd.isna(team_name):
        return ""

    normalized_name = str(team_name).lower()
    normalized_name = TEAM_CATEGORY_PATTERN.sub(" ", normalized_name)
    normalized_name = re.sub(r"[^a-z0-9]+", " ", normalized_name)
    normalized_name = re.sub(r"\s+", " ", normalized_name).strip()
    return normalized_name


def is_internal_category_change(from_team_name, to_team_name):
    """
    Detect whether a transfer is only an internal category move.

    Parameters
    ----------
    from_team_name : str
        Origin team name.
    to_team_name : str
        Destination team name.

    Returns
    -------
    bool
        True when both teams belong to the same club family and at least one
        side looks like a youth or reserve category.
    """
    if pd.isna(from_team_name) or pd.isna(to_team_name):
        return False
    if str(from_team_name) == str(to_team_name):
        return False

    from_family = normalize_team_family(from_team_name)
    to_family = normalize_team_family(to_team_name)

    if from_family == "" or to_family == "":
        return False

    return from_family == to_family and (
        has_team_category_marker(from_team_name) or has_team_category_marker(to_team_name)
    )


def is_continental_competition(competition_name):
    """
    Detect whether a competition is a continental club competition.

    Parameters
    ----------
    competition_name : str
        Competition name.

    Returns
    -------
    int
        1 when the competition looks continental, 0 otherwise.
    """
    if pd.isna(competition_name):
        return 0

    normalized_name = str(competition_name).lower()
    continental_keywords = [
        "uefa",
        "champions league",
        "europa league",
        "conference league",
        "uecl",
        "libertadores",
        "sudamericana",
        "afc",
        "caf",
        "concacaf",
        "copa america",
    ]
    return int(any(keyword in normalized_name for keyword in continental_keywords))


def is_domestic_cup_competition(competition_name):
    """
    Detect whether a competition looks like a domestic cup.

    Parameters
    ----------
    competition_name : str
        Competition name.

    Returns
    -------
    int
        1 when the competition looks like a domestic cup, 0 otherwise.
    """
    if pd.isna(competition_name):
        return 0

    normalized_name = str(competition_name).lower()
    cup_keywords = [
        "cup",
        "copa",
        "coppa",
        "pokal",
        "beker",
        "taça",
        "taca",
        "kypello",
        "trophy",
    ]
    if is_continental_competition(competition_name):
        return 0
    return int(any(keyword in normalized_name for keyword in cup_keywords))


def is_youth_competition(competition_name):
    """
    Detect whether a competition looks like a youth competition.

    Parameters
    ----------
    competition_name : str
        Competition name.

    Returns
    -------
    int
        1 when the competition looks like a youth competition, 0 otherwise.
    """
    if pd.isna(competition_name):
        return 0

    normalized_name = str(competition_name).lower()
    youth_keywords = [
        "u17",
        "u18",
        "u19",
        "u20",
        "u21",
        "u23",
        "youth",
        "primavera",
        "premier league 2",
    ]
    return int(any(keyword in normalized_name for keyword in youth_keywords))


def get_competition_tier_proxy(competition_name):
    """
    Build a simple division proxy from the competition name.

    Parameters
    ----------
    competition_name : str
        Competition name.

    Returns
    -------
    int
        League tier proxy.
        1 = top flight
        2 = first tier
        3 = second tier
        4 = third tier
        5 = lower tier or unknown
        0 = cup, continental or youth competition
    """
    if pd.isna(competition_name):
        return 5

    if is_continental_competition(competition_name) or is_domestic_cup_competition(competition_name):
        return 0
    if is_youth_competition(competition_name):
        return 0

    normalized_name = str(competition_name).lower()

    top_flight_keywords = [
        "premier league",
        "serie a",
        "bundesliga",
        "laliga",
        "ligue 1",
        "eredivisie",
        "primeira liga"
    ]
    first_tier_keywords = [
        "super league 1",
        "superliga",
        "allsvenskan",
        "eliteserien",
        "a lyga",
        "1. hnl",
        "1. liga",
        "1. division",
        "1.division",
        "1.league",
        "1.lig",
        "1.liga",
        "1 deild",
        "championship"
    ]
    second_tier_keywords = [
        "serie b",
        "2. bundesliga",
        "2. division",
        "2.division",
        "2. liga",
        "2.lig",
        "2.liga",
        "2 deild",
        "ligue 2",
        "2ª división",
        "segunda liga",
        "liga portugal 2"
    ]
    third_tier_keywords = [
        "league one",
        "serie c",
        "3. liga",
        "3 liga",
        "3.lig",
        "primera federación",
        "segunda federación",
        "national 1"
    ]

    if any(keyword in normalized_name for keyword in top_flight_keywords):
        return 1
    if any(keyword in normalized_name for keyword in first_tier_keywords):
        return 2
    if any(keyword in normalized_name for keyword in second_tier_keywords):
        return 3
    if any(keyword in normalized_name for keyword in third_tier_keywords):
        return 4
    return 5


def is_top5_league_competition(competition_name):
    """
    Detect whether a competition is one of the big five European leagues.

    Parameters
    ----------
    competition_name : str
        Competition name.

    Returns
    -------
    int
        1 when the competition matches a top-5 league, 0 otherwise.
    """
    if pd.isna(competition_name):
        return 0

    normalized_name = str(competition_name).lower()
    top5_competitions = [
        "premier league",
        "laliga",
        "bundesliga",
        "serie a",
        "ligue 1",
    ]
    return int(any(top5_name == normalized_name for top5_name in top5_competitions))


def get_reference_date(summer_year):
    """
    Return the modeling snapshot date for one summer year.

    Parameters
    ----------
    summer_year : int
        Summer year of the prediction window.

    Returns
    -------
    pd.Timestamp
        June 1st of the requested summer year.
    """
    return pd.Timestamp(year=int(summer_year), month=SUMMER_WINDOW_START_MONTH, day=SUMMER_WINDOW_START_DAY)


def get_window_end_date(summer_year):
    """
    Return the end date of the modeled summer transfer window.

    Parameters
    ----------
    summer_year : int
        Summer year of the prediction window.

    Returns
    -------
    pd.Timestamp
        September 15th of the requested summer year.
    """
    return pd.Timestamp(year=int(summer_year), month=SUMMER_WINDOW_END_MONTH, day=SUMMER_WINDOW_END_DAY)


def get_previous_season_name(season_name):
    """
    Convert a season label into the previous European season label.

    Parameters
    ----------
    season_name : str
        Season label formatted as ``YY/YY``.

    Returns
    -------
    str
        Previous season label.
    """
    season_start = int(str(season_name).split("/")[0])
    season_end = int(str(season_name).split("/")[1])
    previous_start = season_start - 1
    previous_end = season_end - 1
    return "{0:02d}/{1:02d}".format(previous_start, previous_end)


def keep_european_seasons(performances):
    """
    Keep only European-format seasons used in the first test.

    Parameters
    ----------
    performances : pd.DataFrame
        Raw performances table.

    Returns
    -------
    pd.DataFrame
        Filtered performances table.
    """
    filtered_performances = performances.loc[
        performances["season_name"].astype(str).str.match(EUROPEAN_SEASON_PATTERN)
    ].copy()
    return filtered_performances


def build_reference_windows(performances):
    """
    Create one modeling row per player and per summer year.

    Parameters
    ----------
    performances : pd.DataFrame
        Filtered performances table.

    Returns
    -------
    pd.DataFrame
        Reference window table.
    """
    reference_windows = performances[["player_id", "season_name"]].drop_duplicates().copy()
    reference_windows["summer_year"] = reference_windows["season_name"].str[-2:].astype(int) + 2000
    reference_windows["reference_date"] = reference_windows["summer_year"].apply(get_reference_date)
    return reference_windows


def aggregate_performances(performances):
    """
    Aggregate player performances at the player-season level.

    Parameters
    ----------
    performances : pd.DataFrame
        Raw or filtered performances table.

    Returns
    -------
    pd.DataFrame
        Aggregated player-season performance table.
    """
    aggregation_columns = {
        "competition_id": "nunique",
        "team_name": "nunique",
        "nb_in_group": "sum",
        "nb_on_pitch": "sum",
        "goals": "sum",
        "assists": "sum",
        "own_goals": "sum",
        "subed_in": "sum",
        "subed_out": "sum",
        "yellow_cards": "sum",
        "second_yellow_cards": "sum",
        "direct_red_cards": "sum",
        "penalty_goals": "sum",
        "minutes_played": "sum",
        "goals_conceded": "sum",
        "clean_sheets": "sum",
    }

    aggregated_performances = (
        performances.groupby(["player_id", "season_name"], as_index=False)
        .agg(aggregation_columns)
        .rename(
            columns={
                "competition_id": "nb_competitions",
                "team_name": "nb_teams",
                "nb_in_group": "matches_in_squad",
                "nb_on_pitch": "matches_played",
                "goals": "goals_scored",
                "assists": "assists_made",
            }
        )
    )

    aggregated_performances["goals_per_90"] = np.where(
        aggregated_performances["minutes_played"] > 0,
        aggregated_performances["goals_scored"] * 90.0 / aggregated_performances["minutes_played"],
        0.0,
    )
    aggregated_performances["assists_per_90"] = np.where(
        aggregated_performances["minutes_played"] > 0,
        aggregated_performances["assists_made"] * 90.0 / aggregated_performances["minutes_played"],
        0.0,
    )
    aggregated_performances["cards_per_match"] = np.where(
        aggregated_performances["matches_played"] > 0,
        (
            aggregated_performances["yellow_cards"]
            + aggregated_performances["second_yellow_cards"]
            + aggregated_performances["direct_red_cards"]
        )
        / aggregated_performances["matches_played"],
        0.0,
    )
    aggregated_performances["bench_appearance_share"] = np.where(
        aggregated_performances["matches_played"] > 0,
        aggregated_performances["subed_in"] / aggregated_performances["matches_played"],
        0.0,
    )

    return aggregated_performances


def add_performance_trends(aggregated_performances):
    """
    Add previous-season comparison features to player-season performances.

    Parameters
    ----------
    aggregated_performances : pd.DataFrame
        Player-season performance table.

    Returns
    -------
    pd.DataFrame
        Player-season performance table enriched with trends.
    """
    previous_season_performances = aggregated_performances.copy()
    previous_season_performances["season_name"] = previous_season_performances["season_name"].apply(
        lambda season_name: "{0:02d}/{1:02d}".format(
            int(season_name.split("/")[0]) + 1,
            int(season_name.split("/")[1]) + 1,
        )
    )

    previous_season_performances = previous_season_performances.rename(
        columns={
            "minutes_played": "previous_minutes_played",
            "goals_scored": "previous_goals_scored",
            "assists_made": "previous_assists_made",
            "goals_per_90": "previous_goals_per_90",
            "assists_per_90": "previous_assists_per_90",
        }
    )

    enriched_performances = aggregated_performances.merge(
        previous_season_performances[
            [
                "player_id",
                "season_name",
                "previous_minutes_played",
                "previous_goals_scored",
                "previous_assists_made",
                "previous_goals_per_90",
                "previous_assists_per_90",
            ]
        ],
        on=["player_id", "season_name"],
        how="left",
    )

    enriched_performances["minutes_trend"] = (
        enriched_performances["minutes_played"] - enriched_performances["previous_minutes_played"].fillna(0)
    )
    enriched_performances["goals_trend"] = (
        enriched_performances["goals_scored"] - enriched_performances["previous_goals_scored"].fillna(0)
    )
    enriched_performances["assists_trend"] = (
        enriched_performances["assists_made"] - enriched_performances["previous_assists_made"].fillna(0)
    )
    enriched_performances["goals_per_90_trend"] = (
        enriched_performances["goals_per_90"] - enriched_performances["previous_goals_per_90"].fillna(0)
    )
    enriched_performances["assists_per_90_trend"] = (
        enriched_performances["assists_per_90"] - enriched_performances["previous_assists_per_90"].fillna(0)
    )

    return enriched_performances


def build_club_context_features(performances):
    """
    Build simple club and league context features from season performances.

    Parameters
    ----------
    performances : pd.DataFrame
        Filtered performances table.

    Returns
    -------
    pd.DataFrame
        Player-season club and competition context table.
    """
    performance_context = performances.copy()
    performance_context["is_continental_competition"] = performance_context["competition_name"].apply(
        is_continental_competition
    )
    performance_context["is_domestic_cup_competition"] = performance_context["competition_name"].apply(
        is_domestic_cup_competition
    )
    performance_context["is_youth_competition"] = performance_context["competition_name"].apply(
        is_youth_competition
    )
    performance_context["competition_tier_proxy"] = performance_context["competition_name"].apply(
        get_competition_tier_proxy
    )
    performance_context["competition_is_top5_league"] = performance_context["competition_name"].apply(
        is_top5_league_competition
    )

    player_team_minutes = (
        performance_context.groupby(["player_id", "season_name", "team_name"], as_index=False)
        .agg(
            team_minutes_played=("minutes_played", "sum"),
            team_matches_played=("nb_on_pitch", "sum"),
        )
    )

    player_season_minutes = (
        player_team_minutes.groupby(["player_id", "season_name"], as_index=False)["team_minutes_played"]
        .sum()
        .rename(columns={"team_minutes_played": "season_total_minutes_played"})
    )

    primary_team = (
        player_team_minutes.sort_values(
            ["player_id", "season_name", "team_minutes_played", "team_matches_played"],
            ascending=[True, True, False, False],
        )
        .groupby(["player_id", "season_name"], as_index=False)
        .head(1)
        .rename(columns={"team_name": "primary_team_name"})
    )
    primary_team = primary_team.merge(player_season_minutes, on=["player_id", "season_name"], how="left")
    primary_team["primary_team_minutes_share"] = np.where(
        primary_team["season_total_minutes_played"] > 0,
        primary_team["team_minutes_played"] / primary_team["season_total_minutes_played"],
        0.0,
    )
    primary_team["primary_team_is_category_side"] = primary_team["primary_team_name"].apply(
        lambda team_name: int(has_team_category_marker(team_name))
    )

    player_competition_minutes = (
        performance_context.groupby(["player_id", "season_name", "competition_name"], as_index=False)
        .agg(
            competition_minutes_played=("minutes_played", "sum"),
            competition_tier_proxy=("competition_tier_proxy", "max"),
        )
    )

    primary_competition = (
        player_competition_minutes.sort_values(
            ["player_id", "season_name", "competition_minutes_played"],
            ascending=[True, True, False],
        )
        .groupby(["player_id", "season_name"], as_index=False)
        .head(1)
        .rename(columns={"competition_name": "primary_competition_name"})
    )
    primary_competition["primary_competition_is_top_flight"] = (
        primary_competition["competition_tier_proxy"] == 1
    ).astype(int)
    primary_competition["primary_competition_is_second_tier"] = (
        primary_competition["competition_tier_proxy"] == 2
    ).astype(int)
    primary_competition["primary_competition_is_lower_division"] = (
        primary_competition["competition_tier_proxy"] >= 3
    ).astype(int)
    primary_competition["primary_competition_is_top5_league"] = primary_competition[
        "primary_competition_name"
    ].apply(is_top5_league_competition)

    season_competition_flags = (
        performance_context.groupby(["player_id", "season_name"], as_index=False)
        .agg(
            played_in_continental_competition=("is_continental_competition", "max"),
            played_in_domestic_cup_competition=("is_domestic_cup_competition", "max"),
            played_in_youth_competition=("is_youth_competition", "max"),
        )
    )

    team_season_context = (
        performance_context.groupby(["season_name", "team_name"], as_index=False)
        .agg(
            team_season_player_count=("player_id", "nunique"),
            team_season_competition_count=("competition_id", "nunique"),
            team_season_has_continental_competition=("is_continental_competition", "max"),
            team_season_has_top_flight_competition=("competition_tier_proxy", lambda values: int((values == 1).any())),
            team_season_has_youth_competition=("is_youth_competition", "max"),
            team_season_has_top5_league_competition=("competition_is_top5_league", "max"),
        )
    )

    club_context_features = primary_team.merge(
        primary_competition[
            [
                "player_id",
                "season_name",
                "competition_tier_proxy",
                "primary_competition_name",
                "primary_competition_is_top_flight",
                "primary_competition_is_second_tier",
                "primary_competition_is_lower_division",
                "primary_competition_is_top5_league",
            ]
        ],
        on=["player_id", "season_name"],
        how="left",
    )
    club_context_features = club_context_features.merge(
        season_competition_flags,
        on=["player_id", "season_name"],
        how="left",
    )
    club_context_features = club_context_features.merge(
        team_season_context,
        left_on=["season_name", "primary_team_name"],
        right_on=["season_name", "team_name"],
        how="left",
    )

    return club_context_features.drop(columns=["team_name"], errors="ignore")


def build_player_features(players, reference_windows):
    """
    Build player profile features that are stable enough for historical modeling.

    Parameters
    ----------
    players : pd.DataFrame
        Players table.
    reference_windows : pd.DataFrame
        Modeling window table.

    Returns
    -------
    pd.DataFrame
        Player profile features by player and summer year.
    """
    selected_columns = [
        "player_id",
        "player_name",
        "date_of_birth",
        "citizenship",
        "second_citizenship",
        "is_eu",
        "main_position",
        "position",
        "foot",
    ]

    player_features = reference_windows.merge(players[selected_columns], on="player_id", how="left")
    player_features["player_name"] = player_features["player_name"].fillna("Unknown Player")
    player_features["age_at_window"] = (
        (player_features["reference_date"] - player_features["date_of_birth"]).dt.days / 365.25
    )
    player_features["has_second_citizenship"] = player_features["second_citizenship"].notna().astype(int)
    player_features["is_eu"] = player_features["is_eu"].fillna(False).astype(int)
    player_features["citizenship"] = player_features["citizenship"].fillna("Unknown")
    player_features["main_position"] = player_features["main_position"].fillna("Unknown")
    player_features["position"] = player_features["position"].fillna("Unknown")
    player_features["foot"] = player_features["foot"].fillna("Unknown")
    return player_features.drop(columns=["date_of_birth", "second_citizenship"])


def get_next_season_name(season_name):
    """
    Convert a European season label into the next season label.

    Parameters
    ----------
    season_name : str
        Season label formatted as ``YY/YY``.

    Returns
    -------
    str
        Next season label.
    """
    season_start = int(str(season_name).split("/")[0])
    season_end = int(str(season_name).split("/")[1])
    return "{0:02d}/{1:02d}".format(season_start + 1, season_end + 1)


def build_team_season_lookup(performances):
    """
    Build a team-season competition lookup from performance data.

    Parameters
    ----------
    performances : pd.DataFrame
        Filtered performances table.

    Returns
    -------
    pd.DataFrame
        Team-season lookup with primary competition metadata.
    """
    team_competition_minutes = performances.copy()
    team_competition_minutes["competition_tier_proxy"] = team_competition_minutes["competition_name"].apply(
        get_competition_tier_proxy
    )
    team_competition_minutes["competition_is_top5_league"] = team_competition_minutes["competition_name"].apply(
        is_top5_league_competition
    )

    team_competition_minutes = (
        team_competition_minutes.groupby(["season_name", "team_name", "competition_name"], as_index=False)
        .agg(
            competition_minutes_played=("minutes_played", "sum"),
            competition_tier_proxy=("competition_tier_proxy", "max"),
            competition_is_top5_league=("competition_is_top5_league", "max"),
        )
    )

    team_competition_minutes["competition_priority"] = np.where(
        team_competition_minutes["competition_tier_proxy"] > 0,
        team_competition_minutes["competition_tier_proxy"],
        99,
    )

    team_primary_competition = (
        team_competition_minutes.sort_values(
            ["season_name", "team_name", "competition_priority", "competition_minutes_played"],
            ascending=[True, True, True, False],
        )
        .groupby(["season_name", "team_name"], as_index=False)
        .head(1)
        .rename(
            columns={
                "competition_name": "team_primary_competition_name",
                "competition_tier_proxy": "team_primary_competition_tier",
                "competition_is_top5_league": "team_primary_is_top5_league",
            }
        )
    )

    return team_primary_competition[
        [
            "season_name",
            "team_name",
            "team_primary_competition_name",
            "team_primary_competition_tier",
            "team_primary_is_top5_league",
        ]
    ].copy()


def build_transfer_market_labels(transfers, team_season_lookup, summer_years):
    """
    Build target labels and destination-context labels from transfer history.

    Parameters
    ----------
    transfers : pd.DataFrame
        Transfers table.
    team_season_lookup : pd.DataFrame
        Team-season competition lookup.
    summer_years : iterable of int
        Summer years used in the modeling sample.

    Returns
    -------
    pd.DataFrame
        Player-summer target table with additional market context labels.
    """
    transfer_labels = transfers.copy()
    transfer_labels["transfer_date"] = pd.to_datetime(transfer_labels["transfer_date"], errors="coerce")
    transfer_labels["summer_year"] = transfer_labels["transfer_date"].dt.year
    transfer_labels["is_internal_category_change"] = transfer_labels.apply(
        lambda row: is_internal_category_change(row["from_team_name"], row["to_team_name"]),
        axis=1,
    )
    transfer_labels = transfer_labels.loc[
        transfer_labels["summer_year"].isin(list(set(summer_years)))
        & transfer_labels["transfer_type"].isin(["Transfer", "Loan"])
        & (~transfer_labels["is_internal_category_change"])
        & (
            (
                (transfer_labels["transfer_date"].dt.month > 6)
                & (transfer_labels["transfer_date"].dt.month < 9)
            )
            | (
                (transfer_labels["transfer_date"].dt.month == 6)
                & (transfer_labels["transfer_date"].dt.day >= 1)
            )
            | (
                (transfer_labels["transfer_date"].dt.month == 9)
                & (transfer_labels["transfer_date"].dt.day <= 15)
            )
        )
    ].copy()

    transfer_labels = transfer_labels.loc[
        transfer_labels["season_name"].astype(str).str.match(EUROPEAN_SEASON_PATTERN)
    ].copy()
    transfer_labels["previous_season_name"] = transfer_labels["season_name"].apply(get_previous_season_name)

    transfer_labels = transfer_labels.merge(
        team_season_lookup.rename(
            columns={
                "season_name": "previous_season_name",
                "team_name": "from_team_name",
                "team_primary_competition_name": "from_competition_name",
                "team_primary_competition_tier": "from_competition_tier",
                "team_primary_is_top5_league": "from_is_top5_league",
            }
        ),
        on=["previous_season_name", "from_team_name"],
        how="left",
    )

    transfer_labels = transfer_labels.merge(
        team_season_lookup.rename(
            columns={
                "season_name": "season_name",
                "team_name": "to_team_name",
                "team_primary_competition_name": "to_competition_name",
                "team_primary_competition_tier": "to_competition_tier",
                "team_primary_is_top5_league": "to_is_top5_league",
            }
        ),
        on=["season_name", "to_team_name"],
        how="left",
    )

    transfer_labels["target"] = 1
    transfer_labels["target_to_top5_league"] = transfer_labels["to_is_top5_league"].fillna(0).astype(int)
    transfer_labels["last_move_was_intra_league"] = (  # 1 if the player's most recent transfer stayed within the same league level
        transfer_labels["from_competition_name"].fillna("Unknown")
        == transfer_labels["to_competition_name"].fillna("Unknown")
    ).astype(int)
    transfer_labels["last_move_was_upward_league"] = (  # 1 if the player's most recent transfer moved up at least one league tier
        transfer_labels["to_competition_tier"].fillna(99) < transfer_labels["from_competition_tier"].fillna(99)
    ).astype(int)

    return (
        transfer_labels.groupby(["player_id", "summer_year"], as_index=False)
        .agg(
            target=("target", "max"),
            target_to_top5_league=("target_to_top5_league", "max"),
            last_move_was_intra_league=("last_move_was_intra_league", "max"),
            last_move_was_upward_league=("last_move_was_upward_league", "max"),
        )
    )


def build_transfer_features(transfers, summer_years):
    """
    Build transfer history features for every modeled summer year.

    Parameters
    ----------
    transfers : pd.DataFrame
        Transfers table.
    summer_years : iterable of int
        Summer years to model.

    Returns
    -------
    pd.DataFrame
        Transfer history features by player and summer year.
    """
    transfer_feature_frames = []
    transfers = transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers = transfers.dropna(subset=["transfer_date"])
    transfers["is_internal_category_change"] = transfers.apply(
        lambda row: is_internal_category_change(row["from_team_name"], row["to_team_name"]),
        axis=1,
    )

    for summer_year in sorted(set(summer_years)):
        reference_date = get_reference_date(summer_year)
        transfer_history = transfers.loc[transfers["transfer_date"] < reference_date].copy()
        market_transfer_history = transfer_history.loc[~transfer_history["is_internal_category_change"]].copy()

        transfer_counts = (
            market_transfer_history.pivot_table(
                index="player_id",
                columns="transfer_type",
                values="transfer_date",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        rename_mapping = {
            "Loan": "nb_loans_before_window",
            "Return from loan": "nb_returns_from_loan_before_window",
            "Transfer": "nb_transfers_before_window",
            "Draft": "nb_drafts_before_window",
            "(null)": "nb_null_transfer_type_before_window",
        }
        transfer_counts = transfer_counts.rename(columns=rename_mapping)

        expected_columns = [
            "nb_loans_before_window",
            "nb_returns_from_loan_before_window",
            "nb_transfers_before_window",
            "nb_drafts_before_window",
            "nb_null_transfer_type_before_window",
        ]
        for expected_column in expected_columns:
            if expected_column not in transfer_counts.columns:
                transfer_counts[expected_column] = 0

        transfer_counts["nb_market_moves_before_window"] = (
            transfer_counts["nb_loans_before_window"] + transfer_counts["nb_transfers_before_window"]
        )

        internal_move_counts = (
            transfer_history.loc[transfer_history["is_internal_category_change"]]
            .groupby("player_id", as_index=False)
            .size()
            .rename(columns={"size": "nb_internal_category_moves_before_window"})
        )

        transfer_history_sorted = market_transfer_history.sort_values(["player_id", "transfer_date"])
        last_transfer = transfer_history_sorted.groupby("player_id", as_index=False).tail(1).copy()
        last_transfer["days_since_last_move"] = (reference_date - last_transfer["transfer_date"]).dt.days
        last_transfer["was_last_move_loan"] = (last_transfer["transfer_type"] == "Loan").astype(int)
        last_transfer["was_last_move_return_from_loan"] = (
            last_transfer["transfer_type"] == "Return from loan"
        ).astype(int)
        last_transfer["was_last_move_transfer"] = (last_transfer["transfer_type"] == "Transfer").astype(int)
        last_transfer["moved_in_last_365_days"] = (last_transfer["days_since_last_move"] <= 365).astype(int)

        distinct_destination_teams = (
            market_transfer_history.groupby("player_id")["to_team_name"]
            .nunique()
            .reset_index()
            .rename(columns={"to_team_name": "nb_destination_teams_before_window"})
        )

        last_internal_move = (
            transfer_history.sort_values(["player_id", "transfer_date"])
            .groupby("player_id", as_index=False)
            .tail(1)
            .copy()
        )
        last_internal_move["was_last_move_internal_category_change"] = last_internal_move[
            "is_internal_category_change"
        ].astype(int)

        transfer_features = transfer_counts.merge(distinct_destination_teams, on="player_id", how="left")
        transfer_features = transfer_features.merge(internal_move_counts, on="player_id", how="left")
        transfer_features = transfer_features.merge(
            last_transfer[
                [
                    "player_id",
                    "days_since_last_move",
                    "was_last_move_loan",
                    "was_last_move_return_from_loan",
                    "was_last_move_transfer",
                    "moved_in_last_365_days",
                ]
            ],
            on="player_id",
            how="left",
        )
        transfer_features = transfer_features.merge(
            last_internal_move[
                [
                    "player_id",
                    "was_last_move_internal_category_change",
                ]
            ],
            on="player_id",
            how="left",
        )
        transfer_features["summer_year"] = int(summer_year)
        transfer_feature_frames.append(transfer_features)

    if not transfer_feature_frames:
        return pd.DataFrame(columns=["player_id", "summer_year"])

    return pd.concat(transfer_feature_frames, ignore_index=True)


def build_injury_features(injuries, summer_years):
    """
    Build injury exposure features for every modeled summer year.

    Parameters
    ----------
    injuries : pd.DataFrame
        Injuries table.
    summer_years : iterable of int
        Summer years to model.

    Returns
    -------
    pd.DataFrame
        Injury features by player and summer year.
    """
    injury_feature_frames = []
    injuries = injuries.copy()
    injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
    injuries["end_date"] = pd.to_datetime(injuries["end_date"], errors="coerce")
    injuries["days_missed"] = injuries["days_missed"].fillna(0)
    injuries["games_missed"] = injuries["games_missed"].fillna(0)

    for summer_year in sorted(set(summer_years)):
        reference_date = get_reference_date(summer_year)
        lookback_date = reference_date - pd.Timedelta(days=365)

        injuries_last_year = injuries.loc[
            (injuries["from_date"] < reference_date)
            & (
                (injuries["from_date"] >= lookback_date)
                | (injuries["end_date"] >= lookback_date)
                | injuries["end_date"].isna()
            )
        ].copy()

        if injuries_last_year.empty:
            continue

        injury_aggregates = (
            injuries_last_year.groupby("player_id", as_index=False)
            .agg(
                nb_injuries_last_365_days=("injury_reason", "count"),
                total_injury_days_last_365_days=("days_missed", "sum"),
                total_games_missed_last_365_days=("games_missed", "sum"),
                nb_virus_injuries_last_365_days=("is_virus", "sum"),
            )
        )

        active_injury = injuries.loc[
            (injuries["from_date"] <= reference_date)
            & ((injuries["end_date"] >= reference_date) | injuries["end_date"].isna())
        ].copy()
        active_injury_flag = (
            active_injury.groupby("player_id")
            .size()
            .reset_index(name="is_injured_on_reference_date")
        )
        active_injury_flag["is_injured_on_reference_date"] = 1

        severe_injury_flag = (
            injuries_last_year.loc[injuries_last_year["days_missed"] >= 90]
            .groupby("player_id")
            .size()
            .reset_index(name="had_severe_injury_last_365_days")
        )
        severe_injury_flag["had_severe_injury_last_365_days"] = 1

        injury_features = injury_aggregates.merge(active_injury_flag, on="player_id", how="left")
        injury_features = injury_features.merge(severe_injury_flag, on="player_id", how="left")
        injury_features["summer_year"] = int(summer_year)
        injury_feature_frames.append(injury_features)

    if not injury_feature_frames:
        return pd.DataFrame(columns=["player_id", "summer_year"])

    return pd.concat(injury_feature_frames, ignore_index=True)


def build_national_team_features(national_performances, summer_years):
    """
    Build national team exposure features for every modeled summer year.

    Parameters
    ----------
    national_performances : pd.DataFrame
        National performances table.
    summer_years : iterable of int
        Summer years to model.

    Returns
    -------
    pd.DataFrame
        National team features by player and summer year.
    """
    national_feature_frames = []
    national_performances = national_performances.copy()
    national_performances["date_debut"] = pd.to_datetime(
        national_performances["date_debut"], errors="coerce"
    )
    national_performances["is_senior_team"] = (
        ~national_performances["team_name"].astype(str).str.contains(r"U\d", regex=True)
    ).astype(int)

    for summer_year in sorted(set(summer_years)):
        reference_date = get_reference_date(summer_year)
        lookback_date = reference_date - pd.Timedelta(days=365)

        national_history = national_performances.loc[
            national_performances["date_debut"] < reference_date
        ].copy()
        national_last_year = national_history.loc[national_history["date_debut"] >= lookback_date].copy()

        career_features = (
            national_history.groupby("player_id", as_index=False)
            .agg(
                nb_national_records_before_window=("team_name", "count"),
                total_national_caps_before_window=("games_played", "sum"),
                total_national_goals_before_window=("goals_scored", "sum"),
            )
        )

        last_year_features = (
            national_last_year.groupby("player_id", as_index=False)
            .agg(
                national_caps_last_365_days=("games_played", "sum"),
                national_goals_last_365_days=("goals_scored", "sum"),
                nb_national_matchesets_last_365_days=("team_name", "count"),
                has_senior_national_team_record=("is_senior_team", "max"),
            )
        )

        national_features = career_features.merge(last_year_features, on="player_id", how="outer")
        national_features["summer_year"] = int(summer_year)
        national_feature_frames.append(national_features)

    if not national_feature_frames:
        return pd.DataFrame(columns=["player_id", "summer_year"])

    return pd.concat(national_feature_frames, ignore_index=True)


def build_target(transfers, summer_years):
    """
    Build the target label for every player and modeled summer year.

    Parameters
    ----------
    transfers : pd.DataFrame
        Transfers table.
    summer_years : iterable of int
        Summer years to model.

    Returns
    -------
    pd.DataFrame
        Target table with one row per positive player-summer pair.
    """
    transfers = transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["summer_year"] = transfers["transfer_date"].dt.year
    transfers["is_internal_category_change"] = transfers.apply(
        lambda row: is_internal_category_change(row["from_team_name"], row["to_team_name"]),
        axis=1,
    )

    target_transfers = transfers.loc[
        transfers["transfer_type"].isin(["Transfer", "Loan"])
        & (transfers["summer_year"].isin(list(set(summer_years))))
        & (~transfers["is_internal_category_change"])
        & (
            (
                (transfers["transfer_date"].dt.month > SUMMER_WINDOW_START_MONTH - 1)
                & (transfers["transfer_date"].dt.month < SUMMER_WINDOW_END_MONTH)
            )
            | (
                (transfers["transfer_date"].dt.month == SUMMER_WINDOW_END_MONTH)
                & (transfers["transfer_date"].dt.day <= SUMMER_WINDOW_END_DAY)
            )
        )
        & (
            (transfers["transfer_date"].dt.month > SUMMER_WINDOW_START_MONTH)
            | (
                (transfers["transfer_date"].dt.month == SUMMER_WINDOW_START_MONTH)
                & (transfers["transfer_date"].dt.day >= SUMMER_WINDOW_START_DAY)
            )
        )
    ].copy()

    target_transfers = target_transfers[["player_id", "summer_year"]].drop_duplicates()
    target_transfers["target"] = 1
    return target_transfers


def add_performance_boost_features(modeling_dataset):
    """
    Add clear market-signal flags based on recent performance patterns.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Modeling dataset before final filling.

    Returns
    -------
    pd.DataFrame
        Modeling dataset enriched with performance boost features.
    """
    boosted_dataset = modeling_dataset.copy()
    boosted_dataset["goal_contributions_last_season"] = (
        boosted_dataset["goals_scored"].fillna(0) + boosted_dataset["assists_made"].fillna(0)
    )
    boosted_dataset["goal_contributions_per_90"] = (
        boosted_dataset["goals_per_90"].fillna(0) + boosted_dataset["assists_per_90"].fillna(0)
    )
    boosted_dataset["is_elite_goal_scorer"] = (
        (boosted_dataset["goals_scored"] >= 12) & (boosted_dataset["goals_per_90"] >= 0.55)
    ).astype(int)
    boosted_dataset["is_elite_assist_provider"] = (
        (boosted_dataset["assists_made"] >= 8) & (boosted_dataset["assists_per_90"] >= 0.28)
    ).astype(int)
    boosted_dataset["is_goal_contribution_star"] = (
        (boosted_dataset["goal_contributions_last_season"] >= 18)
        & (boosted_dataset["minutes_played"] >= 1200)
    ).astype(int)
    boosted_dataset["strong_previous_season_then_minutes_drop_no_injury"] = (
        (boosted_dataset["previous_minutes_played"] >= 1600)
        & (boosted_dataset["minutes_played"] <= 700)
        & (boosted_dataset["total_injury_days_last_365_days"] <= 30)
    ).astype(int)
    boosted_dataset["breakout_performer"] = (
        (boosted_dataset["minutes_played"] >= 1200)
        & (
            (boosted_dataset["goals_per_90_trend"] >= 0.18)
            | (boosted_dataset["assists_per_90_trend"] >= 0.12)
        )
    ).astype(int)
    boosted_dataset["senior_national_team_boost"] = (
        (boosted_dataset["has_senior_national_team_record"] == 1)
        & (boosted_dataset["national_caps_last_365_days"] >= 3)
    ).astype(int)
    boosted_dataset["elite_national_output_boost"] = (
        (boosted_dataset["national_caps_last_365_days"] >= 5)
        | (boosted_dataset["national_goals_last_365_days"] >= 2)
    ).astype(int)
    boosted_dataset["performance_boost_score"] = (
        boosted_dataset["is_elite_goal_scorer"]
        + boosted_dataset["is_elite_assist_provider"]
        + boosted_dataset["is_goal_contribution_star"]
        + boosted_dataset["strong_previous_season_then_minutes_drop_no_injury"]
        + boosted_dataset["breakout_performer"]
        + boosted_dataset["senior_national_team_boost"]
        + boosted_dataset["elite_national_output_boost"]
    )

    return boosted_dataset


def add_market_profile_features(modeling_dataset):
    """
    Add explicit football-market profile flags on top of raw performance signals.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Modeling dataset with historical market context already merged.

    Returns
    -------
    pd.DataFrame
        Modeling dataset enriched with interpretable market-profile features.
    """
    profiled_dataset = modeling_dataset.copy()
    citizenship_series = profiled_dataset["citizenship"].fillna("Unknown")

    profiled_dataset["team_is_high_churn_market"] = (
        (profiled_dataset["team_previous_market_move_rate"].fillna(0) >= 0.18)
        | (profiled_dataset["team_previous_market_move_count"].fillna(0) >= 4)
    ).astype(int)
    profiled_dataset["team_is_top5_exporter_market"] = (
        (profiled_dataset["team_previous_top5_export_rate"].fillna(0) >= 0.06)
        | (profiled_dataset["team_previous_top5_export_count"].fillna(0) >= 1)
    ).astype(int)
    profiled_dataset["competition_is_top5_export_market"] = (
        (profiled_dataset["competition_previous_top5_export_rate"].fillna(0) >= 0.08)
        | (profiled_dataset["competition_previous_top5_export_count"].fillna(0) >= 4)
    ).astype(int)
    profiled_dataset["competition_is_intra_league_market"] = (
        profiled_dataset["competition_previous_intra_league_rate"].fillna(0) >= 0.35
    ).astype(int)

    profiled_dataset["young_export_talent_profile"] = (
        (profiled_dataset["age_at_window"].fillna(99) <= 23)
        & (profiled_dataset["primary_competition_is_top5_league"].fillna(0) == 0)
        & (
            citizenship_series.isin(["Brazil", "Argentina"])
            | citizenship_series.isin(["France", "Spain", "Portugal"])
        )
        & (
            (profiled_dataset["is_goal_contribution_star"].fillna(0) == 1)
            | (profiled_dataset["breakout_performer"].fillna(0) == 1)
            | (profiled_dataset["performance_boost_score"].fillna(0) >= 2)
        )
    ).astype(int)

    profiled_dataset["english_domestic_market_profile"] = (
        (citizenship_series == "England")
        & (profiled_dataset["primary_competition_is_top5_league"].fillna(0) == 1)
        & (profiled_dataset["competition_is_intra_league_market"].fillna(0) == 1)
        & (profiled_dataset["minutes_played"].fillna(0) >= 900)
    ).astype(int)

    profiled_dataset["stalled_star_profile"] = (
        (profiled_dataset["previous_minutes_played"].fillna(0) >= 1800)
        & (
            (profiled_dataset["previous_goals_per_90"].fillna(0) >= 0.45)
            | (profiled_dataset["previous_assists_per_90"].fillna(0) >= 0.20)
        )
        & (profiled_dataset["minutes_played"].fillna(0) <= 900)
        & (profiled_dataset["total_injury_days_last_365_days"].fillna(0) <= 30)
    ).astype(int)

    profiled_dataset["national_team_showcase_profile"] = (
        (profiled_dataset["has_senior_national_team_record"].fillna(0) == 1)
        & (
            (profiled_dataset["national_caps_last_365_days"].fillna(0) >= 4)
            | (profiled_dataset["national_goals_last_365_days"].fillna(0) >= 2)
        )
        & (
            (profiled_dataset["performance_boost_score"].fillna(0) >= 1)
            | (profiled_dataset["minutes_played"].fillna(0) <= 1200)
        )
    ).astype(int)

    profiled_dataset["loan_relaunch_profile"] = (
        (profiled_dataset["was_last_move_loan"].fillna(0) == 1)
        & (profiled_dataset["minutes_played"].fillna(0) >= 1200)
        & (
            (profiled_dataset["goals_per_90"].fillna(0) >= 0.30)
            | (profiled_dataset["assists_per_90"].fillna(0) >= 0.20)
            | (profiled_dataset["performance_boost_score"].fillna(0) >= 1)
        )
    ).astype(int)

    profiled_dataset["top5_ready_profile"] = (
        (profiled_dataset["primary_competition_is_top5_league"].fillna(0) == 0)
        & (profiled_dataset["age_at_window"].fillna(99) <= 24)
        & (profiled_dataset["team_is_top5_exporter_market"].fillna(0) == 1)
        & (
            (profiled_dataset["performance_boost_score"].fillna(0) >= 1)
            | (
                (profiled_dataset["minutes_played"].fillna(0) >= 1500)
                & (
                    profiled_dataset["goals_per_90"].fillna(0)
                    + profiled_dataset["assists_per_90"].fillna(0)
                    >= 0.35
                )
            )
        )
    ).astype(int)

    profiled_dataset["market_profile_score"] = (
        profiled_dataset["team_is_high_churn_market"]
        + profiled_dataset["team_is_top5_exporter_market"]
        + profiled_dataset["competition_is_top5_export_market"]
        + profiled_dataset["young_export_talent_profile"]
        + profiled_dataset["english_domestic_market_profile"]
        + profiled_dataset["stalled_star_profile"]
        + profiled_dataset["national_team_showcase_profile"]
        + profiled_dataset["loan_relaunch_profile"]
        + profiled_dataset["top5_ready_profile"]
    )

    return profiled_dataset


def build_historical_market_context_features(modeling_dataset):
    """
    Build historical club and competition transfer-rate features.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Modeling dataset with club context and target columns.

    Returns
    -------
    pd.DataFrame
        Historical market context features by player and summer year.
    """
    historical_feature_frames = []

    for summer_year in sorted(modeling_dataset["summer_year"].unique()):
        current_rows = modeling_dataset.loc[modeling_dataset["summer_year"] == summer_year].copy()
        historical_rows = modeling_dataset.loc[modeling_dataset["summer_year"] < summer_year].copy()

        if historical_rows.empty:
            current_rows["team_previous_market_move_rate"] = 0.0
            current_rows["team_two_year_market_move_rate"] = 0.0
            current_rows["team_previous_market_move_count"] = 0.0
            current_rows["tier_previous_market_move_rate"] = 0.0
            current_rows["category_side_previous_market_move_rate"] = 0.0
            current_rows["competition_previous_market_move_rate"] = 0.0
            current_rows["competition_previous_market_move_count"] = 0.0
            current_rows["competition_previous_top5_export_rate"] = 0.0
            current_rows["competition_previous_top5_export_count"] = 0.0
            current_rows["competition_previous_intra_league_rate"] = 0.0
            current_rows["competition_previous_upward_move_rate"] = 0.0
            current_rows["team_previous_top5_export_rate"] = 0.0
            current_rows["team_previous_top5_export_count"] = 0.0
            historical_feature_frames.append(
                current_rows[
                    [
                        "player_id",
                        "summer_year",
                        "team_previous_market_move_rate",
                        "team_two_year_market_move_rate",
                        "team_previous_market_move_count",
                        "tier_previous_market_move_rate",
                        "category_side_previous_market_move_rate",
                        "competition_previous_market_move_rate",
                        "competition_previous_market_move_count",
                        "competition_previous_top5_export_rate",
                        "competition_previous_top5_export_count",
                        "competition_previous_intra_league_rate",
                        "competition_previous_upward_move_rate",
                        "team_previous_top5_export_rate",
                        "team_previous_top5_export_count",
                    ]
                ]
            )
            continue

        previous_year_rows = historical_rows.loc[historical_rows["summer_year"] == summer_year - 1].copy()
        last_two_year_rows = historical_rows.loc[historical_rows["summer_year"] >= summer_year - 2].copy()

        team_previous_rate = (
            previous_year_rows.groupby("primary_team_name", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "team_previous_market_move_rate"})
        )
        team_previous_count = (
            previous_year_rows.groupby("primary_team_name", as_index=False)["target"]
            .sum()
            .rename(columns={"target": "team_previous_market_move_count"})
        )
        team_two_year_rate = (
            last_two_year_rows.groupby("primary_team_name", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "team_two_year_market_move_rate"})
        )
        tier_previous_rate = (
            previous_year_rows.groupby("competition_tier_proxy", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "tier_previous_market_move_rate"})
        )
        category_previous_rate = (
            previous_year_rows.groupby("primary_team_is_category_side", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "category_side_previous_market_move_rate"})
        )
        competition_previous_rate = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "competition_previous_market_move_rate"})
        )
        competition_previous_count = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["target"]
            .sum()
            .rename(columns={"target": "competition_previous_market_move_count"})
        )
        competition_top5_export_rate = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["target_to_top5_league"]
            .mean()
            .rename(columns={"target_to_top5_league": "competition_previous_top5_export_rate"})
        )
        competition_top5_export_count = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["target_to_top5_league"]
            .sum()
            .rename(columns={"target_to_top5_league": "competition_previous_top5_export_count"})
        )
        competition_intra_league_rate = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["last_move_was_intra_league"]
            .mean()
            .rename(columns={"last_move_was_intra_league": "competition_previous_intra_league_rate"})
        )
        competition_upward_move_rate = (
            previous_year_rows.groupby("primary_competition_name", as_index=False)["last_move_was_upward_league"]
            .mean()
            .rename(columns={"last_move_was_upward_league": "competition_previous_upward_move_rate"})
        )
        team_top5_export_rate = (
            previous_year_rows.groupby("primary_team_name", as_index=False)["target_to_top5_league"]
            .mean()
            .rename(columns={"target_to_top5_league": "team_previous_top5_export_rate"})
        )
        team_top5_export_count = (
            previous_year_rows.groupby("primary_team_name", as_index=False)["target_to_top5_league"]
            .sum()
            .rename(columns={"target_to_top5_league": "team_previous_top5_export_count"})
        )

        current_rows = current_rows.merge(team_previous_rate, on="primary_team_name", how="left")
        current_rows = current_rows.merge(team_two_year_rate, on="primary_team_name", how="left")
        current_rows = current_rows.merge(team_previous_count, on="primary_team_name", how="left")
        current_rows = current_rows.merge(tier_previous_rate, on="competition_tier_proxy", how="left")
        current_rows = current_rows.merge(
            category_previous_rate,
            on="primary_team_is_category_side",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_previous_rate,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_previous_count,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_top5_export_rate,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_top5_export_count,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_intra_league_rate,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            competition_upward_move_rate,
            on="primary_competition_name",
            how="left",
        )
        current_rows = current_rows.merge(
            team_top5_export_rate,
            on="primary_team_name",
            how="left",
        )
        current_rows = current_rows.merge(
            team_top5_export_count,
            on="primary_team_name",
            how="left",
        )

        for feature_name in [
            "team_previous_market_move_rate",
            "team_two_year_market_move_rate",
            "team_previous_market_move_count",
            "tier_previous_market_move_rate",
            "category_side_previous_market_move_rate",
            "competition_previous_market_move_rate",
            "competition_previous_market_move_count",
            "competition_previous_top5_export_rate",
            "competition_previous_top5_export_count",
            "competition_previous_intra_league_rate",
            "competition_previous_upward_move_rate",
            "team_previous_top5_export_rate",
            "team_previous_top5_export_count",
        ]:
            current_rows[feature_name] = current_rows[feature_name].fillna(0.0)

        historical_feature_frames.append(
            current_rows[
                [
                    "player_id",
                    "summer_year",
                    "team_previous_market_move_rate",
                    "team_two_year_market_move_rate",
                    "team_previous_market_move_count",
                    "tier_previous_market_move_rate",
                    "category_side_previous_market_move_rate",
                    "competition_previous_market_move_rate",
                    "competition_previous_market_move_count",
                    "competition_previous_top5_export_rate",
                    "competition_previous_top5_export_count",
                    "competition_previous_intra_league_rate",
                    "competition_previous_upward_move_rate",
                    "team_previous_top5_export_rate",
                    "team_previous_top5_export_count",
                ]
            ]
        )

    return pd.concat(historical_feature_frames, ignore_index=True)


def fill_missing_feature_values(modeling_dataset):
    """
    Fill missing values in the modeling dataset with simple explicit rules.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Raw modeling table before imputation.

    Returns
    -------
    pd.DataFrame
        Modeling table with missing values filled.
    """
    modeling_dataset = modeling_dataset.copy()

    categorical_columns = [
        "citizenship",
        "main_position",
        "position",
        "foot",
    ]

    for categorical_column in categorical_columns:
        if categorical_column in modeling_dataset.columns:
            modeling_dataset[categorical_column] = modeling_dataset[categorical_column].fillna("Unknown")

    numeric_columns = modeling_dataset.select_dtypes(include=[np.number]).columns.tolist()
    for numeric_column in numeric_columns:
        modeling_dataset[numeric_column] = modeling_dataset[numeric_column].fillna(0)

    return modeling_dataset


def build_modeling_dataset(datasets):
    """
    Build the final player-summer dataset used for the first test.

    Parameters
    ----------
    datasets : dict
        Dictionary of source datasets.

    Returns
    -------
    pd.DataFrame
        Final modeling dataset.
    """
    performances = keep_european_seasons(datasets["performances"])
    reference_windows = build_reference_windows(performances)
    summer_years = sorted(reference_windows["summer_year"].unique())
    team_season_lookup = build_team_season_lookup(performances)

    performance_features = aggregate_performances(performances)
    performance_features = add_performance_trends(performance_features)
    club_context_features = build_club_context_features(performances)

    player_features = build_player_features(datasets["players"], reference_windows)
    transfer_features = build_transfer_features(datasets["transfers"], summer_years)
    injury_features = build_injury_features(datasets["injuries"], summer_years)
    national_team_features = build_national_team_features(
        datasets["national_performances"], summer_years
    )
    target_table = build_transfer_market_labels(datasets["transfers"], team_season_lookup, summer_years)

    modeling_dataset = reference_windows.merge(performance_features, on=["player_id", "season_name"], how="left")
    modeling_dataset = modeling_dataset.merge(
        club_context_features,
        on=["player_id", "season_name"],
        how="left",
    )
    modeling_dataset = modeling_dataset.merge(
        player_features.drop(columns=["reference_date"]),
        on=["player_id", "season_name", "summer_year"],
        how="left",
    )
    modeling_dataset = modeling_dataset.merge(transfer_features, on=["player_id", "summer_year"], how="left")
    modeling_dataset = modeling_dataset.merge(injury_features, on=["player_id", "summer_year"], how="left")
    modeling_dataset = modeling_dataset.merge(
        national_team_features, on=["player_id", "summer_year"], how="left"
    )
    modeling_dataset = modeling_dataset.merge(target_table, on=["player_id", "summer_year"], how="left")
    for target_column in [
        "target",
        "target_to_top5_league",
        "last_move_was_intra_league",
        "last_move_was_upward_league",
    ]:
        modeling_dataset[target_column] = modeling_dataset[target_column].fillna(0).astype(int)
    modeling_dataset = add_performance_boost_features(modeling_dataset)
    historical_market_context = build_historical_market_context_features(modeling_dataset)
    modeling_dataset = modeling_dataset.merge(
        historical_market_context,
        on=["player_id", "summer_year"],
        how="left",
    )
    modeling_dataset = add_market_profile_features(modeling_dataset)
    modeling_dataset = fill_missing_feature_values(modeling_dataset)

    return modeling_dataset.sort_values(["summer_year", "player_id"]).reset_index(drop=True)


def save_modeling_dataset(modeling_dataset, output_path):
    """
    Save the final modeling dataset to CSV.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Final modeling dataset.
    output_path : str or Path
        Output CSV path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_dataset.to_csv(output_path, index=False)
