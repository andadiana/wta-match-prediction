import pandas as pd
import re

PLAYER_NAME = "Simona Halep"
RESULT_FILENAME = "data/wta/matches_Simona_Halep.csv"

COLUMNS = ['best_of', 'draw_size', 'loser_age', 'loser_entry', 'loser_hand', 'loser_ht', 'loser_id',
           'loser_ioc', 'loser_name', 'loser_rank', 'loser_rank_points', 'loser_seed', 'match_num',
           'minutes', 'round', 'score', 'surface', 'tourney_date', 'tourney_id', 'tourney_level',
           'tourney_name', 'winner_age', 'winner_entry', 'winner_hand', 'winner_ht', 'winner_id',
           'winner_ioc', 'winner_name', 'winner_rank', 'winner_rank_points', 'winner_seed']


TWO_SET_1 = '^(6-[0-4] 6-[0-4])'
TWO_SET_2 = '^((6-[0-4] 7-[5-6](\([0-9]\))?)|(7-[5-6](\([0-9]\))? 6-[0-4]))'
TWO_SET_3 = '^(7-[5-6](\([0-9]\))? 7-[5-6](\([0-9]\))?)'
THREE_SET_1 = '^((([6-7]-[0-6](\([0-9]\))? [0-6]-[6-7](\([0-9]\))? 6-[0-3]))|' \
              '([0-6]-[6-7](\([0-9]\))? [6-7]-[0-6](\([0-9]\))? 6-[0-3]))'
THREE_SET_2 = '^((([6-7]-[0-6](\([0-9]\))? [0-6]-[6-7](\([0-9]\))? ([6-9]|[0-9]{2})-([4-9]|[0-9]{2})(\([0-9]\))?))|' \
              '([0-6]-[6-7](\([0-9]\))? [6-7]-[0-6](\([0-9]\))? ([6-9]|[0-9]{2})-([4-9]|[0-9]{2})(\([0-9]\))?))'


two_set_1_regex = re.compile(TWO_SET_1)
two_set_2_regex = re.compile(TWO_SET_2)
two_set_3_regex = re.compile(TWO_SET_3)
three_set_1_regex = re.compile(THREE_SET_1)
three_set_2_regex = re.compile(THREE_SET_2)


ROUND_ORDERING = ['RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']


def encode_score(score):
    global two_set_regex
    global three_set_regex

    if score == 'W/O':
        # walkover
        return score
    elif 'RET' in score:
        # retired match
        return 'RET'
    elif two_set_1_regex.match(score):
        # 2 set match - first category - easy win
        return 'EASY_WIN'
    elif two_set_2_regex.match(score):
        # 2 set match - second category - medium win
        return 'MEDIUM_WIN'
    elif two_set_3_regex.match(score):
        # 2 set match - third category - hard win
        return 'HARD_WIN'
    elif three_set_1_regex.match(score):
        # 3 set match - second category - medium win
        return 'MEDIUM_WIN'
    elif three_set_2_regex.match(score):
        # 3 set match = third category - hard win
        return 'HARD_WIN'
    else:
        return 'OTHER'


def get_last_meetings_score(player_df):
    for index, row in player_df.iterrows():
        opponent_name = row['opponent_name']
        match_date = row['tourney_date']
        previous_meetings = player_df[(player_df.opponent_name == opponent_name) & (player_df.tourney_date < match_date)]
        if not previous_meetings.empty:
            previous_meetings.sort_values('tourney_date', inplace=True, ascending=False)
            player_df.at[index,'previous_meeting_score'] = previous_meetings.iloc[0]['score_category']
            player_df.at[index, 'previous_meeting_tourney'] = previous_meetings.iloc[0]['tourney_name']
            player_df.at[index, 'previous_meeting_tourney_id'] = previous_meetings.iloc[0]['tourney_id']
            player_df.at[index, 'previous_meeting_tourney_date'] = previous_meetings.iloc[0]['tourney_date']

            # print(previous_meetings)
    # player_df.drop('score', inplace=True, axis=1)
    return player_df

#
# def get_last_player_match_score(player_df):
#     player_df['tourney_round'] = pd.Categorical(player_df['tourney_round'], categories=ROUND_ORDERING)
#     for index, row in player_df.iterrows():
#         match_date = row['tourney_date']
#
#         previous_matches = player_df[(player_df.tourney_date <= match_date) & (player_df.index != index)]
#         if not previous_matches.empty:
#             previous_matches.sort_values(['tourney_round', 'tourney_date'], inplace=True, ascending=[False, False])
#             player_df.at[index,'previous_match_score'] = previous_matches.iloc[0]['score_category']
#             player_df.at[index, 'previous_match_tourney'] = previous_matches.iloc[0]['tourney_name']
#             player_df.at[index, 'previous_match_tourney_id'] = previous_matches.iloc[0]['tourney_id']
#             player_df.at[index, 'previous_match_tourney_date'] = previous_matches.iloc[0]['tourney_date']
#
#             # print(previous_meetings)
#     # player_df.drop('score', inplace=True, axis=1)
#     return player_df


def encode_score_column(player_df):
    for index, row in player_df.iterrows():
        encoded_score = encode_score(row.score)
        print("Old score ", row.score, " encoded score: ", encoded_score)
        # print(row.score)
        player_df.at[index, 'score_category'] = encoded_score
    # player_df.drop('score', inplace=True, axis=1)
    return player_df


def fill_nan(matches_df):
    # Fill in the missing ranking - for unranked players - use a large value
    matches_df.loser_rank = matches_df.loser_rank.fillna(value=1500)
    matches_df.winner_rank = matches_df.winner_rank.fillna(value=1500)
    # Fill in the missing ranking points - 0
    matches_df.loser_rank_points = matches_df.loser_rank_points.fillna(0)
    matches_df.winner_rank_points = matches_df.winner_rank_points.fillna(0)

    # Fill in missing height for opponents - use average height
    average_ht = (matches_df.loser_ht.mean() + matches_df.winner_ht.mean()) / 2
    print("Average height is: ", average_ht)
    matches_df.loser_ht =  matches_df.loser_ht.fillna(value=average_ht)
    matches_df.winner_ht = matches_df.winner_ht.fillna(value=average_ht)

    return matches_df


def main():
    # Import dataset
    all_matches = pd.read_csv("data/wta/matches.csv", low_memory=False)
    matches_2017 = pd.read_csv("data/wta/wta_matches_2017.csv", low_memory=False)
    matches_2018 = pd.read_csv("data/wta/wta_matches_2018.csv", low_memory=False)

    matches_2017 = matches_2017[COLUMNS]
    matches_2017['year'] = 2017
    matches_2018 = matches_2018[COLUMNS]
    matches_2018['year'] = 2018

    all_matches = pd.concat([all_matches, matches_2017, matches_2018])

    # Fill nan values
    all_matches = fill_nan(all_matches)

    # Filter matches for given player
    player_matches = all_matches[(all_matches.loser_name == PLAYER_NAME) | (all_matches.winner_name == PLAYER_NAME)]
    player_matches = encode_score_column(player_matches)

    # Create a new dataframe from the point of view of the player

    # One dateframe for the wins
    winner_df = player_matches[(player_matches.winner_name == PLAYER_NAME)]
    winner_df['win'] = 1
    winner_df.rename(columns={'loser_age': 'opponent_age', 'loser_entry': 'opponent_entry',
                              'loser_hand': 'opponent_hand', 'loser_ht': 'opponent_ht',
                              'loser_id': 'opponent_id', 'loser_ioc': 'opponent_ioc',
                              'loser_name': 'opponent_name', 'loser_rank': 'opponent_rank',
                              'loser_rank_points': 'opponent_rank_points', 'loser_seed': 'opponent_seed',
                              'winner_age': 'player_age', 'winner_entry': 'player_entry',
                              'winner_hand': 'player_hand', 'winner_ht': 'player_ht',
                              'winner_id': 'player_id', 'winner_ioc': 'player_ioc',
                              'winner_name': 'player_name', 'winner_rank': 'player_rank',
                              'winner_rank_points': 'player_rank_points', 'winner_seed': 'player_seed'}, inplace=True)
    print(winner_df.head())

    # One dataframe for the losses
    loser_df = player_matches[(player_matches.loser_name == PLAYER_NAME)]
    loser_df['win'] = 0
    loser_df.rename(columns={'winner_age': 'opponent_age', 'winner_entry': 'opponent_entry',
                             'winner_hand': 'opponent_hand', 'winner_ht': 'opponent_ht',
                             'winner_id': 'opponent_id', 'winner_ioc': 'opponent_ioc',
                             'winner_name': 'opponent_name', 'winner_rank': 'opponent_rank',
                             'winner_rank_points': 'opponent_rank_points', 'winner_seed': 'opponent_seed',
                             'loser_age': 'player_age', 'loser_entry': 'player_entry',
                             'loser_hand': 'player_hand', 'loser_ht': 'player_ht',
                             'loser_id': 'player_id', 'loser_ioc': 'player_ioc',
                             'loser_name': 'player_name', 'loser_rank': 'player_rank',
                             'loser_rank_points': 'player_rank_points', 'loser_seed': 'player_seed'}, inplace=True)
    loser_df['score_category'] = loser_df['score_category'].map(lambda s: 'HARD_LOSS' if s == 'EASY_WIN' else ('EASY_LOSS' if s == 'HARD_WIN' else ('MEDIUM_LOSS' if s == 'MEDIUM_WIN' else s)))
    print(loser_df.head())

    # Concatenate the two dataframes into a final one, containing all the matches (wins and losses) for the given player
    player_df = pd.concat([winner_df, loser_df])
    player_df.rename(columns={'round': 'tourney_round'}, inplace=True)
    print(player_df.head())
    print(player_df[['score_category', 'score', 'win']])

    # Columns for last meeting between players
    player_df = get_last_meetings_score(player_df)
    # player_df = get_last_player_match_score(player_df)
    print(player_df[['tourney_date', 'opponent_name', 'previous_meeting_score']])

    # Save dataframe to CSV
    player_df.to_csv(RESULT_FILENAME, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()