# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_data():
    assigned_names = ['Timestamp', 'Age', 'Gender', 'Play_frequency', 'Study_videogames', 'Work_videogames',
                      'Time_change',
                      'Slowed_down', 'Smaller_self', 'Oneness_things', 'Humbling', 'Connected_humanity',
                      'Something_greater',
                      'Chills_goosebumps', 'All_at_once_struggle', 'Mentally_challenged', 'Positive_impact',
                      'Fear_discomfort',
                      'Peace_of_mind', 'Admiration_game_dev', 'Better_person', 'Comments1', 'Title', 'Genre',
                      'Graphics_rating',
                      'Graphics_descr', 'Story_rating', 'Story_descr', 'Soundtrack_rating', 'Soundtrack_descr',
                      'Main_character_rating', 'Main_char_descr', 'Locations', 'Pace_and_difficulty', 'VR', 'VR_descr',
                      'Comments2']
    df = pd.read_csv('MT survey (Responses).tsv', sep='\t', header=0,
                     skiprows=0)
    # print(df.head(5))
    df.columns = assigned_names
    return df


def drop_irrelevant_columns(df, col):
    # remove columns that are not useful from the dataframe
    df = df.drop(col, axis=1)
    print(df.info)
    return df


def check_awe(row):  # check if a row of the dataset corresponds to an awe experience (true) or not (false)
    if row['Mentally_challenged'] == 1 or row['All_at_once_struggle'] == 1:  # main component of awe is not experienced
        return False
    #    coherence = row['Mentally_challenged'] - row['All_at_once_struggle']
    #    if coherence <= -2 or coherence >= 2:  # respondent might have not put enough effort in the survey
    #        return False
    count = 0  # how many of the important items were rated 1 ('Strongly disagree')
    for i in range(0, 12):
        if row[i] == 1:
            count = count + 1
    # if we got to the for loop, it means neither of mentally_challenged or struggle were ones, so we have to reduce
    # our "max" count from 12 (total features) to 10. Therefore, if 6 or more of these aspects were rated as ones, maybe
    # the experience was not as intense as I would like to know...
    if count >= 6:
        return False
    return True


def clean_dataset(df):
    indexes = []
    num_rows = df.shape[0]
    for i in range(0, num_rows):
        if not check_awe(df.iloc[i]):
            indexes.append(i)
    df = df.drop(indexes, axis=0)
    drop_awe = ['Time_change', 'Slowed_down', 'Smaller_self', 'Oneness_things', 'Humbling', 'Connected_humanity',
                'Something_greater', 'Chills_goosebumps', 'All_at_once_struggle', 'Mentally_challenged',
                'Positive_impact', 'Fear_discomfort', 'Peace_of_mind', 'Admiration_game_dev', 'Better_person']
    df = drop_irrelevant_columns(df, drop_awe)
    print(df.info())
    return df


def generate_ratings(df):
    to_drop = ['Graphics_descr', 'Story_descr', 'Soundtrack_descr', 'Main_char_descr', 'VR_descr', 'Comments2']
    return drop_irrelevant_columns(df, to_drop)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = read_data()
    general_info = ['Timestamp', 'Age', 'Gender', 'Play_frequency', 'Study_videogames', 'Work_videogames', 'Comments1']
    data = drop_irrelevant_columns(data, general_info)

    clean_data = clean_dataset(data)

#    ratings = generate_ratings(clean_data)
#    print(ratings.info())

    # todo: reduce ratings so that every game appears only once, with the average of all the ratings per each column
#    unique_games = clean_data.groupby(['Title'], axis=0, as_index=False)
    unique_games = clean_data.agg()
    print(unique_games.info())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
