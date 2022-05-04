# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


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
    # remove columns that are not useful from the dataframe [mostly for debugging purposes]
    df = df.drop(col, axis=1)
    print(df.info)
    return df


def is_awe(row):  # check if a row of the dataset corresponds to an awe experience (true) or not (false)
    if row['Mentally_challenged'] == 1 or row['All_at_once_struggle'] == 1:  # main component of awe is not experienced
        return False

    count = 0  # how many of the important items were rated 1 ('Strongly disagree')
    for i in range(0, 12):
        if row[i] == 1:
            count = count + 1
    # if we got to the for loop, it means neither of mentally_challenged or struggle were 1, so we have to reduce
    # our "max" count from 12 (total features) to 10. Therefore, if 6 or more of these aspects were rated as ones, maybe
    # the experience was not as intense as I would like to know...
    if count >= 6:
        return False
    return True  # passed both checks


def lazy(row):
    answer_streak = 1   # count how many consecutive answers are the same (straight-lining)
    for i in range(1, 15):
        if row[i] == row[i-1]:
            answer_streak = answer_streak + 1
            if answer_streak >= 7:  # as soon as the streak reaches 7 out of 15, the respondent is considered lazy
                return True
    return False


def check_pairs(row):
    # if the rating is empty, the description should be 'not applicable'
    # if the rating is not empty, the description should not be 'not applicable'
    if row['Story_rating'] == np.NaN and row['Story_descr'] != 'not applicable':
        return False
    if row['Story_rating'] != np.NaN and row['Story_descr'] == 'not applicable':
        return False
    if row['Main_character_rating'] == np.NaN and row['Main_char_descr'] != 'Not applicable':
        return False
    if row['Main_character_rating'] != np.NaN and row['Main_char_descr'] == 'Not applicable':
        return False
    return True


def remove_lazy(df):
    indexes = []  # track rows to remove (respondent did not put effort into the survey)
    num_rows = df.shape[0]
    for i in range(0, num_rows):
        if lazy(df.iloc[i]):
            indexes.append(i)
        if not check_pairs(df.iloc[i]):
            indexes.append(i)
    df = df.drop(indexes, axis=0)
    print(df.info())
    return df


def create_label(df):
    labels = []  # list that will become the new column
    num_rows = df.shape[0]
    for i in range(0, num_rows):
        labels.append(is_awe(df.iloc[i]))
    df['Felt_awe'] = labels
    drop_awe = ['Time_change', 'Slowed_down', 'Smaller_self', 'Oneness_things', 'Humbling', 'Connected_humanity',
                'Something_greater', 'Chills_goosebumps', 'All_at_once_struggle', 'Mentally_challenged',
                'Positive_impact', 'Fear_discomfort', 'Peace_of_mind', 'Admiration_game_dev', 'Better_person']
    df = drop_irrelevant_columns(df, drop_awe)
    print(df.info())
    return df


def replace(df):
    # first, remove the parenthesis because they were just supposed to help respondents
    df.Genre = df.Genre.str.replace(' \([^)]*\)', '', regex=True)  # grazie stackOverflow
    # second, convert yes and no into true and false (so we already have boolean types in some answers)
    df = df.replace({'Study_videogames': 'Yes', 'Work_videogames': 'Yes'}, True)
    df = df.replace({'Study_videogames': 'No', 'Work_videogames': 'No'}, False)
    # now the part that I have to do because I ~fucked up~ intentionally wrote the answers in the poll with commas for
    # legibility and ease of understanding ;)
    df.Main_char_descr = df.Main_char_descr.str.replace('(, at least partially,) | (, to a certain extent)', ' ',
                                                        regex=True)
    df.Story_descr = df.Story_descr.str.replace('weak, contains plot holes', 'weak and containing plot holes')
    df.Soundtrack_descr = df.Soundtrack_descr.str.replace('on spot, perfect for the game',
                                                          'on spot; perfect for the game')
    df.Soundtrack_descr = df.Soundtrack_descr.str.replace('irrelevant, left me indifferent',
                                                          'irrelevant; left me indifferent')
    df.Locations = df.Locations.str.replace('woods, forests', 'woods/forests')
    df.Locations = df.Locations.str.replace('sea, ocean', 'sea/ocean')
    df.Locations = df.Locations.str.replace('space, spacecraft', 'space/spacecraft')
    df.Pace_and_difficulty = df.Pace_and_difficulty.str.replace('the game was challenging, sometimes too much',
                                                                'the game was challenging - sometimes too much')
    # finally, replace NaNs:
    df = df.replace({'Story_rating': np.nan, 'Main_character_rating': np.nan}, 0)
    df = df.astype({'Story_rating': np.int64, 'Main_character_rating': np.int64})
    print(df.info())
    return df


def generate_ratings(df):
    to_drop = ['Graphics_descr', 'Story_descr', 'Soundtrack_descr', 'Main_char_descr', 'VR_descr']
    return drop_irrelevant_columns(df, to_drop)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = read_data()
    general_info = ['Timestamp', 'Comments1', 'Comments2']
    data = drop_irrelevant_columns(data, general_info)

    data = remove_lazy(data)
    awe_data = create_label(data)

    awe_data = replace(awe_data)    # adjust text answers

    # todo: make objects into categorical values
    # todo: use one-hot encoding for genres, and all descriptions
    # it's going to be a big dataset but it's probably necessary for decision trees

    # todo: try merging answers relative to the same game but in separate df, and just with the ratings (makes no sense
    #       to average genre, age of participants, gender, and whatever)

    # todo: create decision tree I guess

#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
