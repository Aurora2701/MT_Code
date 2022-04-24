# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
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


def clean_dataset(df):
    indexes = []  # track rows to remove (not an awe experience)
    num_rows = df.shape[0]
    for i in range(0, num_rows):
        if not is_awe(df.iloc[i]):
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
    general_info = ['Timestamp', 'Comments1', 'Comments2']
    data = drop_irrelevant_columns(data, general_info)

    # todo: manually clean columns for main character, maybe locations...
    clean_data = clean_dataset(data)    # todo: check straight-lining...

    # todo: check feature cross_correlation
    # todo: create label column
    # todo: remove parenthesis in genre column + make a list out of comma separated items
    # todo: decide how to treat each NaN
    # todo: try merging answers relative to the same game but in separate df, just with the ratings (makes no sense
    #       to average genre, age of participants, gender, and whatever)
    fig, axes = plt.subplots()
    corr = clean_data.corr(method='spearman', min_periods=0)
    print(corr.shape)
    sns.heatmap(corr, annot=True, mask=np.zeros_like(corr, dtype=bool), ax=axes)
    plt.show()

#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
