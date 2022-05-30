# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import sklearn.tree
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_validate
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
    # print(df.info)
    return df


def is_awe(row):    # uses column indexes
    # check if a row of the dataset corresponds to an awe experience (true) or not (false)
    if row['Mentally_challenged'] == 1 or row['All_at_once_struggle'] == 1:  # main component of awe is not experienced
        return 0  # interesting note: result doesn't change if I use an AND instead of an OR

    count = 0  # how many of the important items were rated 1 ('Strongly disagree')
    for i in range(0, 12):  # the first 12 columns *of the second section* measure awe
        if row[i + 5] == 1:
            count = count + 1
    # if we got to the for loop, it means neither of mentally_challenged or struggle were 1, so we have to reduce
    # our "max" count from 12 (total features) to 10. Therefore, if 6 or more of these aspects were rated as ones, maybe
    # the experience was not as intense as I would like to know...
    if count >= 6:
        return 0
    return 1  # passed both checks


def lazy(row):  # uses column indexes
    answer_streak = 1  # count how many consecutive answers *in the second section* are the same (straight-lining)
    for i in range(6, 20):
        if row[i] == row[i - 1]:
            answer_streak = answer_streak + 1
            if answer_streak >= 8:  # as soon as the streak reaches 8 out of 15, the respondent is considered lazy
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
    # print(df.info())
    return df


def create_label(df):
    labels = []  # list that will become the new column (bool)
    num_rows = df.shape[0]

    for i in range(0, num_rows):
        labels.append(is_awe(df.iloc[i]))  # append True if this row corresponds to an awe experience, False otherwise

    df['Felt_awe'] = labels  # add new column to the dataframe

    drop_awe = ['Time_change', 'Slowed_down', 'Smaller_self', 'Oneness_things', 'Humbling', 'Connected_humanity',
                'Something_greater', 'Chills_goosebumps', 'All_at_once_struggle', 'Mentally_challenged',
                'Positive_impact', 'Fear_discomfort', 'Peace_of_mind', 'Admiration_game_dev', 'Better_person']
    df = drop_irrelevant_columns(df, drop_awe)  # the columns about awe are not needed anymore
    # print(df.info())
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

    # finally, replace NaNs (and VR ^^'):
    df = df.replace({'Story_rating': np.nan, 'Main_character_rating': np.nan, 'VR': 'No'}, 0)
    df = df.replace({'VR': 'Yes'}, 5)
    df = df.astype({'Story_rating': np.int64, 'Main_character_rating': np.int64, 'VR': np.int64})

    # print(df.info())
    return df


def make_columns_from_lists(dframe, range_min=6, range_max=13):
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame()
    for i in range(range_min, range_max):  # columns from 'genre' = 6 to 'vr_descr' = 12
        tmp = pd.DataFrame(mlb.fit_transform(dframe.iloc[:, i]), columns=mlb.classes_, index=dframe.index)
        res = pd.concat([res, tmp], axis=1)
    print(res.info())
    return res


def get_ratings(source, cols):
    # Good argument for not grouping: even if the game is the same, different users might have different opinions
    #                                 also, we have a problem now that I'm including gender and play frequency ^^'
    # If grouping, ratings are apparently enough to give 100% accuracy already at depth 1
    # If not grouping but with class_weight='balanced', the tree is shitty (probably overfitting) at any depth

    res = pd.DataFrame(source[cols])
    # res['label'] = source['Felt_awe']
    # res = res.groupby(['Title'], sort=False).mean()     # drops title column automatically
    # res[res['label'] >= 0.5] = 1
    # res[res['label'] != 1] = 0
    # res['label'] = res['label'].astype(np.int64)
    res = res.drop('Title', axis=1)
    return res


def make_categories_and_one_hot(df):  # uses column indexes
    # order items alphabetically and make lists
    objects = ['Genre', 'Graphics_descr', 'Story_descr', 'Soundtrack_descr', 'Main_char_descr', 'Locations',
               'Pace_and_difficulty', 'VR_descr']
    for obj in objects:
        df[obj] = [','.join(sorted(i.split(', '))) for i in df[obj]]
        df[obj] = df[obj].str.split(',')    # to make a list again
        # print(df[obj])

    # before returning, split lists into multiple columns and one-hpt encode them
    # todo: you're forgetting gender, play freq and title in the returned df - they should probably also be one-hot enc
    df = make_columns_from_lists(df)
    print(df.info())
    return df


def get_description_df(df, column):
    # order items alphabetically and make lists out of them
    df[column] = [','.join(sorted(i.split(', '))) for i in df[column]]
    df[column] = df[column].str.split(',')  # to make a list again
    # one hot encoding
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame()
    tmp = pd.DataFrame(mlb.fit_transform(df[column]), columns=mlb.classes_, index=df.index)
    res = pd.concat([res, tmp], axis=1)
    res['label'] = df['Felt_awe']
    print(res.info())
    return res


def decision_tree(df, criterion='gini', n=13):
    # 1st attempt: just use ratings
    # 2nd attempt: use ratings after group by - everything gives accuracy 1, which worries me
    # 3rd attempt: use ratings + demographics - it's decent, although tr_acc=89% and val_acc=65%
    tr_accuracies = {}
    val_accuracies = {}
    x = df.iloc[:, :n].to_numpy()
    y = df.label.to_numpy()
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=12)
    kf = KFold(8)

    for i in range(5, 20):      # max depth
        clf = sklearn.tree.DecisionTreeClassifier(criterion=criterion, max_depth=i, random_state=12,
                                                  class_weight='balanced')
        tr_accuracies[i] = []
        val_accuracies[i] = []
        for train_index, test_index in kf.split(x, y):
            x_train, y_train, x_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_train)
            y_test_pred = clf.predict(x_test)

            train_acc = accuracy_score(y_train, y_pred)
            val_acc = accuracy_score(y_test, y_test_pred)
            tr_accuracies[i].append(train_acc)
            val_accuracies[i].append(val_acc)

        avg_tr_acc = np.average(tr_accuracies[i])
        avg_val_acc = np.average(val_accuracies[i])
        # print('Training and validation accuracies at depth ', i, ': ')     # Best depth before overfitting is 6-7
        # print(tr_accuracies[i])
        # print(val_accuracies[i])
        # cm = confusion_matrix(y_test, y_test_pred)
        # print('Confusion matrix at depth ', i, ': ')
        # print(cm)
        print('Average training accuracy at depth ', i, ': ', avg_tr_acc)
        print('Average validation accuracy at depth ', i, ': ', avg_val_acc)
    # cm = confusion_matrix(y_val, y_pred)
    # print('Confusion matrix (default): ', cm)
    # print(clf.feature_names_in_)
    # print(clf.feature_importances_)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = read_data()
    general_info = ['Timestamp', 'Comments1', 'Comments2']
    data = drop_irrelevant_columns(data, general_info)

    data = remove_lazy(data)
    awe_data = create_label(data)

    awe_data = replace(awe_data)  # adjust text answers

    # todo: forgetting age, need to make ranges
    objects = ['Gender', 'Play_frequency']      # title not a category for now
    for obj in objects:
        awe_data[obj] = awe_data[obj].astype('category')

    # split dataset in ratings + demography vs other data
    ratings_names = ['Title', 'Graphics_rating', 'Story_rating', 'Soundtrack_rating', 'Main_character_rating', 'VR']
    ratings = get_ratings(awe_data, ratings_names)
    # add one-hot encoded gender, play frequency
    ratings = pd.concat([ratings, pd.get_dummies(awe_data.iloc[:, 1], prefix='is')], axis=1)
    ratings = pd.concat([ratings, pd.get_dummies(awe_data.iloc[:, 2], prefix='plays')], axis=1)
    ratings['label'] = awe_data['Felt_awe']
    print(ratings.info())
    ratings_names.remove('Title')
    awe_data = drop_irrelevant_columns(awe_data, ratings_names)

    # final_data = make_categories_and_one_hot(awe_data)
    graphics_description = get_description_df(awe_data, 'Graphics_descr')

    # So, at this point, we have a DF with title (?), ratings, and label,
    # and another DF with almost everything one-hot encoded

    # Time to create the stupid tree(s) and start feeding them data. Which one? We'll see
    decision_tree(graphics_description, n=34)   # best max depth = 8
    decision_tree(ratings)      # best max depth = 15
    # From confusion matrices, the 'best' splitter seems better than the random one, having fewer mispredictions in the
    # second row (5-29 vs 3-31), so I'm dropping that already

    # todo: use select from model. Also, use it multiple times - because trees don't work well with too many features,
    #       split them in several rating + descriptions tables, and then find feature importance. Finally, test overall
    #       accuracy with the most important features from each group

#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
