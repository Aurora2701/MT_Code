# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt


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


def is_awe(row):
    # check if a row of the dataset corresponds to an awe experience (1) or not (0)
    if row['Mentally_challenged'] == 1 or row['All_at_once_struggle'] == 1:  # main component of awe is not experienced
        return 0

    count = 0  # how many of the relevant items were rated 1 ('Strongly disagree')
    for i in range(0, 12):  # the first 12 columns *of the second section* measure awe
        if row[i + 5] == 1:     # i + 5 to skip the answers of the first section
            count = count + 1
    # if we got to the for loop, it means neither of mentally_challenged or struggle were 1, so we have to reduce
    # our "max" count from 12 (total features) to 10. Therefore, if 6 or more of these aspects were rated as ones
    # the experience is not considered awe
    if count >= 6:
        return 0
    return 1  # passed both checks


def lazy(row):
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
    if row['VR'] == 'No' and row['VR_descr'] != 'not applicable':
        return False
    if row['VR'] == 'Yes' and row['VR_descr'] == 'not applicable':
        return False

    return True


def remove_lazy(df):
    indexes = []  # track rows to remove (respondent did not put effort into the survey)
    num_rows = df.shape[0]
    for i in range(0, num_rows):
        if lazy(df.iloc[i]):
            indexes.append(i)
            continue
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
    print("Number of awe experiences: ", labels.count(1))
    return df


def replace(df):
    # first, remove the parenthesis because they were just supposed to help respondents
    df.Genre = df.Genre.str.replace(' \([^)]*\)', '', regex=True)

    # second, convert yes and no into true and false (so we already have boolean types in some answers)
    df = df.replace({'Study_videogames': 'Yes', 'Work_videogames': 'Yes', 'VR': 'Yes'}, True)
    df = df.replace({'Study_videogames': 'No', 'Work_videogames': 'No', 'VR': 'No'}, False)

    # now change answers in the poll that contained commas for legibility and ease of understanding
    df.Main_char_descr = df.Main_char_descr.str.replace(', at least partially,', ' ', regex=True)
    df.Main_char_descr = df.Main_char_descr.str.replace(', to a certain extent', ' ', regex=True)
    df.Story_descr = df.Story_descr.str.replace('weak, contains plot holes', 'weak and containing plot holes')
    df.Soundtrack_descr = df.Soundtrack_descr.str.replace('on spot, perfect for the game',
                                                          'on spot; perfect for the game')
    df.Soundtrack_descr = df.Soundtrack_descr.str.replace('irrelevant, left me indifferent',
                                                          'irrelevant; left me indifferent')
    df.Locations = df.Locations.str.replace('woods, forests', 'woods/forests')
    df.Locations = df.Locations.str.replace('sea, ocean', 'sea/ocean')
    df.Locations = df.Locations.str.replace('space, spacecraft', 'space/spacecraft')
    df.Pace_and_difficulty = df.Pace_and_difficulty.str.replace('the game was challenging, sometimes too much',
                                                                'challenging')

    # finally, replace NaNs:
    df = df.replace({'Story_rating': np.nan, 'Main_character_rating': np.nan}, 0)
    df = df.astype({'Story_rating': np.int64, 'Main_character_rating': np.int64})

    # print(df.info())
    return df


def get_ratings(source, cols):
    res = pd.DataFrame(source[cols])

    objects = ['Play_frequency']
    for obj in objects:
        awe_data[obj] = awe_data[obj].astype('category')

    # add one-hot encoded play frequency
    res = pd.concat([res, pd.get_dummies(source.iloc[:, 2], prefix='plays')], axis=1)

    return res


def get_description_df(df, column):
    # order items alphabetically and make lists out of them
    df[column] = [','.join(sorted(i.split(', '))) for i in df[column]]
    df[column] = df[column].str.split(',')  # to make a list again
    # one hot encoding
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame()
    tmp = pd.DataFrame(mlb.fit_transform(df[column]), columns=mlb.classes_, index=df.index)
    res = pd.concat([res, tmp], axis=1)
    # print(res.info())
    return res


def final_decision_tree(df, y, depth, msl=1):
    # creates the tree at the best depth, and returns most important features
    n = df.shape[1]
    X = df.iloc[:, :n].to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # makes no sense to use k-fold coz I'm not looking for accuracy here, the feat are selected from one clf either way

    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=12, class_weight='balanced')
    clf.fit(X_train, y_train)

    model = SelectFromModel(clf, prefit=True, max_features=depth, threshold="1.5*mean")
    X_new = model.transform(X)      # reduce X to selected features
    support = model.get_support(indices=True)       # get indices of selected features
    features_names = []
    for i in support:
        features_names.append(df.columns[i])        # get names of selected features for columns

    important_features = pd.DataFrame(X_new, columns=features_names)        # df with selected features

    plt.figure(figsize=[12.8, 7.2])
    plot_tree(clf, feature_names=df.columns, class_names=["Non awe", "Awe"], label='none', impurity=False,
              filled=True, proportion=True, rounded=True, fontsize=7)
    plt.show()
    print(important_features.info())
    print("Training accuracy: ", accuracy_score(y_train, clf.predict(X_train)))
    print("Validation accuracy: ", accuracy_score(y_test, clf.predict(X_test)))
    return important_features


def decision_trees(df, y, criterion='gini'):
    n = df.shape[1]
    tr_accuracies = {}
    val_accuracies = {}
    X = df.iloc[:, :n].to_numpy()
    y = y.to_numpy()
    kf = KFold(8)

    degrees = range(2, 15)
    train = []
    valid = []
    for i in degrees:      # max depth - later min_samples_leaf
        clf = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=i, random_state=12,
                                                  class_weight='balanced')
        tr_accuracies[i] = []
        val_accuracies[i] = []
        for train_index, test_index in kf.split(X, y):
            X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred)
            val_acc = accuracy_score(y_test, y_test_pred)
            tr_accuracies[i].append(train_acc)
            val_accuracies[i].append(val_acc)

        avg_tr_acc = np.average(tr_accuracies[i])
        train.append(avg_tr_acc)
        avg_val_acc = np.average(val_accuracies[i])
        valid.append(avg_val_acc)
        print('Average training accuracy at min_samples_leaf=', i, ': ', avg_tr_acc)
        print('Average validation accuracy at min_samples_leaf=', i, ': ', avg_val_acc)

    plt.plot(degrees, train, label='Train')
    plt.plot(degrees, valid, label='Valid')
    plt.legend(loc='upper left')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Accuracy')
    plt.title('Ratings')
    plt.show()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = read_data()
    general_info = ['Timestamp', 'Comments1', 'Comments2']
    data = drop_irrelevant_columns(data, general_info)
    data = remove_lazy(data)
    awe_data = create_label(data)
    awe_data = replace(awe_data)  # adjust text answers

    # split dataset in ratings + demography vs other data
    ratings_names = ['Study_videogames', 'Work_videogames', 'Graphics_rating', 'Story_rating', 'Soundtrack_rating',
                     'Main_character_rating', 'VR']
    ratings = get_ratings(awe_data, ratings_names)
    # print(ratings.info())
    # ratings_names.remove('Title')
    awe_data = drop_irrelevant_columns(awe_data, ratings_names)

    genre = get_description_df(awe_data, 'Genre')
    graphics_description = get_description_df(awe_data, 'Graphics_descr')
    story_description = get_description_df(awe_data, 'Story_descr')
    soundtrack_description = get_description_df(awe_data, 'Soundtrack_descr')
    main_char_description = get_description_df(awe_data, 'Main_char_descr')
    locations = get_description_df(awe_data, 'Locations')
    pace_diff = get_description_df(awe_data, 'Pace_and_difficulty')
    vr_descr = get_description_df(awe_data, 'VR_descr')

    label = awe_data['Felt_awe']

    # decision_trees(ratings, y=label)      # best max depth = 3, msl = 4/11 senza max depth
    # decision_trees(genre, y=label, j=2, title='Genre')      # best max depth = 1
    # decision_trees(graphics_description, y=label, j=3, title='Graphics')   # best max depth = 2
    # decision_trees(story_description, y=label)   # best max depth = 2
    # decision_trees(soundtrack_description, y=label, j=5, title='Soundtrack')   # best max depth = 2
    # decision_trees(main_char_description, y=label, j=6, title='Main character')   # best max depth = 5, msl indifferent
    # decision_trees(locations, y=label, j=7, title='Locations')   # best max depth = 2;4
    # decision_trees(pace_diff, y=label, j=8, title='Pace and difficulty')   # best max depth = 4
    # decision_trees(vr_descr, y=label)   # best max depth = 1 and same results (78% tr and val) until 20 so idk maybe
    # i should drop it

    important_features_df = pd.DataFrame()
    important_features_df = pd.concat([important_features_df, final_decision_tree(ratings, depth=10, msl=11, y=label)],
                                      axis=1)
    important_features_df = pd.concat([important_features_df, final_decision_tree(genre, depth=1, y=label)], axis=1)
    important_features_df = pd.concat(
        [important_features_df, final_decision_tree(graphics_description, msl=2, y=label)], axis=1)
    important_features_df = pd.concat(
        [important_features_df, final_decision_tree(story_description, depth=2, y=label)], axis=1)
    important_features_df = pd.concat(
        [important_features_df, final_decision_tree(soundtrack_description, depth=2, y=label)], axis=1)
    important_features_df = pd.concat(
        [important_features_df, final_decision_tree(main_char_description, depth=4, msl=5, y=label)], axis=1)
    important_features_df = pd.concat([important_features_df, final_decision_tree(locations, depth=4, y=label)], axis=1)
    important_features_df = pd.concat([important_features_df, final_decision_tree(pace_diff, depth=4, y=label)], axis=1)
    # important_features_df = pd.concat([important_features_df, final_decision_tree(vr_descr, depth=1, y=label)], axis=1)

    print(important_features_df.info())
#    decision_trees(important_features_df, y=label)  # overfitting starts at depth 3
#    final_decision_tree(important_features_df, depth=2, y=label)    # 65% val accuracy

#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
