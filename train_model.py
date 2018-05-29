from sklearn import svm, metrics, preprocessing
import pandas as pd
import numpy as np
import datetime
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib import pyplot as plt

FILENAME = 'data/wta/matches_Simona_Halep.csv'

PICKLED_MODEL_NAME = 'SVM' + str(datetime.datetime.now()) + '.pkl'

LOAD_MODEL_FILE = 'SVM2018-05-24 10:38:38.042465.pkl'

FEATURES = ['best_of', 'draw_size', 'opponent_age', #'opponent_entry',
            'opponent_hand', 'opponent_ht', #'opponent_id',
            'opponent_ioc', 'opponent_name',
            'opponent_rank', 'opponent_rank_points', #'opponent_seed',
            'match_num', 'tourney_round', #'score',
            'score_category' 'surface', 'tourney_date', 'tourney_id', 'tourney_level', # 'tourney_name',
            'player_age', #'player_entry',
            'player_rank', 'player_rank_points', #'player_seed',
            'year']

COLUMNS_TO_DROP = ['minutes', 'opponent_entry', 'opponent_seed', 'player_entry', 'player_seed', 'opponent_id',
                   'player_hand', 'player_ht', 'player_id', 'player_ioc', 'player_name', 'tourney_id',
                   'score', 'score_category', 'previous_meeting_tourney_id', 'previous_meeting_tourney',
                   'previous_meeting_tourney_date',
                   'best_of', 'match_num', 'draw_size']

CATEGORICAL_COLUMNS = ['opponent_hand', 'opponent_name', 'opponent_ioc', 'surface', 'tourney_date', 'tourney_name',
                       'tourney_level', 'tourney_round', 'year', 'previous_meeting_score']

CONTINUOUS_COLUMNS = ['opponent_age', 'opponent_ht', 'opponent_rank', 'opponent_rank_points',
                      'player_age', 'player_rank', 'player_rank_points']


def drop_redundant_columns(player_df):
    player_df.drop(COLUMNS_TO_DROP, inplace=True, axis=1)
    return player_df


def one_hot_encode(player_df):
    print(player_df.shape)
    dummy_columns = []
    for col in CATEGORICAL_COLUMNS:
        column = pd.get_dummies(player_df[col])
        if col == 'previous_meeting_score':
            column.rename(columns={'EASY_WIN': 'EASY_WIN_PREV', 'MEDIUM_WIN': 'MEDIUM_WIN_PREV',
                                   'HARD_WIN': 'HARD_WIN_PREV', 'RET': 'RET_PREV', 'W/O': 'W/O_PREV',
                                   'EASY_LOSS': 'EASY_LOSS_PREV', 'MEDIUM_LOSS': 'MEDIUM_LOSS_PREV',
                                   'HARD_LOSS': 'HARD_LOSS_PREV'}, inplace=True)
        dummy_columns.append(column)
        print("Shape of dummy dataframe: ", col, column.shape)

    concat_columns = dummy_columns
    concat_columns.append(player_df)
    result_df = pd.concat(concat_columns, axis=1)
    print("After adding dummy columns: ", result_df.shape)
    result_df.drop(CATEGORICAL_COLUMNS, inplace=True, axis=1)
    print("After dropping categorical columns: ", result_df.shape)
    return result_df


def get_sample_weights(train_set):
    nr_training_examples = train_set.shape[0]
    sample_weights = nr_training_examples * [1]
    i = 0
    for index, row in train_set.iterrows():
        if row.win == 0:
            sample_weights[i] = 2
        i += 1
    return sample_weights


def plot_learning_curves(clf, title, X, y, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    plt.figure()
    plt.title(title)
    # if ylim is not None:
    #     plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def svm_param_selection(X, y, sw_train):
    c_list = [0.001, 0.01, 0.1, 1., 10., 100.]
    gamma_list = [0.001, 0.01, 0.1, 1., 10., 100.]
    # param_grid = {'C': c_list, 'gamma' : gamma_list}
    param_grid = {'C': c_list}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid)
    fit_params = {'sample_weight': sw_train}
    grid_search.fit(X, y, **fit_params)
    print("Grid best score:", grid_search.best_score_)
    return grid_search.best_estimator_


def main():

    # Import data
    player_matches = pd.read_csv(FILENAME, low_memory=False)

    # Drop columns we don't need
    player_matches = drop_redundant_columns(player_matches)

    # missing_values_count_matches = player_matches.isnull().sum()
    # print(missing_values_count_matches)

    # One-hot encode the categorical columns
    print(player_matches['previous_meeting_score'])
    player_matches = one_hot_encode(player_matches)

    # print(player_matches.shape)

    print("Features: ", list(player_matches.columns.values))

    # Normalize features (min-max scaling)
    # minmax_scale = preprocessing.MinMaxScaler().fit(player_matches[CONTINUOUS_COLUMNS])
    # player_matches[CONTINUOUS_COLUMNS] = minmax_scale.transform(player_matches[CONTINUOUS_COLUMNS])

    # Shuffle data
    player_matches = player_matches.sample(frac=1)

    # Split into training, validation, testing sets
    train, validate, test = np.split(
        player_matches.sample(frac=1), [int(.6 * len(player_matches)), int(.8 * len(player_matches))])

    # Double the weight for the negative examples
    # sample_weights = get_sample_weights(train)

    print(len(player_matches))
    print(train.shape)
    train_labels = train.win
    train_features = train.drop('win', axis=1)

    print("Train Features: ", list(train_features.columns.values))

    validate_labels = validate.win
    validate_features = validate.drop('win', axis=1)

    test_labels = test.win
    test_features = test.drop('win', axis=1)

    print("Number of training samples: %d" % train_features.shape[0])
    print("Number of training labels: %d" % len(train_labels))
    print("Number of validation samples: %d" % validate_features.shape[0])
    print("Number of validation labels: %d" % len(validate_labels))
    print("Number of testing samples: %d" % test_features.shape[0])
    print("Number of testing labels: %d" % len(test_labels))

    sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)

    # Training
    time_before = datetime.datetime.now()
    print("Starting training")

    # Create the SVC model
    # svc_model = svm.SVC(gamma=0.1, C=10, kernel='linear')
    # svc_model.fit(train_features, train_labels, sample_weight=sample_weights)

    # Grid search to select the best combination of parameters
    svc_model = svm_param_selection(train_features, train_labels, sample_weights)

    print("Finished training")
    print("Training took: ", (datetime.datetime.now() - time_before).total_seconds())
    print("Best performance parameters: ", svc_model.get_params())

    # Save model
    # joblib.dump(svc_model, PICKLED_MODEL_NAME)

    predicted_validate_labels = svc_model.predict(validate_features)
    f1 = metrics.f1_score(validate_labels, predicted_validate_labels, average=None)
    print("F1 score: ", f1)

    precision = metrics.precision_score(validate_labels, predicted_validate_labels, average=None)
    print("Precision score: ", precision)

    recall = metrics.recall_score(validate_labels, predicted_validate_labels, average=None)
    print("Recall score: ", recall)

    # accuracy = svc_model.score(validate_features, validate_labels)
    accuracy = metrics.accuracy_score(validate_labels, predicted_validate_labels)
    print("Accuracy: %f" % accuracy)

    print("Plotting learning curves")
    time_before = datetime.datetime.now()
    print(train_features.shape[0])
    plot_learning_curves(svc_model, "SVM learning curves", train_features, train_labels, verbose=2)
    print("Plotting learning curves took: ", (datetime.datetime.now() - time_before).total_seconds())


if __name__ == '__main__':
    main()





