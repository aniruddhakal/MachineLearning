import pandas as pd
import numpy as np

from time import time

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


def get_numerical_catogorical_features_list(df):
    """
    Returns a column names lists of numerical features and categorical features
    :param df:
    :return: numerical_features, categorical_features
    """
    # select only numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return numerical_features, categorical_features


def drop_unnecessary_values(df):
    df.drop('movie_imdb_link', axis=1, inplace=True)
    df.drop('genres', axis=1, inplace=True)
    df.drop('plot_keywords', axis=1, inplace=True)

    return df


def preprocess_data(df, numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    df = pd.DataFrame(preprocessor.fit_transform(df))

    return df, preprocessor


def read_data(file_name):
    df = pd.read_csv(file_name)

    return df


def shuffle_data_sequence(df):
    # shuffle/randomize row sequence
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def split_feature_target(df):
    features = df.loc[:, df.columns != 'quartile_class']
    target = df['quartile_class']

    return features, target


# x_train, y_train, x_test, y_test
def cross_validation(x_train, y_train, x_test, y_test, full_data, full_target, max_depth, column_names):
    print("\n\n|----------- Running Cross Validations using RandomForestClassifier -----------|")

    # cv_classifier = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=max_depth, n_jobs=-1, random_state=450)
    # test_classifier = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=max_depth, n_jobs=-1, random_state=450)
    cv_classifier = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=max_depth, n_jobs=-1, random_state=450)
    test_classifier = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=max_depth, n_jobs=-1,
                                                      random_state=450)

    cv_classifier.fit(full_data, full_target)
    test_classifier.fit(x_train, y_train)

    # # print each feature's importance %
    # column_names = x_train.columns.values
    # for idx, feature in enumerate(column_names):
    #     print("%s: " % feature + "%f" % (features_importance[idx]))

    # estimator = neighbors.KNeighborsRegressor(n_neighbors=2280, algorithm="kd_tree", weights='distance')
    # estimator = neighbors.KNeighborsRegressor()
    # estimator = neighbors.KNeighborsClassifier(n_neighbors=3380, algorithm="auto", weights='distance')

    print("\nBefore Pruning - Test Score: %f" % (test_classifier.score(x_test, y_test) * 100) + "%")
    features_importance = cv_classifier.feature_importances_
    ser = pd.Series(features_importance)
    ser.index = column_names
    # ser.plot(kind='barh')
    # plt.title("Feature Importance")
    # plt.show()

    # cv_score = model_selection.cross_val_score(cv_classifier, full_data, full_target, cv=10, scoring='accuracy',
    #                                            n_jobs=-1)
    #
    # print("\nBefore Pruning - CV Score Using ExtraTreeClf" + str(cv_score))
    # print("Mean, Std Deviation")
    # print(cv_score.mean(), cv_score.std())

    # prune features with importance less than features_importance_threshold
    features_importance_threshold = 0.04

    while len(features_importance[features_importance < features_importance_threshold]) > 0:
        bool_array = features_importance >= features_importance_threshold
        x_train = x_train.loc[:, bool_array]
        x_test = x_test.loc[:, bool_array]
        full_data = full_data.loc[:, bool_array]

        cv_classifier.fit(full_data, full_target)
        test_classifier.fit(x_train, y_train)

        features_importance = cv_classifier.feature_importances_

        print("\n\n#--------- After Pruning Less Important Features ---------#")

        predicted = test_classifier.predict(x_test)

        print("\nClassification Report - Test:\n" + str(classification_report(y_test, predicted)))
        print("\nConfusion Matrix - Test:\n" + str(confusion_matrix(y_test, predicted)))
        print("\nAfter Pruning - Test Score: %f" % (test_classifier.score(x_test, y_test) * 100) + "%")

        cv_score = model_selection.cross_val_score(cv_classifier, full_data, full_target, cv=10,
                                                   scoring='accuracy', n_jobs=-1)

        print("\nCV Score Using RandomForestClf: " + str(cv_score * 100))
        print("Mean: %f" % (cv_score.mean() * 100) + "Std Deviation: %f" % (cv_score.std() * 100))


def param_cv(feature_train, target_train, feature_test, target_test, full_train, full_target):
    features_size = len(full_train.columns.values)
    # param_grid = [{'n_neighbors': list(range(1, 2000, 50))}, {'max_depth': list(range(1, features_size + 1))}]

    # param_grid = [{'n_neighbors': list([1, 10, 100, 200, 500])}]
    # classifier = model_selection.GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=param_grid, cv=10,
    #                                           refit=False)

    columns_len = len(full_train.columns)
    param_grid = [{'n_estimators': list([100, 300, 500, 750, 1000, 1250, 1500, 1750, 2000])}]
    # {'max_depth': list(range(1, columns_len + 1))}]
    # param_grid = [{'max_depth': list(range(1, columns_len + 1))}]

    # grid_search = model_selection.GridSearchCV(estimator=ensemble.ExtraTreesClassifier(n_estimators=1000, n_jobs=-1),
    #                                            param_grid=param_grid, cv=10,
    #                                            refit=True, n_jobs=-1)
    grid_search = model_selection.GridSearchCV(
        ensemble.RandomForestClassifier(n_jobs=-1),
        param_grid=param_grid, cv=10, refit=True, n_jobs=-1)

    # classifier.fit(full_train, full_target)
    grid_search.fit(feature_train, target_train)

    # print(classifier.best_estimator_)
    print("Best Score: " + str(grid_search.best_score_))
    print("Best Params: " + str(grid_search.best_params_))
    print("CV Result: " + str(grid_search.cv_results_))

    # refit best params or set refit=True for GridSearchCV
    # grid_search.fit(feature_train, target_train)
    print("CV Score:" + str(grid_search.score(feature_test, target_test)))


def imbalance_handling(x_train, y_train, x_test, y_test, data, target):
    # Imbalance Handling - re-balance using SMOTE
    smote = SMOTE(random_state=450, n_jobs=-1)
    data, target = smote.fit_sample(data, target)
    x_train, y_train = smote.fit_sample(x_train, y_train)
    x_test, y_test = smote.fit_sample(x_test, y_test)

    data, x_train, x_test = pd.DataFrame(data), pd.DataFrame(x_train), pd.DataFrame(x_test)
    target, y_train, y_test = pd.Series(target), pd.Series(y_train), pd.Series(y_test)

    target, y_train, y_test = target.ravel(), y_train.ravel(), y_test.ravel()

    return x_train, y_train, x_test, y_test, data, target


def plot_learning_curve(train_sizes, train_scores, test_scores, title):
    plt.clf()
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
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


def main():
    file_name = "./data/movie_metadata.csv"
    df = pd.read_csv(file_name)

    df = drop_unnecessary_values(df)

    scores = df['imdb_score']

    mean = scores.mean()
    # classifying the target between above mean(2) and below mean(1) categories
    df['quartile_class'] = scores.apply(lambda x: 2 if x >= mean else 1)

    # dropping imdb score from features otherwise it may have biggest biased impact on the outcome
    df.drop('imdb_score', axis=1, inplace=True)

    # converting data of dtype 'category' to 'int' type
    category_columns = df.select_dtypes(['category']).columns
    df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)

    # features, target = split_feature_target(df)
    target = df['quartile_class']
    df.drop('quartile_class', axis=1, inplace=True)

    all_column_names = df.columns.values
    numerical_features, categorical_features = get_numerical_catogorical_features_list(df)

    df, preprocessor = preprocess_data(df, numerical_features=numerical_features,
                                       categorical_features=categorical_features)

    print("\nPreprocessing Finished")

    # a df to store scores for different models
    scores_df = pd.DataFrame(None)
    series = pd.Series(None)
    # these scores are approx
    # newton-cg - 73%
    # lbfgs - 70%
    # liblinear - 70%
    # sag - 65%
    # saga - 64%
    classifier = LogisticRegression(solver='newton-cg', max_iter=3500, n_jobs=-1)

    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=450)

    x_train, y_train, x_test, y_test, df, target = imbalance_handling(x_train, y_train, x_test, y_test, df, target)

    classifier.fit(x_train, y_train)
    score = (classifier.score(x_test, y_test) * 100)
    series['Logistic Regressor'] = score
    scores_df = scores_df.append(series, ignore_index=True)
    print("Logistic Regression Score: %f" % score + "%")

    start_time = time()
    print("\n\n#----- Ensemble Techniques -----#")

    test_classifier = ensemble.RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=19)
    test_classifier.fit(x_train, y_train)
    score = (test_classifier.score(x_test, y_test) * 100)
    series['RandomForestRegressor'] = score
    scores_df = scores_df.append(series, ignore_index=True)
    print("\nRandomForestRegressor Test Score: %f" % score + "%")
    # print("Running Learning Curve...")
    # train_sizes, train_scores, test_scores = learning_curve(test_classifier, df, target, random_state=450, n_jobs=-1,
    #                                                         cv=10)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "Learning Curve - Random Forest Regressor")

    test_classifier = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=19, n_jobs=-1)
    test_classifier.fit(x_train, y_train)
    score = (test_classifier.score(x_test, y_test) * 100)
    # series['Model'] = 'RandomForestClassifier'
    series['RandomForestClassifier'] = score
    # series['Score'] = score
    scores_df = scores_df.append(series, ignore_index=True)
    print("\nRandomForestClassifier Test Score: %f" % score + "%")
    # print("Running Learning Curve...")
    # train_sizes, train_scores, test_scores = learning_curve(test_classifier, df, target, random_state=450, n_jobs=-1,
    #                                                         cv=10)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "Learning Curve - Random Forest Classifier")

    test_classifier = ensemble.BaggingClassifier(n_estimators=1000, n_jobs=-1)
    test_classifier.fit(x_train, y_train)
    score = (test_classifier.score(x_test, y_test) * 100)
    # series['Model'] = 'BaggingClassifier'
    series['BaggingClassifier'] = score
    # series['Score'] = score
    scores_df = scores_df.append(series, ignore_index=True)
    print("\nBaggingClassifier Test Score: %f" % score + "%")
    # print("Running Learning Curve...")
    # train_sizes, train_scores, test_scores = learning_curve(test_classifier, df, target, random_state=450, n_jobs=-1,
    #                                                         cv=10)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "Learning Curve - Bagging Classifier")

    test_classifier = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=19, n_jobs=-1, random_state=450)
    test_classifier.fit(x_train, y_train)
    score = (test_classifier.score(x_test, y_test) * 100)
    # series['Model'] = 'ExtraTreeClassifier'
    series['ExtraTreeClassifier'] = score
    # series['Score'] = score
    scores_df = scores_df.append(series, ignore_index=True)
    print("\nExtraTreeClassifier Test Score: %f" % score + "%")

    plt.title("Accuracy Scores For Models")
    # scores_df.plot(kind='barh')
    series.plot(kind='barh')
    # plt.show()

    # print("Running Learning Curve...")
    # train_sizes, train_scores, test_scores = learning_curve(test_classifier, df, target, random_state=450, n_jobs=-1,
    #                                                         cv=10)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "Learning Curve - Extra Tree Classifier")

    # Pruning less important features and running cross validations using Random Forest
    cross_validation(x_train, y_train, x_test, y_test, full_data=df, full_target=target, max_depth=19,
                     column_names=all_column_names)

    # hyperparameter optimization
    # print("#---------------- ParamCV output ----------------#")
    param_cv(x_train, y_train, x_test, y_test, df, target)

    finish_time = time() - start_time

    print("Total Time %f seconds" % finish_time)


if __name__ == '__main__':
    main()
