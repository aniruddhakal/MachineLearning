"""
:author: Aniruddha Kalburgi
"""

import numpy as np
import time
import matplotlib.pyplot as plt


class SolutionToRegression:
    def __init__(self):
        pass

    def calculateEuclideanDistances(self, train_data, query_instance, K):
        query_param = query_instance[:12]

        euclidean_distances = np.sqrt(np.sum(np.square(np.subtract(train_data, query_param)), axis=1))

        return euclidean_distances, np.argsort(euclidean_distances)

    def calculateManhattanDistances(self, train_data, query_instance, K):
        query_param = query_instance[:12]

        manhattan_distances = np.sum(np.absolute(np.subtract(train_data, query_param)), axis=1)

        return manhattan_distances, np.argsort(manhattan_distances)

    def calculateMinkowskiDistances(self, train_data, query_instance, K, a_factor):
        query_param = query_instance[:12]

        minkowski_distances = np.power(
            np.sum(np.power(np.absolute(np.subtract(train_data, query_param)), a_factor), axis=1), (1 / a_factor))

        return minkowski_distances, np.argsort(minkowski_distances)

    def simple_classification_using_regression(self, all_columns_training_data, sorted_indices, K):
        k_nearest_last_column = all_columns_training_data[sorted_indices[:K]][:, -1]
        regression_predicted_value = np.mean(k_nearest_last_column)

        return regression_predicted_value

    def weighted_classification_using_regression(self, train_data, distances, sorted_indices, K):
        k_nearest_samples = train_data[sorted_indices[:K]][:, -1]
        k_nearest_distances = distances[sorted_indices[:K]]
        k_nearest_weights = np.divide(1, np.square(k_nearest_distances))

        regression_predicted_value = np.divide(np.sum(np.multiply(k_nearest_samples, k_nearest_weights)),
                                               np.sum(k_nearest_weights))

        return regression_predicted_value

    def r_squared_metric(self, regression_values, test_data):
        actual_values = test_data[:, -1]

        sum_of_squared_residuals = np.sum(np.square(np.subtract(regression_values, actual_values)))
        y_bar_mean_of_actual = np.mean(actual_values)
        total_sum_of_squares = np.sum(np.square(np.subtract(y_bar_mean_of_actual, actual_values)))

        division_result = sum_of_squared_residuals / total_sum_of_squares

        r_square = np.subtract(1, division_result)

        return r_square


def normalize_values(dataset):
    initial_12_columns = dataset[:, : 12]

    new_values = (initial_12_columns - np.min(initial_12_columns)) / (
            np.max(initial_12_columns) - np.min(initial_12_columns))

    return new_values


def weight_based_regression(training_data, training_data_12_columns, test_data, KNN_K_Value):
    start_time = time.time()

    regression = SolutionToRegression()

    """-----------------------------------------------------------------------------
       # Euclidean Distance Metric
       # -----------------------------------------------------------------------------"""
    regression_values = []
    eucl_start_time = time.time()
    for query_instance in test_data:
        distances, sorted_indices = regression.calculateEuclideanDistances(training_data_12_columns, query_instance,
                                                                           KNN_K_Value)
        regression_value = regression.weighted_classification_using_regression(training_data, distances,
                                                                               sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    eucl_r_square = regression.r_squared_metric(regression_values, test_data)
    eucl_finish_time = time.time() - eucl_start_time

    """-----------------------------------------------------------------------------
        # Manhattan Distance Metric
        # -----------------------------------------------------------------------------"""
    regression_values = []
    manhattan_start_time = time.time()
    for query_instance in test_data:
        mink_eucl_distances, sorted_indices = regression.calculateManhattanDistances(training_data_12_columns,
                                                                                     query_instance,
                                                                                     KNN_K_Value)
        regression_value = regression.weighted_classification_using_regression(training_data, mink_eucl_distances,
                                                                               sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    manhat_r_square = regression.r_squared_metric(regression_values, test_data)
    manhattan_finish_time = time.time() - manhattan_start_time

    """-----------------------------------------------------------------------------
        # Minkowski Euclidean Variant
        # -----------------------------------------------------------------------------"""
    regression_values = []
    mink_eucl_start_time = time.time()
    for query_instance in test_data:
        mink_eucl_distances, sorted_indices = regression.calculateMinkowskiDistances(training_data_12_columns,
                                                                                     query_instance, KNN_K_Value, 2)

        regression_value = regression.weighted_classification_using_regression(training_data, mink_eucl_distances,
                                                                               sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    mink_eucl_r_square = regression.r_squared_metric(regression_values, test_data)
    mink_eucl_finish_time = time.time() - mink_eucl_start_time

    """-----------------------------------------------------------------------------
        #Minkowski Manhattan Variant
        -----------------------------------------------------------------------------"""
    regression_values = []
    mink_manhat_start_time = time.time()
    for query_instance in test_data:
        mink_manhat_distances, sorted_indices = regression.calculateMinkowskiDistances(training_data_12_columns,
                                                                                       query_instance, KNN_K_Value, 1)

        regression_value = regression.weighted_classification_using_regression(training_data, mink_manhat_distances,
                                                                               sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    mink_manhat_r_square = regression.r_squared_metric(regression_values, test_data)
    mink_manhat_finish_time = time.time() - mink_manhat_start_time

    print(
        "\n---------------------------------------\nUsing %d nearest neighbour(s)" % KNN_K_Value + "\n---------------------------------------")

    print(
        "\n---------------------------------------\nEuclidean Distance Evaluations:\n---------------------------------------")
    print("\nEuclidean Metric Accuracy: %.2f" % (eucl_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nManhattan Distance Evaluations:\n---------------------------------------")
    print("\nManhattan Metric Accuracy: %.2f" % (manhat_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Euclidean Distance Evaluations:\n---------------------------------------")
    print("\nMink-Euclidean Metric Accuracy: %.2f" % (mink_eucl_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Manhattan Distance Evaluations:\n---------------------------------------")
    print("\nMink-Manhattan Metric Accuracy: %.2f" % (mink_manhat_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nFinish Time Evaluations:\n---------------------------------------")
    print("Euclidean Metric Finish Time\t\t\t%f seconds" % eucl_finish_time)
    print("Manhattan Metric Finish Time\t\t\t%f seconds" % manhattan_finish_time)
    print("Minkowski-Euclidean Metric Finish Time\t%f seconds" % mink_eucl_finish_time)
    print("Minkowski-Manhattan Metric Finish Time\t%f seconds" % mink_manhat_finish_time)
    print("---------------------------------------")

    print("\nWeighted Regression Total Time : %.4f seconds" % (time.time() - start_time))

    return eucl_r_square * 100, manhat_r_square * 100, mink_eucl_r_square * 100, mink_manhat_r_square * 100


def simple_regression(training_data, training_data_12_columns, test_data, KNN_K_Value):
    start_time = time.time()

    regression = SolutionToRegression()

    """-----------------------------------------------------------------------------
       # Euclidean Distance Metric
       # -----------------------------------------------------------------------------"""
    regression_values = []
    eucl_start_time = time.time()
    for query_instance in test_data:
        distances, sorted_indices = regression.calculateEuclideanDistances(training_data_12_columns, query_instance,
                                                                           KNN_K_Value)
        regression_value = regression.simple_classification_using_regression(training_data, sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    eucl_r_square = regression.r_squared_metric(regression_values, test_data)
    eucl_finish_time = time.time() - eucl_start_time

    """-----------------------------------------------------------------------------
        # Manhattan Distance Metric
        # -----------------------------------------------------------------------------"""
    regression_values = []
    manhattan_start_time = time.time()
    for query_instance in test_data:
        mink_eucl_distances, sorted_indices = regression.calculateManhattanDistances(training_data_12_columns,
                                                                                     query_instance,
                                                                                     KNN_K_Value)
        regression_value = regression.simple_classification_using_regression(training_data, sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    manhat_r_square = regression.r_squared_metric(regression_values, test_data)
    manhattan_finish_time = time.time() - manhattan_start_time

    """-----------------------------------------------------------------------------
        # Minkowski Euclidean Variant
        # -----------------------------------------------------------------------------"""
    regression_values = []
    mink_eucl_start_time = time.time()
    for query_instance in test_data:
        mink_eucl_distances, sorted_indices = regression.calculateMinkowskiDistances(training_data_12_columns,
                                                                                     query_instance, KNN_K_Value, 2)

        regression_value = regression.simple_classification_using_regression(training_data, sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    mink_eucl_r_square = regression.r_squared_metric(regression_values, test_data)
    mink_eucl_finish_time = time.time() - mink_eucl_start_time

    """-----------------------------------------------------------------------------
        #Minkowski Manhattan Variant
        -----------------------------------------------------------------------------"""
    regression_values = []
    mink_manhat_start_time = time.time()
    for query_instance in test_data:
        mink_manhat_distances, sorted_indices = regression.calculateMinkowskiDistances(training_data_12_columns,
                                                                                       query_instance, KNN_K_Value, 1)

        regression_value = regression.simple_classification_using_regression(training_data, sorted_indices, KNN_K_Value)

        regression_values.append(regression_value)

    mink_manhat_r_square = regression.r_squared_metric(regression_values, test_data)
    mink_manhat_finish_time = time.time() - mink_manhat_start_time

    print(
        "\n---------------------------------------\nUsing %d nearest neighbour(s)" % KNN_K_Value + "\n---------------------------------------")

    print(
        "\n---------------------------------------\nEuclidean Distance Evaluations:\n---------------------------------------")
    print("\nEuclidean Metric Accuracy: %.2f" % (eucl_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nManhattan Distance Evaluations:\n---------------------------------------")
    print("\nManhattan Metric Accuracy: %.2f" % (manhat_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Euclidean Distance Evaluations:\n---------------------------------------")
    print("\nMink-Euclidean Metric Accuracy: %.2f" % (mink_eucl_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Manhattan Distance Evaluations:\n---------------------------------------")
    print("\nMink-Manhattan Metric Accuracy: %.2f" % (mink_manhat_r_square * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nFinish Time Evaluations:\n---------------------------------------")
    print("Euclidean Metric Finish Time\t\t\t%f seconds" % eucl_finish_time)
    print("Manhattan Metric Finish Time\t\t\t%f seconds" % manhattan_finish_time)
    print("Minkowski-Euclidean Metric Finish Time\t%f seconds" % mink_eucl_finish_time)
    print("Minkowski-Manhattan Metric Finish Time\t%f seconds" % mink_manhat_finish_time)
    print("---------------------------------------")

    print("\nSimple Regression Total Time : %.4f seconds" % (time.time() - start_time))

    return eucl_r_square * 100, manhat_r_square * 100, mink_eucl_r_square * 100, mink_manhat_r_square * 100


def main():
    euclidean_accuracies = []
    manhattan_accuracies = []
    minkowski_euclidean_accuracies = []
    minkowski_manhattan_accuracies = []
    k_values = []

    k_samples = []
    k_samples.extend(list(range(1,10, 1)))
    k_samples.extend(list(range(10, 1610, 10)))

    print(k_samples)

    for k in k_samples:
        k_values.append(k)
        KNN_K_Value = k

        USE_WEIGHT_BASED_APPROACH = True
        # USE_WEIGHT_BASED_APPROACH = False

        # USE_NORMALISATION = True
        USE_NORMALISATION = False

        train_data_file = "../dataFiles/regressionData/trainingData.csv"
        test_data_file = "../dataFiles/regressionData/testData.csv"

        training_data = np.genfromtxt(train_data_file, delimiter=",")
        test_data = np.genfromtxt(test_data_file, delimiter=",")

        if USE_NORMALISATION:
            print("______________ USING NORMALIZATION ______________")
            training_data = normalize_values(training_data)
            test_data = normalize_values(test_data)

        train_data_12_columns = training_data[:, :12]

        if USE_WEIGHT_BASED_APPROACH:
            accuracies = weight_based_regression(training_data, train_data_12_columns, test_data, KNN_K_Value)

            euclidean_accuracies.append(accuracies[0])
            manhattan_accuracies.append(accuracies[1])
            minkowski_euclidean_accuracies.append(accuracies[2])
            minkowski_manhattan_accuracies.append(accuracies[3])
        else:
            simple_regression(training_data, train_data_12_columns, test_data, KNN_K_Value)

    print("Euclidean Accuracies:\n" + str(euclidean_accuracies))
    print("K Values:\n" + str(k_values))
    plt.plot(k_values, euclidean_accuracies)
    plt.show()


if __name__ == "__main__":
    main()
