"""
:author: Aniruddha Kalburgi
"""
import numpy as np
import time


class KNNImplementation:
    def __init__(self):
        pass

    def calculateDistances(self, training_sub_data, query_instance):
        """
        Calculates the Euclidean distances for each row in training_data to the query_instance and
        returns the euclidean distances array and sorted indices for its elements - sorted using np.argsort()
        :param training_data:
        :param query_instance:
        :return:
        """
        # slices and stores first 5 columns of query_instance
        query_params = query_instance[:5]

        # calculates the euclidean distance from query_instance to each row in training_sub_data - a 5-column subset of training_data
        euclidean_distances = np.sqrt(np.sum(np.square(np.subtract(training_sub_data, query_params)), axis=1))

        return euclidean_distances, np.argsort(euclidean_distances)

    def calculateManhattanDistances(self, training_sub_data, query_instance):
        """
        Calculates the Manhattan distances from each instance in training_sub_data to the query_instance
        :param training_sub_data:
        :param query_instance:
        :return:
        """
        # slices and stores first 5 columns of query_instance
        query_params = query_instance[:5]

        manhattan_distances = np.sum(np.absolute(np.subtract(training_sub_data, query_params)), axis=1)

        return manhattan_distances, np.argsort(manhattan_distances)

    def calculateMinkowskiDistances(self, training_sub_data, query_instance, a_factor):
        """
        The Generalized distance calculation - Minkowski Distances
        Takes additional a_factor to choose whether to calculate Manhattan or Euclidean distances
        :param training_sub_data:
        :param query_instance:
        :param a_factor: Value 1 returns distances equivalent to Manhattan distances where 2 returns Euclidean distances
        :return:
        """
        # slices and stores first 5 columns of query_instance
        query_params = query_instance[:5]

        # calculates the Minkowski distances from query_instance to each row in training_sub_data - a 5-column subset of training_data
        minkowski_distances = np.power(
            np.sum(np.power(np.absolute(np.subtract(training_sub_data, query_params)), a_factor), axis=1),
            (1 / a_factor))

        return minkowski_distances, np.argsort(minkowski_distances)

    def weight_based_classification(self, training_data, distances, sorted_indices, k):
        # selecting k nearest samples and then slicing down only last column - class
        k_nearest_samples_class = training_data[sorted_indices[:k]][:, -1]
        k_nearest_distances = distances[sorted_indices[:k]]
        k_nearest_weights = np.divide(1, np.square(k_nearest_distances))

        # selecting from k_nearest_samples_class the boolean array where class is particularly 0 or 1 and selecting corresponding weights from k_nearest_weights array and summing them up
        # In short, summing up weights from the same class
        class_0_weights_sum = np.sum(k_nearest_weights[k_nearest_samples_class == 0])
        class_1_weights_sum = np.sum(k_nearest_weights[k_nearest_samples_class == 1])

        # Combining > & = sign so even in tie-up cases the instance will be classified to malignant - assumed it would be okay in order to reduce false positives
        if class_1_weights_sum >= class_0_weights_sum:
            return 1

        return 0

    def distance_based_classification(self, training_data, sorted_indices, k):

        # # selecting k-nearest neighbours with this line
        # k_nearest = training_data[sorted_indices[:k]]
        #
        # # k_nearest[:, -1] - selecting last column from each row - i.e. only the classification values
        # # summing all classification values and dividing it by K to find an average
        # average_classification = np.sum(k_nearest[:, -1]) / k

        # TODO replace with vote count approach
        # count_0 = k_nearest[:, -1]k_nearest[:, -1] > 0
        k_nearest_samples_class = training_data[sorted_indices[:k]][:, -1]

        # counting number of occurrences of either 0's or 1's with below 2 lines
        class_0_count = len(k_nearest_samples_class[k_nearest_samples_class == 0])
        class_1_count = len(k_nearest_samples_class[k_nearest_samples_class == 1])

        # Combining > & = sign so even in tie-up cases the instance will be classified to malignant - assumed it would be okay in order to reduce false positives
        if class_1_count >= class_0_count:
            return 1

        return 0


def weight_based_approach(training_data, test_data, knn_k_value):
    start_time = time.time()

    # slices and stores first five columns for each row in training data
    training_data_5_columns = training_data[:, :5]

    knn = KNNImplementation()

    """-----------------------------------------------------------------------------
    # Euclidean Distance Metric
    # -----------------------------------------------------------------------------"""
    euclidean_start_time = time.time()
    eucl_weight_prediction_count = 0
    for query_instance in test_data:
        distances, euclidean_indices = knn.calculateDistances(training_data_5_columns, query_instance)

        weight_based_average = knn.weight_based_classification(training_data, distances, euclidean_indices, knn_k_value)

        if query_instance[-1] == weight_based_average:
            eucl_weight_prediction_count += 1

    print("Weight Based Eucl Accuracy : %.2f" % ((eucl_weight_prediction_count / len(test_data)) * 100) + "%")

    euclidean_finish_time = time.time() - euclidean_start_time

    """-----------------------------------------------------------------------------
    # Manhattan Distance Metric
    # -----------------------------------------------------------------------------"""
    manhattan_start_time = time.time()
    manhattan_metric_prediction_count = 0
    for query_instance in test_data:
        distances, manhattan_indices = knn.calculateManhattanDistances(training_data_5_columns, query_instance)

        classification_average = knn.weight_based_classification(training_data, distances, manhattan_indices,
                                                                 knn_k_value)

        if query_instance[-1] == classification_average:
            manhattan_metric_prediction_count += 1
    manhattan_finish_time = time.time() - manhattan_start_time

    """-----------------------------------------------------------------------------
    # Minkowski Euclidean Variant
    # -----------------------------------------------------------------------------"""
    minkowski_euclidean_start_time = time.time()
    minkowski_euclidean_metric_prediction_count = 0
    minkowski_euclidean_factor = 2
    for query_instance in test_data:
        distances, mink_eucl_indices = knn.calculateMinkowskiDistances(training_data_5_columns, query_instance,
                                                                       minkowski_euclidean_factor)

        classification_average = knn.weight_based_classification(training_data, distances, mink_eucl_indices,
                                                                 knn_k_value)

        if query_instance[-1] == classification_average:
            minkowski_euclidean_metric_prediction_count += 1
    minkowski_euclidean_finish_time = time.time() - minkowski_euclidean_start_time

    """-----------------------------------------------------------------------------
    #Minkowski Manhattan Variant
    -----------------------------------------------------------------------------"""
    minkowski_manhattan_start_time = time.time()
    minkowski_manhattan_metric_prediction_count = 0
    minkowski_manhattan_factor = 1
    for query_instance in test_data:
        distances, mink_man_indices = knn.calculateMinkowskiDistances(training_data_5_columns, query_instance,
                                                                      minkowski_manhattan_factor)

        classification_average = knn.weight_based_classification(training_data, distances, mink_man_indices,
                                                                 knn_k_value)

        if query_instance[-1] == classification_average:
            minkowski_manhattan_metric_prediction_count += 1
    minkowski_manhattan_finish_time = time.time() - minkowski_manhattan_start_time

    print(
        "\n---------------------------------------\nUsing %d nearest neighbour(s)" % knn_k_value + "\n---------------------------------------")

    print(
        "\n---------------------------------------\nEuclidean Distance Evaluations:\n---------------------------------------")
    print("Euclidean Correct Predictions: %d" % eucl_weight_prediction_count)
    print("\nEuclidean Metric Accuracy: %.2f" % ((eucl_weight_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nManhattan Distance Evaluations:\n---------------------------------------")
    print("Manhattan Correct Predictions: %d" % manhattan_metric_prediction_count)
    print("\nManhattan Metric Accuracy: %.2f" % ((manhattan_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Euclidean Distance Evaluations:\n---------------------------------------")
    print("Minkowski Euclidean Correct Predictions: %d" % minkowski_euclidean_metric_prediction_count)
    print("\nMink-Euclidean Metric Accuracy: %.2f" % (
            (minkowski_euclidean_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Manhattan Distance Evaluations:\n---------------------------------------")
    print("Minkowski Manhattan Correct Predictions: %d" % minkowski_manhattan_metric_prediction_count)
    print("\nMink-Manhattan Metric Accuracy: %.2f" % (
            (minkowski_manhattan_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nFinish Time Evaluations:\n---------------------------------------")
    print("Euclidean Metric Finish Time\t\t\t%f seconds" % euclidean_finish_time)
    print("Manhattan Metric Finish Time\t\t\t%f seconds" % manhattan_finish_time)
    print("Minkowski-Euclidean Metric Finish Time\t%f seconds" % minkowski_euclidean_finish_time)
    print("Minkowski-Manhattan Metric Finish Time\t%f seconds" % minkowski_manhattan_finish_time)
    print("---------------------------------------")

    print("\nTotal Time : %.4f seconds" % (time.time() - start_time))

    return (eucl_weight_prediction_count / len(test_data) * 100), (
            manhattan_metric_prediction_count / len(test_data) * 100), (
                   minkowski_euclidean_metric_prediction_count / len(test_data) * 100), (
                   minkowski_manhattan_metric_prediction_count / len(test_data) * 100)


def distanace_based_approach(training_data, test_data, knn_k_value):
    start_time = time.time()

    # slices and stores first five columns for each row in training data
    training_data_5_columns = training_data[:, :5]

    knn = KNNImplementation()

    """-----------------------------------------------------------------------------
    # Euclidean Distance Metric
    # -----------------------------------------------------------------------------"""
    euclidean_start_time = time.time()
    euclidean_metric_prediction_count = 0

    for query_instance in test_data:
        distances, euclidean_indices = knn.calculateDistances(training_data_5_columns, query_instance)

        # calculates average value of classification column for K nearest neighbours and returns the same
        classification_average = knn.distance_based_classification(training_data, euclidean_indices, knn_k_value)

        if query_instance[-1] == classification_average:
            euclidean_metric_prediction_count += 1

    euclidean_finish_time = time.time() - euclidean_start_time

    """-----------------------------------------------------------------------------
    # Manhattan Distance Metric
    # -----------------------------------------------------------------------------"""
    manhattan_start_time = time.time()
    manhattan_metric_prediction_count = 0
    for query_instance in test_data:
        distances, manhattan_indices = knn.calculateManhattanDistances(training_data_5_columns, query_instance)

        classification_average = knn.distance_based_classification(training_data, manhattan_indices, knn_k_value)

        if query_instance[-1] == classification_average:
            manhattan_metric_prediction_count += 1
    manhattan_finish_time = time.time() - manhattan_start_time

    """-----------------------------------------------------------------------------
    # Minkowski Euclidean Variant
    # -----------------------------------------------------------------------------"""
    minkowski_euclidean_start_time = time.time()
    minkowski_euclidean_metric_prediction_count = 0
    minkowski_euclidean_factor = 2
    for query_instance in test_data:
        distances, mink_eucl_indices = knn.calculateMinkowskiDistances(training_data_5_columns, query_instance,
                                                                       minkowski_euclidean_factor)

        classification_average = knn.distance_based_classification(training_data, mink_eucl_indices, knn_k_value)

        if query_instance[-1] == classification_average:
            minkowski_euclidean_metric_prediction_count += 1
    minkowski_euclidean_finish_time = time.time() - minkowski_euclidean_start_time

    """-----------------------------------------------------------------------------
    #Minkowski Manhattan Variant
    -----------------------------------------------------------------------------"""
    minkowski_manhattan_start_time = time.time()
    minkowski_manhattan_metric_prediction_count = 0
    minkowski_manhattan_factor = 1
    for query_instance in test_data:
        distances, mink_man_indices = knn.calculateMinkowskiDistances(training_data_5_columns, query_instance,
                                                                      minkowski_manhattan_factor)

        classification_average = knn.distance_based_classification(training_data, mink_man_indices, knn_k_value)

        if query_instance[-1] == classification_average:
            minkowski_manhattan_metric_prediction_count += 1
    minkowski_manhattan_finish_time = time.time() - minkowski_manhattan_start_time

    print(
        "\n---------------------------------------\nUsing %d nearest neighbour(s)" % knn_k_value + "\n---------------------------------------")

    print(
        "\n---------------------------------------\nEuclidean Distance Evaluations:\n---------------------------------------")
    print("Euclidean Correct Predictions: %d" % euclidean_metric_prediction_count)
    print("\nEuclidean Metric Accuracy: %.2f" % ((euclidean_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nManhattan Distance Evaluations:\n---------------------------------------")
    print("Manhattan Correct Predictions: %d" % manhattan_metric_prediction_count)
    print("\nManhattan Metric Accuracy: %.2f" % ((manhattan_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Euclidean Distance Evaluations:\n---------------------------------------")
    print("Minkowski Euclidean Correct Predictions: %d" % minkowski_euclidean_metric_prediction_count)
    print("\nMink-Euclidean Metric Accuracy: %.2f" % (
            (minkowski_euclidean_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nMinkowski Manhattan Distance Evaluations:\n---------------------------------------")
    print("Minkowski Manhattan Correct Predictions: %d" % minkowski_manhattan_metric_prediction_count)
    print("\nMink-Manhattan Metric Accuracy: %.2f" % (
            (minkowski_manhattan_metric_prediction_count / len(test_data)) * 100) + "%")
    print("---------------------------------------")

    print(
        "\n---------------------------------------\nFinish Time Evaluations:\n---------------------------------------")
    print("Euclidean Metric Finish Time\t\t\t%f seconds" % euclidean_finish_time)
    print("Manhattan Metric Finish Time\t\t\t%f seconds" % manhattan_finish_time)
    print("Minkowski-Euclidean Metric Finish Time\t%f seconds" % minkowski_euclidean_finish_time)
    print("Minkowski-Manhattan Metric Finish Time\t%f seconds" % minkowski_manhattan_finish_time)
    print("---------------------------------------")

    print("\nTotal Time : %.4f seconds" % (time.time() - start_time))

    return (euclidean_metric_prediction_count / len(test_data) * 100), (
                manhattan_metric_prediction_count / len(test_data) * 100), (
                       minkowski_euclidean_metric_prediction_count / len(test_data) * 100), (
                       minkowski_manhattan_metric_prediction_count / len(test_data) * 100)


def main():
    euclidean_accuracies = []
    manhattan_accuracies = []
    minkowski_euclidean_accuracies = []
    minkowski_manhattan_accuracies = []

    k_samples = []
    k_samples.extend(list(range(1, 129, 1)))

    print("k_samples = " + str(k_samples))

    for k in k_samples:
        training_file = "../dataFiles/cancer_updated/trainingData2.csv"
        test_file = "../dataFiles/cancer_updated/testData2.csv"
        knn_k_value = k
        # use_weight_based_approach = False
        use_weight_based_approach = True

        training_data = np.genfromtxt(training_file, delimiter=",")
        test_data = np.genfromtxt(test_file, delimiter=",")

        if use_weight_based_approach:
            print("______________Using Weight Based Approach______________")
            accuracies = weight_based_approach(training_data, test_data, knn_k_value)

            euclidean_accuracies.append(accuracies[0])
            manhattan_accuracies.append(accuracies[1])
            minkowski_euclidean_accuracies.append(accuracies[2])
            minkowski_manhattan_accuracies.append(accuracies[3])
        else:
            print("______________Using Distance Based Approach______________")
            accuracies = distanace_based_approach(training_data, test_data, knn_k_value)

            euclidean_accuracies.append(accuracies[0])
            manhattan_accuracies.append(accuracies[1])
            minkowski_euclidean_accuracies.append(accuracies[2])
            minkowski_manhattan_accuracies.append(accuracies[3])

    print("euclidean_accuracies = " + str(euclidean_accuracies))
    print("manhattan_accuracies = " + str(manhattan_accuracies))
    print("mink_eucl_accuracies = " + str(minkowski_euclidean_accuracies))
    print("mink_manhatt_accuracies = " + str(minkowski_manhattan_accuracies))


if __name__ == '__main__':
    main()
