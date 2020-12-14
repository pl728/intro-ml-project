# AUTHOR: CHLOE NGUYEN

from sklearn.impute import KNNImputer
from project.utils import *

import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))

    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    sparse_matrix = matrix.T
    mat = nbrs.fit_transform(sparse_matrix)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)



    # Using knn_impute_by_user
    # print("\nUsing knn_impute_by_user: \n")
    # val_acc = []
    # k_star = 1
    # for k in range(1, 27, 5):
    #     acc = knn_impute_by_user(sparse_matrix, val_data, k)
    #     val_acc.append(acc)
    #     # Keep track of k with highest validation accuracy
    #     if acc > val_acc[(k_star-1)//5]:
    #         k_star = k

    # Plot for validation accuracy
    # k_range = range(1, 27, 5)
    # plt.plot(k_range, val_acc)
    # plt.title("Value of K vs. Validation Accuracy")
    # plt.xlabel("Value of K")
    # plt.ylabel("Validation Accuracy")
    # plt.show()
    #
    # # Test accuracy of k star
    # print("\nk* = ", k_star)
    # nbrs = KNNImputer(n_neighbors=k_star)
    # # We use NaN-Euclidean distance measure.
    # mat = nbrs.fit_transform(sparse_matrix)
    # acc = sparse_matrix_evaluate(test_data, mat)
    # print("Test Accuracy: {}".format(acc))



    # Using knn_impute_by_item
    print("\nUsing knn_impute_by_item: \n")
    val_acc = []
    k_star = 1
    for k in range(1, 27, 5):
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc.append(acc)
        # Keep track of k with highest validation accuracy
        if acc > val_acc[(k_star-1)//5]:
            k_star = k

    # Plot for validation accuracy
    k_range = range(1, 27, 5)
    plt.plot(k_range, val_acc)
    plt.title("Value of K vs. Validation Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Validation Accuracy")
    plt.show()

    # Test accuracy of k star
    print("\nk* = ", k_star)
    nbrs = KNNImputer(n_neighbors=k_star)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(sparse_matrix.T)
    acc = sparse_matrix_evaluate(test_data, mat.T)
    print("Test Accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
