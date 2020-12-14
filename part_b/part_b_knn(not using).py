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


def knn_impute_by_subject(matrix, subjects, valid_data, k, num_students, num_questions):
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

    # For each student
    for s in range(matrix.shape[0]):
        b = (matrix[s].T).reshape(-1, 1)
        sparse_matrix = np.concatenate((b, subjects), axis=1)
        mat = nbrs.fit_transform(sparse_matrix)
        missing = np.isnan(matrix[s])
        matrix[s][missing] = (mat.T)[0][missing]

    acc = sparse_matrix_evaluate(valid_data, matrix)
    print("Validation Accuracy: {}".format(acc))
    return acc


def load_questions_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }

    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                subjects = (row[1][1:-1]).split(',')
                all_subjects = []
                for a in subjects:
                    all_subjects.append(int(a))
                data["subject_id"].append(all_subjects)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def main():
    a = np.arange(8)
    print(type(a))
    print(a.shape)
    a = np.reshape(a, (2, 4))
    print(a)
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    num_questions = sparse_matrix.shape[1]

    subjects_d = load_questions_meta_csv("../data/question_meta.csv")
    subjects_m = -np.ones((num_questions, 388))
    for q in range(num_questions):
        for i in subjects_d["subject_id"][q]:
            subjects_m[q][i] += 2

    # Using knn_impute_by_item
    print("\nUsing knn_impute_by_subject: \n")
    val_acc = []
    k_star = 1

    for k in range(1, 27, 5):
        print("\nK val: ", k)
        acc = knn_impute_by_subject(sparse_matrix, subjects_m, val_data, k, sparse_matrix.shape[0],
                                    sparse_matrix[1])
        val_acc.append(acc)

        # Keep track of k with highest validation accuracy
        if acc > val_acc[(k_star - 1) // 5]:
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
