
from sklearn.impute import KNNImputer
from project.utils import *
import os

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


def load_subject_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "subject_id": [],
        "name": []
    }

    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                # data["subject_id"].append(int(row[0]))
                # subjects = (row[1][1:-1]).split(',')
                # all_subjects = []
                # for a in subjects:
                #     all_subjects.append(a)
                # data["name"].append(all_subjects)
                data["subject_id"].append(int(row[0]))
                data["name"].append(row[1])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def main():
    train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")
    question_metadata = load_questions_meta_csv("../data/question_meta.csv")
    subject_metadata = load_subject_meta_csv("../data/subject_meta.csv")
    print(question_metadata.keys(), subject_metadata.keys(), train_data.keys())
    # print(len(question_metadata['question_id']), len(question_metadata['subject_id']))
    # print(len(subject_metadata["subject_id"]), len(subject_metadata["name"]))
    # print(len(train_data['user_id']))

    data_augmented = np.empty((542, 1774, 388))
    data_augmented[:] = np.NaN

    for i in range(56688):
        uid = train_data['user_id'][i]
        qid = train_data['question_id'][i]
        correct = train_data['is_correct'][i]
        q_subjects_idx = question_metadata['question_id'].index(qid)

        # gets list of subjects associated with question
        q_subjects = question_metadata['subject_id'][q_subjects_idx]
        for s in q_subjects:
            data_augmented[uid][qid][s] = correct

    print(data_augmented, data_augmented.shape)

    imputer = KNNImputer(n_neighbors=3)
    mat = np.reshape(imputer.fit_transform(np.reshape(data_augmented, (542, 1774 * 388))), (542, 1774, 388))


if __name__ == "__main__":
    main()
