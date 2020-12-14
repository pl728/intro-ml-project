# AUTHOR : PATRICK LIN

import sys

from project.utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    N = len(user_id)
    for n in range(N):
        i = user_id[n]
        j = question_id[n]
        if is_correct[n] == 1:
            log_lklihood += np.log(sigmoid(theta[0][i] - beta[0][j]))
        else:
            log_lklihood += np.log(1 - sigmoid(theta[0][i] - beta[0][j]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta_a(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    # get data
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    dl_dbeta = np.zeros(len(beta[0]))

    def beta_helper_inner(c, theta, beta):
        top = np.exp(theta - beta) - 2 * c * np.exp(theta - beta)
        bot = (c * np.exp(theta - beta) + (1 - c) * (-(np.exp(theta - beta)) / (np.exp(theta - beta) + 1) + 1) * (
                    np.exp(theta - beta) + 1)) * (np.exp(theta - beta) + 1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        c_ij = is_correct[n]
        dl_dbeta[j] += beta_helper_inner(c_ij, theta_i, beta_j)
    beta = beta + lr * dl_dbeta
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    # update theta
    dl_dtheta = np.zeros(len(theta[0]))

    def theta_helper_inner(c, theta, beta):
        top = 2 * c * np.exp(theta - beta) - np.exp(theta - beta)
        bot = (c * np.exp(theta - beta) + (1 - c) * (-(np.exp(theta - beta)) / (np.exp(theta - beta) + 1) + 1) * (
                    np.exp(theta - beta) + 1)) * (np.exp(theta-beta)+1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        c_ij = is_correct[n]
        dl_dtheta[i] += theta_helper_inner(c_ij, theta_i, beta_j)

    theta = theta + lr * dl_dtheta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # theta = np.full((1, 542), 100)
    # beta = np.full((1, 1774), 50)
    # theta = np.random.rand(1, 542)
    # beta = np.random.rand(1, 7774)
    theta = np.zeros((1, 542))
    beta = np.zeros((1, 1774))

    val_acc_lst = []
    log_likelihood_list = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)

        # added
        log_likelihood_list.append(neg_lld)
        #
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("ITERATION = {} \t NLLK = {} \t Score = {}".format(i, neg_lld, score))
        theta, beta = update_theta_beta_a(data, lr, theta, beta)

    print("==================================================================")
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, log_likelihood_list


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[0][u] - beta[0][q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

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


def main():
    train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")
    question_metadata = load_questions_meta_csv("../data/question_meta.csv")
    subject_metadata = load_subject_meta_csv("../data/subject_meta.csv")
    print(question_metadata.keys(), subject_metadata.keys(), train_data.keys())
    print(len(question_metadata['question_id']), len(question_metadata['subject_id']))
    print(len(subject_metadata["subject_id"]), len(subject_metadata["name"]))
    print(len(train_data['user_id']))

    best_alpha = 0.00625
    best_iterations = 125

    student_subject = np.empty((542, 388))
    student_subject[:] = np.NaN

    data = {
        'user_id': [],
        'subject_id': [],
        'is_correct': []
    }

    N = len(train_data['user_id'])
    for n in range(N):
        uid = train_data['user_id'][n]
        qid = train_data['question_id'][n]
        c = train_data['is_correct'][n]
        qid_subjects = question_metadata['subject_id'][question_metadata['question_id'].index(qid)]
        for subject in qid_subjects:
            if student_subject[uid][subject] != np.NaN:
                student_subject[uid][subject] += c
            data['user_id'].append(uid)
            data['subject_id'].append(subject)
            data['is_correct'].append(c)
    print(data)
    for i in range(len(data['user_id'])):
        ...



if __name__ == "__main__":
    main()
