# AUTHOR : PATRICK LIN

import sys

from project.utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(theta, beta, k):
    """ Apply sigmoid function.
    """
    # return np.exp(k * (theta - beta)) / (1 + np.exp(k * (theta - beta)))
    return 1 / (1 + np.exp(-k * (theta - beta)))


def neg_log_likelihood(data, theta, beta, k):
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
    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]
        if is_correct[n] == 1:
            log_lklihood += np.log(
                sigmoid(theta[0][i], beta[0][j], k[0][j])
            )
        else:
            log_lklihood += np.log(
                1 - sigmoid(theta[0][i], beta[0][j], k[0][j])
            )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_params(data, lr, theta, beta, k):
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

    dl_dtheta = np.zeros(len(theta[0]))

    def theta_helper_inner(c, theta, beta, k):
        top = 2 * c * k * np.exp(theta - beta) - k * np.exp(theta - beta)
        bot = (c * k * np.exp(theta - beta) + (1 - c) * (
                    -(k * np.exp(theta - beta)) / (k * np.exp(theta - beta) + 1) + 1) * (
                           k * np.exp(theta - beta) + 1)) * (k * np.exp(theta - beta) + 1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        k_j = k[0][j]
        c_ij = is_correct[n]
        dl_dtheta[i] += theta_helper_inner(c_ij, theta_i, beta_j, k_j)

    dl_dbeta = np.zeros(len(beta[0]))

    def beta_helper_inner(c, theta, beta, k):
        top = k * np.exp(theta - beta) - 2 * c * k * np.exp(theta - beta)
        bot = (c * k * np.exp(theta - beta) + (1 - c) * (
                    -(k * np.exp(theta - beta)) / (k * np.exp(theta - beta) + 1) + 1) * (
                           k * np.exp(theta - beta) + 1)) * (k * np.exp(theta - beta) + 1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        k_j = k[0][j]
        c_ij = is_correct[n]
        dl_dbeta[j] += beta_helper_inner(c_ij, theta_i, beta_j, k_j)

    dl_dk = np.zeros(len(beta[0]))

    def k_helper_inner(c, theta, beta, k):
        top = 2 * c * np.exp(theta - beta) - np.exp(theta - beta)
        bot = (c * k * np.exp(theta - beta) + (1 - c) * (
                    -(k * np.exp(theta - beta)) / (k * np.exp(theta - beta) + 1) + 1) * (
                           k * np.exp(theta - beta) + 1)) * (k * np.exp(theta - beta) + 1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        k_j = k[0][j]
        c_ij = is_correct[n]
        dl_dk[j] += k_helper_inner(c_ij, theta_i, beta_j, k_j)

    beta = beta + lr * dl_dbeta
    theta = theta + lr * dl_dtheta
    k = k + lr * dl_dk

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, k


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
    # TODO: Initialize parameters.
    theta = np.zeros((1, 542))
    beta = np.zeros((1, 1774))
    k = np.ones((1, 1774))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, k=k)
        score = evaluate(data=val_data, theta=theta, beta=beta, k=k)
        val_acc_lst.append(score)
        print("ITERATION = {} \t NLLK = {} \t Score = {}".format(i, neg_lld, score))
        theta, beta, k = update_params(data, lr, theta, beta, k)

    print("==================================================================")
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, k, val_acc_lst


def evaluate(data, theta, beta, k):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    N = len(data['user_id'])
    for n in range(N):
        i = data["user_id"][n]
        j = data["question_id"][n]
        p_a = sigmoid(theta[0][i], beta[0][j], k[0][j])
        pred.append(p_a >= 0.5)

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # experiment to see if validation data sucks:
    print('validation data length', 'train data length')
    print(len(val_data['user_id']), len(train_data['question_id']))

    alpha = [0.00625]
    iterations = [100]
    best_alpha, best_iterations, best_val_acc = None, None, -float(sys.maxsize)
    val_acc_plot = []
    for a in alpha:
        for i in iterations:
            val_theta, val_beta, val_k, val_acc_list = irt(val_data, val_data, a, i)
            val_accuracy = evaluate(test_data, val_theta, val_beta, val_k)
            print('validation accuracy: ' + str(val_accuracy))

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_alpha, best_iterations = a, i
                val_acc_plot = val_acc_list

    print('best validation accuracy: ' + str(best_val_acc))

    train_theta, train_beta, train_k, train_acc = irt(train_data,
                                                      train_data,
                                                      best_alpha,
                                                      best_iterations)

    plt.plot(val_acc_plot, label='validation')
    plt.plot(train_acc, label='training')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    plt.show()

    print("test accuracy: " + str(evaluate(test_data, train_theta, train_beta, train_k)))


if __name__ == "__main__":
    main()
