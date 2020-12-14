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
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    # update theta
    dl_dtheta = np.zeros(len(theta[0]))

    def theta_helper_inner(c, theta, beta):
        top = 2 * c * np.exp(theta - beta) - np.exp(theta - beta)
        bot = (c * np.exp(theta - beta) + (1 - c) * (-(np.exp(theta - beta)) / (np.exp(theta - beta) + 1) + 1) * (
                np.exp(theta - beta) + 1)) * (np.exp(theta - beta) + 1)
        return top / bot

    for n in range(len(user_id)):
        i = user_id[n]
        j = question_id[n]

        theta_i = theta[0][i]
        beta_j = beta[0][j]
        c_ij = is_correct[n]
        dl_dtheta[i] += theta_helper_inner(c_ij, theta_i, beta_j)

    beta = beta + lr * (dl_dbeta - beta)
    theta = theta + lr * (dl_dtheta - theta)

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


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # find best alpha, iterations
    # tune hyperparameters, commented to save time
    # alpha = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
    # iterations = [10, 50, 75, 100, 125, 150, 200, 250]
    # alpha = [0.0005, 0.005, 0.05]
    # iterations = [100, 150, 200]
    # alpha = [0.0025, 0.00375, 0.005, 0.00625, 0.0075]
    # iterations = [125, 150, 175]

    # alpha = [0.0005, 0.05]
    # iterations = [20, 30, 40, 50, 60]

    alpha = [0.00625]
    iterations = [125]
    best_alpha, best_iterations, best_val_acc = None, None, -float(sys.maxsize)
    val_acc_plot = []
    for a in alpha:
        for i in iterations:
            val_theta, val_beta, val_acc_list, val_log_likelihood_list = irt(val_data, val_data, a, i)
            val_accuracy = evaluate(test_data, val_theta, val_beta)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_alpha, best_iterations = a, i
                val_acc_plot = val_log_likelihood_list

    # best_alpha = 0.00625
    # best_iterations = 100

    # plot validation
    # print('best alpha chosen = ' + str(best_alpha))
    # print('best iterations chosen= ' + str(best_iterations))

    # training
    train_theta, train_beta, train_acc_list, train_acc_plot = irt(train_data,
                                                                             train_data,
                                                                             best_alpha,
                                                                             best_iterations)

    plt.plot(val_acc_plot, label='val')
    plt.plot(train_acc_plot, label='train')
    plt.xlabel("iterations")
    plt.ylabel("NLLK")
    plt.legend(loc="best")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    ## evaluate on test data
    print('validation accuracy: ' + str(best_val_acc))
    print("test accuracy: " + str(evaluate(test_data, train_theta, train_beta)))
    #####################################################################
    plt.clf()
    five_questions_idx = random.sample([j for j in range(len(train_beta[0]))], 5)
    for i, j in enumerate(five_questions_idx):
        probabilities = []
        for theta in train_theta:
            probabilities.append((np.exp(theta - train_beta[0][j])) / (1 + (np.exp(theta - train_beta[0][j]))))

        plt.subplot(1, 5, i + 1)
        plt.scatter(train_theta, probabilities)
        plt.xlabel("theta")
        plt.ylabel("p(c)")

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
