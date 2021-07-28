from sklearn import model_selection, metrics
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from lightfm.evaluation import auc_score
import click
from scipy import sparse
import json
import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def train_test_split(user_ad_csr, Ad_Data_pos, test_size):

    rs = np.random.RandomState(3)
    # train test split
    # pass csr matrix of user_ad
    rec_tr, rec_test = random_train_test_split(
        user_ad_csr, test_percentage=test_size, random_state=rs)

    # train test split of ad_pos
    # data splitting
    X = Ad_Data_pos.drop(columns=['Position'])
    Y = Ad_Data_pos['Position']

    # train test split
    Xpos_train, Xpos_test, ypos_train, ypos_test = model_selection.train_test_split(
        X, Y, test_size=0.20, random_state=42)

    return rec_tr, rec_test, Xpos_train, Xpos_test, ypos_train, ypos_test


def rec_sys(rec_tr, no_components, learning_schedule, loss, lr, i_alpha, u_alpha, sam, e, num):
    # initializing the model
    model = LightFM(no_components=no_components,
                    learning_schedule=learning_schedule,
                    loss=loss,
                    learning_rate=lr,
                    item_alpha=i_alpha,
                    user_alpha=u_alpha,
                    max_sampled=sam)

    # pass csr matrix of user_ad
    model_h = model.fit(rec_tr,
                        epochs=e,
                        num_threads=num,
                        verbose=False)

    # printing training result
    auc_train = auc_score(model_h, rec_tr).mean()
    print("Training AUC mean score of recommendation: " + str(auc_train))
    return model_h


def ad_pos(X_train, y_train):

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    # print("score on test: " + str(lr.score(x_test, y_test)))
    print("LR score on train: " + str(lr.score(X_train, y_train)))
    print("\n")

    """
    #Naive Bayes
    mnb = MultinomialNB().fit(X_train, y_train)
    # print("score on test: " + str(mnb.score(x_test, y_test)))
    print("MNB score on train: "+ str(mnb.score(X_train, y_train)))
    print("\n")

    #KNN
    knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    knn.fit(X_train, y_train)
    print("train shape: " + str(X_train.shape))
    print("KNN score on train: "+ str(knn.score(X_train, y_train)))
    print("\n")

    #SVM
    svm=LinearSVC()
    svm.fit(X_train, y_train)
    print("SVM score on train: "+ str(svm.score(X_train, y_train)))
    print("\n")

    #Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    #print("score on test: "  + str(clf.score(x_test, y_test)))
    print("score on train: " + str(clf.score(X_train, y_train)))
    print("\n")

    """
    # change return algorithm as per needed

    return lr


@click.command()
@click.option('--no-components', default=58)
@click.option('--learning-schedule', default='adadelta')
@click.option('--loss', default='warp-kos')
@click.option('--lr', default=0.05909687790375976)
@click.option('--i-alpha', default=1.2663639497700659e-08)
@click.option('--u-alpha', default=9.952080491129124e-10)
@click.option('--sam', default=8)
@click.option('--e', default=6)
@click.option('--num', default=16)
@click.option('--test-size', default=0.2)
############################ TRAINING ##############################
# Ad_Data_csr, user_ad_csr, interactions, item_dict, user_dict, Ad_Data_pos
def training(no_components, learning_schedule, loss, lr, i_alpha, u_alpha, sam, e, num, test_size):

    # loading the artifacts
    ######################################

    user_ad_csr = sparse.load_npz(r"E:\sapient\personalize\user_ad_csr.npz")

    with open(r'E:\sapient\personalize\Ad_Data_pos.pkl', 'rb') as fp:
        Ad_Data_pos = pickle.load(fp)
    ######################################

    """
    Takes Ad_Data CSR file, User_ad CSR file, user_ad_interaction table, item_dict, user_dict, AD_position_Data

    Returns:
    1. Recommendation system model
    2. Ad_position model
    3. recommendation test data
    4. Ad_position test data

    Displays:
    1. Training evaluation of rec sys
    2. Training evaluation of ad position

    """

    rec_tr, rec_test, Xpos_train, Xpos_test, ypos_train, ypos_test = train_test_split(
        user_ad_csr, Ad_Data_pos, test_size)

    rec_model = rec_sys(rec_tr, no_components, learning_schedule,
                        loss, lr, i_alpha, u_alpha, sam, e, num)

    pos_model_lr = ad_pos(Xpos_train, ypos_train)

    ######################################
    #rec_model, pos_model, rec_test, Xpos_test, ypos_test
    # save model

    with open('rec_model.pkl', 'wb') as file:
        pickle.dump(rec_model, file)

    with open('pos_model_lr.pkl', 'wb') as file:
        pickle.dump(pos_model_lr, file)

    """    
    with open('pos_model_nb.pkl', 'wb') as file:
        pickle.dump(pos_model_nb,file)

    with open('pos_model_knn.pkl', 'wb') as file:
        pickle.dump(pos_model_knn,file)

    with open('pos_model_svm.pkl', 'wb') as file:
        pickle.dump(pos_model_svm,file)

    with open('pos_model_clf.pkl', 'wb') as file:
        pickle.dump(pos_model_clf,file)

    """

    # save test data
    with open('rec_test.pkl', 'wb') as file:
        pickle.dump(rec_test, file)

    with open('Xpos_test.pkl', 'wb') as file:
        pickle.dump(Xpos_test, file)

    with open('ypos_test.pkl', 'wb') as file:
        pickle.dump(ypos_test, file)
    ######################################

    return


if __name__ == "__main__":
    training()
