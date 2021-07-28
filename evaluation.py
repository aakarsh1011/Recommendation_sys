from sklearn import model_selection, metrics
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from sklearn.linear_model import LogisticRegression
from lightfm.evaluation import auc_score
import pickle

############################ Evaluation Class ###############################
# Evaluation function takes recommendation model, position prediction model
# Followed by test data for recommendation and for position test data
def rec_eval(model, data):
    # printing training result
    auc = auc_score(model, data).mean()
    print("Testing AUC mean score of recommendation: " + str(auc))


def pos_eval(model, xdata, ydata):
    #ypred = model.predict(xdata)
    # printing testing result
    #fpr, tpr, thresholds = metrics.roc_curve(ydata, ypred)
    #print("Testing AUC mean score of Ad position: " + str(metrics.auc(fpr, tpr)))
    print("LR score on test: "+ str(model.score(xdata, ydata)))
    
   
"""

with open('pos_model_lr', 'rb') as file:
    pickle.load(pos_model_lr,file)

with open('pos_model_knn', 'rb') as file:
    pickle.load(pos_model_knn,file)

with open('pos_model_svm', 'rb') as file:
    pickle.load(pos_model_svm,file)

with open('pos_model_clf', 'rb') as file:
    pickle.load(pos_model_clf,file)
"""

# save test data
#with open('rec_test', 'wb') as file:
#    pickle.load(rec_test,file)

#with open('Xpos_test', 'wb') as file:
#    pickle.load(Xpos_test,file)

#with open('ypos_test', 'wb') as file:
#    pickle.load(ypos_test,file)



def evaluation():
    ## loading artifacts

    with open('rec_model.pkl', 'rb') as file:
        rec_model = pickle.load(file)

    with open('pos_model_lr.pkl','rb') as file:
        pos_model = pickle.load(file)
    with open('rec_test.pkl','rb') as file:
        rec_test = pickle.load(file)
    with open('Xpos_test.pkl','rb') as file:
        Xpos_test = pickle.load(file)

    with open('ypos_test.pkl','rb') as file:
        ypos_test = pickle.load(file)

    rec_eval(rec_model, rec_test)
    pos_eval(pos_model, Xpos_test, ypos_test)

    with open('rec_model_final.pkl', 'wb') as file:
        pickle.dump(rec_model,file)
    
    with open('pos_model_final.pkl', 'wb') as file:
        pickle.dump(pos_model,file)

    return


if __name__ == "__main__":
    evaluation()
