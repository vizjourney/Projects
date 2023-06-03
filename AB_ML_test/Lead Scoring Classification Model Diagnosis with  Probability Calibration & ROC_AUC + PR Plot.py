#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import visualizer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
import warnings
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
warnings.filterwarnings("ignore")


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[5]:


print(train.shape)
print(test.shape)


# In[6]:


train.drop("Unnamed: 0", axis = 1, inplace = True)
test.drop("Unnamed: 0", axis = 1, inplace = True)


# In[7]:


train.sample(frac = 0.1).reset_index().drop("index", axis = 1, inplace = True)
test.sample(frac = 0.1).reset_index().drop("index", axis = 1, inplace = True)


# In[8]:


x_train = train.drop("Converted", axis = 1)
y_train = train.iloc[:, -1]
x_test = test.drop("Converted", axis = 1)
y_test = test.iloc[:, -1]


# In[9]:


pd.set_option('display.max_columns', 500)


# In[10]:


x_train.head()


# In[11]:


ct = ColumnTransformer([('se', StandardScaler(), ['Total Time Spent on Website', 'Page Views Per Visit', 'TotalVisits'])], remainder='passthrough')


# In[12]:


random_forest_pipeline = Pipeline([('transformer', ct), ('RandomForest', RandomForestClassifier(random_state = 42))])
adaboost_pipeline = Pipeline([('transformer', ct), ('Adaboost', AdaBoostClassifier(random_state = 42))])
ExtraTree_pipeline = Pipeline([('transformer', ct), ('ExtraTreeClassifier', ExtraTreesClassifier(random_state = 42))])
BaggingClassifier_pipeline = Pipeline([('transformer', ct), ('BaggingClassifier', BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state = 42))])
GradientBoost_pipeline = Pipeline([('transformer', ct), ('GradientBoosting', GradientBoostingClassifier(random_state = 42))])
dtree_pipeline = Pipeline([('transformer', ct), ('DecisionTree', DecisionTreeClassifier(random_state = 42))])
knn_pipeline = Pipeline([('transformer', ct), ('KNN', KNeighborsClassifier())])
lr_pipeline = Pipeline([('transformer', ct), ('LogisticRegression', LogisticRegression(random_state = 42))])
sgd_pipeline = Pipeline([('transformer', ct), ('StochasticGradient', SGDClassifier(random_state = 42))])
mlp_pipeline = Pipeline([('transformer', ct), ('MLPClassifier', MLPClassifier(random_state = 42))])
naive_pipeline = Pipeline([('transformer', ct), ('NaiveBayes', GaussianNB())])
svc_pipeline = Pipeline([('transformer', ct), ('SVM', SVC(random_state = 42))])
lightgbm_pipeline = Pipeline([('transformer', ct), ('lightgbm', LGBMClassifier(random_state = 42))])
catboost_pipeline = Pipeline([('transformer', ct), ('CatBoost', CatBoostClassifier(random_state = 42, silent = True))])
xgboost_pipeline = Pipeline([('transformer', ct), ('XGBoost', XGBClassifier(random_state = 42))])


# In[13]:


pipeline_list = [random_forest_pipeline, adaboost_pipeline, ExtraTree_pipeline, BaggingClassifier_pipeline, GradientBoost_pipeline,
                dtree_pipeline, knn_pipeline, lr_pipeline, sgd_pipeline, mlp_pipeline, naive_pipeline, svc_pipeline,
                lightgbm_pipeline, catboost_pipeline, xgboost_pipeline]


# In[14]:


pipe_dict = {0: "RandomForest", 1: "Adaboost", 2: "ExtraTree", 3: "BaggingClassifier", 4: "GradientBoosting", 5: "DecisionTree",
            6: "KNN", 7: "Logistic", 8: "SGD Classifier", 9: "MLPClassifier", 10: "NaiveBayes",
            11: "SVM", 12: "LightGBM", 13: "Catboost", 14: "XGBoost"}


# In[15]:


for idx, pipe in enumerate(pipeline_list):
    score = cross_val_score(pipe, x_train, y_train, cv = 10, scoring = 'accuracy')
    print(pipe_dict[idx], ":", score.mean())


# Based on the above results, we will be choosing the **RandomForest Classifier, GradientBoosting, LightGBM & Catboost** on which we are going to test the other metrics to see in depth performance of these 4 models based on several different metrics to choose the best model for our analysis.

# In[16]:


def evaluate_model(model, x_train, y_train, x_test, y_test):
    model = model.fit(x_train, y_train)
    predict_train_y = model.predict(x_train)
    predict_test_y = model.predict(x_test)
    
    print("**Accuracy Score**")
    train_accuracy = accuracy_score(y_train, predict_train_y)
    test_accuracy = accuracy_score(y_test, predict_test_y)
    print("Train Accuracy is: %s"%(train_accuracy))
    print("\nTest Accuracy is: %s"%(test_accuracy))
    print("---------------------------------------------------------")
    
    print("\n**Accuracy Error**")
    train_error = (1-train_accuracy)
    test_error = (1-test_accuracy)
    print("Train Error: %s"%(train_error))
    print("\nTest Error: %s"%(test_error))
    print("---------------------------------------------------------")
    
    print("\n**Classification Report**")
    train_cf_report = pd.DataFrame(classification_report(y_train, predict_train_y, output_dict = True))
    test_cf_report = pd.DataFrame(classification_report(y_test, predict_test_y, output_dict = True))
    print("Train Classification Report:")
    print(train_cf_report)
    print("\n Test Classification Report:")
    print(test_cf_report)
    print("---------------------------------------------------------")
    
    print("\n**Confusion Matrix**")
    train_conf = confusion_matrix(y_train, predict_train_y)
    test_conf = confusion_matrix(y_test, predict_test_y)
    print("Train Confusion Matrix Report:")
    print((train_conf))
    print("\n Test Confusion Matrix Report:")
    print((test_conf))


# ### RANDOM FOREST CLASSIFIER

# In[17]:


rforest = RandomForestClassifier(random_state= 42)


# In[18]:


evaluate_model(rforest, x_train, y_train, x_test, y_test)


# ### GRADIENT BOOSTING CLASSIFIER

# In[19]:


GradientBoost = GradientBoostingClassifier(random_state = 42)


# In[20]:


evaluate_model(GradientBoost, x_train, y_train, x_test, y_test)


# ### LIGHTGBM CLASSIFIER

# In[21]:


lgbm = LGBMClassifier(random_state=42)


# In[22]:


evaluate_model(lgbm, x_train, y_train, x_test, y_test)


# ### CATBOOST CLASSIFIER

# In[23]:


catboost_classif = CatBoostClassifier(random_state=42, silent = True)


# In[24]:


evaluate_model(catboost_classif, x_train, y_train, x_test, y_test)


# ## Models Evaluation & Performance Benchmarking
# <img src = "https://storage.googleapis.com/kaggle-media/launches/model-evaluation-workshop/model-evaluation-spot.png">

# #### A) Model Accuracy:
# 
# **1) Random Forest:** When it comes to train accuracy, Random Forest have the accuracy of 98.4677% while test accuracy has been declined to 91.693% which is significant drop.
# 
# **2) Gradient Boosting:** For train dataset, we have a accuracy score of 91.7350% while for test dataset, we have a accuracy score of 91.657% which is pretty good as there is no much accuracy drop as compared to Random Forest.
# 
# **3) LightGBM:** The LightGBM algorithm gives us a train accuracy of 94.582% while test accuracy of 91.549%.
# 
# **4) CatBoost:** Under Catboost, we have a train accuracy of 94.05% while test accuracy of 92.018%. In Catboost algorithm, we have the highest test accuracy as compared to Random Forest, Gradient Boosting, LightGBM.
# 
# #### B) Model Precision:
# **1) Random Forest:** When it comes to train precision for our class labels, we have a precision score of 97.95% for class label "0" and 99.30% for class label "1" while on test dataset this has been reduced. On testing dataset, precision score for class label "0" is coming out to be 91.84% while for class label "1" it is coming out to be 91.42%.
# 
# This indicating that our model requires parameters needs to be change as the score has come down significantly on the testing dataset.
# 
# **2) Gradient Boosting:** On our training data for class label "0" we have a precision score of 90.71% while for class label "1" we have a precision score of 93.63%,
# 
# On testing dataset for our class label "0" this has been increased from 90.71% to 91.37% while for class label "1" this is slightly down i.e; 92.18% but still it is pretty good as compared to Random Forest.
# 
# **3) Light GBM:** When it comes to Light GBM, our training precision score for class label "0" is coming out to be 94.12% while for class label "1" it is coming 95.37%.
# 
# As far as the testing dataset concern, the precision score of class label "0" is coming out to be 92.11% while for class label "1" it is coming out to be 90.56%.
# 
# **4) CatBoost:** Under CatBoost, for class label "0" under training dataset our precision score is coming out to be 93.43% while for class label "1" it is coming out to be 95.15%.
# 
# For testing dataset, the precision score class label "0" it is slightly down from 93.43% to 92.07% while for class label "1" it is coming out to be 91.91%.
# 
# #### C) F1-Score:
# 
# **1) Random Forest:** If we take a look at the F1-Score for Random Forest Classifier on training dataset, it is coming out to be 98.75% for class label "0" while 97.99% for class label "1".
# 
# On testing dataset, our F1-score has come down from 98.75% to 93.42% for class label "0" while for class label "1" it is coming out to be 88.73% which is again huge drop.
# 
# **2) Gradient Boosting:** On training dataset for class label "0" our F1-score is coming out to be 93.45% while for class label "1" it is coming as 88.80%. For testing dataset, the F1-score for class label "0" has been reduced to 93.42% while for class label "1" it is 88.58%.
# 
# **3) LightGBM:** On training dataset for class label "0" our F1-score is coming out to be 95.64% while for class label "1" it is coming as 92.83%. For testing dataset, the F1-score for class label "0" has been reduced to 93.27% while for class label "1" it is 88.62%.
# 
# **4) CatBoost:** On training dataset for class label "0" our F1-score is coming out to be 95.23% while for class label "1" it is coming as 92.09%. For testing dataset, the F1-score for class label "0" has been increased to 93.68% while for class label "1" it is 89.17%.
# 
# Also when it comes to confusion matrix, we are looking to increase our TP (True Positive) & TN (True Negative) as well as aiming to reduce the FN (False Negative). So for further analysis, we are taking random forest classifier and catboost classifier on which we're going to perform the hyper parameter tuning. 

# ### Random Forest Hyperparameter Tuning

# In[25]:


new_pipeline = Pipeline([('transformer', ct), ('classifier', RandomForestClassifier(random_state=42))])


# In[28]:


rf_param_grid = [{
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': np.arange(100,2000, 200),
                'classifier__max_depth': [None, 10, 20, 30, 50, 70, 80, 100],
                'classifier__min_samples_split': [2, 3, 5, 7, 10],
                'classifier__min_samples_leaf': [1,2,3,4,5,],
                'classifier__max_features': ['auto', 'sqrt', 'log2'],
                'classifier__bootstrap': [True, False]
               }]


# In[29]:


random_search = RandomizedSearchCV(estimator = new_pipeline, param_distributions = rf_param_grid, scoring = 'accuracy', n_jobs = -1, cv = 10, random_state = 42)


# In[30]:


best_rf_model = random_search.fit(x_train, y_train)


# In[31]:


best_rf_model.best_params_


# In[32]:


print("Best Score: %s" %(best_rf_model.best_score_))


# In[33]:


best_rf_model.best_estimator_


# In[34]:


rf_classif_pipeline = Pipeline([('transformer', ct), ('RandomForest', RandomForestClassifier(n_estimators = 300, min_samples_split = 10, min_samples_leaf = 2, max_features = 'auto', bootstrap = False, max_depth = None, random_state = 42))])


# In[35]:


rf_classif_pipeline.fit(x_train, y_train)


# In[36]:


rf_test_prediction = rf_classif_pipeline.predict(x_test)


# In[37]:


rf_test_prob = rf_classif_pipeline.predict_proba(x_test)


# In[39]:


accuracy_score(y_test, rf_test_prediction)


# In[40]:


def check_metric(y_test, y_predict):
    
    print("**Accuracy Score**")
    test_accuracy = accuracy_score(y_test, y_predict)
    print("\nTest Accuracy is: %s"%(test_accuracy))
    print("---------------------------------------------------------")
    
    print("\n**Accuracy Error**")
    test_error = (1-test_accuracy)
    print("\nTest Error: %s"%(test_error))
    print("---------------------------------------------------------")
    
    print("\n**Classification Report**")
    test_cf_report = pd.DataFrame(classification_report(y_test, y_predict, output_dict = True))
    print("\n Test Classification Report:")
    print(test_cf_report)
    print("---------------------------------------------------------")
    
    print("\n**Confusion Matrix**")
    test_conf = confusion_matrix(y_test, y_predict)
    print("\n Test Confusion Matrix Report:")
    print((test_conf))


# In[41]:


check_metric(y_test, rf_test_prediction)


# ### Cat Boost Hyperparameter Tuning

# In[42]:


cb_new_pipeline = Pipeline([('transformer', ct), ('classifier', CatBoostClassifier(random_state=42, task_type = 'CPU', silent = True, eval_metric = 'accuracy'))])


# In[43]:


catboost_params = [{
    'classifier': [CatBoostClassifier()],
    'classifier__iterations': [10],
    'classifier__learning_rate': [0.0001, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    'classifier__depth': [2,4,6,8,10,12],
    'classifier__l2_leaf_reg': [2,3,5,7,9,11,12,15,18,20,25,27],
    'classifier__random_strength': [1],
    'classifier__border_count': [50, 100, 150, 200, 254],
}]


# In[44]:


cb_random_search = RandomizedSearchCV(estimator = cb_new_pipeline, param_distributions = catboost_params, scoring = 'accuracy', n_jobs = -1, cv = 10, random_state = 42)


# In[45]:


cb_random_search.fit(x_train, y_train)


# In[46]:


cb_random_search.best_params_


# In[47]:


cb_random_search.best_score_


# In[48]:


catboost_model = CatBoostClassifier(random_strength=1, learning_rate=0.5, l2_leaf_reg=7, iterations=10, depth=10, border_count=50, 
                                    silent = True, eval_metric='Accuracy', task_type='CPU')


# In[49]:


catboost_model.fit(x_train, y_train, silent = True, plot = True)


# In[53]:


cb_test_prediction = catboost_model.predict(x_test)


# In[54]:


cb_test_prob = catboost_model.predict_proba(x_test)


# In[55]:


check_metric(y_test, cb_test_prediction)


# ## Performance Diagnostic Plot

# In[57]:


from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve


# In[72]:


plt.figure(figsize=(16,6))
ax1 = plt.subplot(121)
rf_disp = plot_precision_recall_curve(rf_classif_pipeline, x_test, y_test, ax = ax1, name = 'Random Forest Precision Recall Curve')
ax2 = plt.subplot(122)
cb_disp = plot_precision_recall_curve(catboost_model, x_test, y_test, ax = ax2, name = 'Catboost Precision Recall Curve')


# In[76]:


#Random Forest Algorithm

fpr, tpr, thresh = roc_curve(y_test, rf_test_prob[:, 1])
rf_auc_score = roc_auc_score(y_test, rf_test_prob[:, 1])


# In[87]:


#CatBoost Algorithm
c_fpr, c_tpr, c_thresh = roc_curve(y_test, cb_test_prob[:, 1])
cb_auc_score = roc_auc_score(y_test, cb_test_prob[:, 1])


# In[91]:


# Plotting ROC/AUC Curve

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.plot(fpr, tpr)
plt.title("AUC Score of Random Forest: %s" %(rf_auc_score))
plt.plot([0,1], [0,1], color = 'k', linestyle = '--')
plt.subplot(122)
plt.plot(c_fpr, c_tpr, color = 'darkgreen')
plt.title("AUC Score of CatBoost: %s" %(cb_auc_score))
plt.plot([0,1], [0,1], color = 'k', linestyle = '--')


# ### Probability Calibration Plot

# In[124]:


rf_y, rf_x = calibration_curve(y_test, rf_test_prob[:, 1], n_bins = 5)
cb_y, cb_x = calibration_curve(y_test, cb_test_prob[:, 1], n_bins = 5)


# In[125]:


# Probability Calibration Plot

plt.figure(figsize=(12,8))
plt.title("Probability Calibration Plot")
plt.plot(rf_x, rf_y, marker = 'o', label = 'RForest')
plt.plot(cb_x, cb_y, marker = 'o', label = 'CatBoost')
plt.plot([0.0,1.0], [0.0,1.0], color = 'k', linestyle = '--')
plt.xlabel("Predicted Probability")
plt.xlabel("True, Probability in each bin")
plt.legend()


# When we take a look at the calibration plot, we can see that from the 1st bin to 3rd bin, our Random Forest is under-predicting the probability where as the CatBoost is over-predicting it in between 3rd to 5th bin.

# In[143]:


x_train_ct = ct.fit_transform(x_train)
x_test_ct = ct.fit_transform(x_test)


# In[144]:


x_train_ct = pd.DataFrame(x_train_ct, columns = x_train.columns)
x_test_ct = pd.DataFrame(x_test_ct, columns = x_test.columns)


# In[147]:


rf_model = RandomForestClassifier(n_estimators = 300, min_samples_split = 10, min_samples_leaf = 2, max_features = 'auto', bootstrap = False, max_depth = None, random_state = 42)
cb_model = CatBoostClassifier(random_strength=1, learning_rate=0.5, l2_leaf_reg=7, iterations=10, depth=10, border_count=50, silent = True, eval_metric='Accuracy')


# In[149]:


rf_sigmoid = CalibratedClassifierCV(base_estimator=rf_model, cv = 5)
rf_iso = CalibratedClassifierCV(base_estimator=rf_model, method = 'isotonic', cv = 5)


# In[148]:


cb_sigmoid = CalibratedClassifierCV(base_estimator=cb_model, cv = 5)
cb_iso = CalibratedClassifierCV(base_estimator=cb_model, method = 'isotonic', cv = 5)


# In[150]:


rf_sigmoid.fit(x_train_ct, y_train)


# In[151]:


rf_sigmoid_prob = rf_sigmoid.predict_proba(x_test_ct)


# In[152]:


rf_iso.fit(x_train_ct, y_train)


# In[153]:


rf_iso_prob = rf_iso.predict_proba(x_test_ct)


# In[156]:


rf_y, rf_x = calibration_curve(y_test, rf_test_prob[:, 1], n_bins = 5)
rf_sm_y, rf_sm_x = calibration_curve(y_test, rf_sigmoid_prob[:, 1], n_bins = 5)
rf_iso_y, rf_iso_x = calibration_curve(y_test, rf_iso_prob[:, 1], n_bins = 5)


# ### Random Forest Probability Calibration Plot -  (Sigmoid/Isotonic/Uncalibrated)

# In[157]:


# Probability Calibration Plot

plt.figure(figsize=(12,8))
plt.title("Probability Calibration Plot")
plt.plot(rf_x, rf_y, marker = 'o', label = 'Uncalibrated')
plt.plot(rf_sm_x, rf_sm_y, marker = 'o', label = 'Sigmoid Calibrated')
plt.plot(rf_iso_x, rf_iso_y, marker = 'o', label = 'Isotonic Calibrated')
plt.plot([0.0,1.0], [0.0,1.0], color = 'k', linestyle = '--')
plt.xlabel("Predicted Probability")
plt.xlabel("True, Probability in each bin")
plt.legend()


# In[158]:


cb_sigmoid = CalibratedClassifierCV(base_estimator=cb_model, cv = 5)
cb_iso = CalibratedClassifierCV(base_estimator=cb_model, method = 'isotonic', cv = 5)


# In[159]:


cb_sigmoid.fit(x_train_ct, y_train)


# In[161]:


cb_sigmoid_prob = cb_sigmoid.predict_proba(x_test_ct)


# In[162]:


cb_iso.fit(x_train_ct, y_train)


# In[163]:


cb_iso_prob = cb_iso.predict_proba(x_test_ct)


# In[164]:


cb_y, cb_x = calibration_curve(y_test, cb_test_prob[:, 1], n_bins = 5)
cb_sm_y, cb_sm_x = calibration_curve(y_test, cb_sigmoid_prob[:, 1], n_bins = 5)
cb_iso_y, cb_iso_x = calibration_curve(y_test, cb_iso_prob[:, 1], n_bins = 5)


# ### CatBost Probability Calibration Plot -  (Sigmoid/Isotonic/Uncalibrated)

# In[165]:


# Probability Calibration Plot

plt.figure(figsize=(12,8))
plt.title("Probability Calibration Plot")
plt.plot(cb_x, cb_y, marker = 'o', label = 'Uncalibrated')
plt.plot(cb_sm_x, cb_sm_y, marker = 'o', label = 'Sigmoid Calibrated')
plt.plot(cb_iso_x, cb_iso_y, marker = 'o', label = 'Isotonic Calibrated')
plt.plot([0.0,1.0], [0.0,1.0], color = 'k', linestyle = '--')
plt.xlabel("Predicted Probability")
plt.xlabel("True, Probability in each bin")
plt.legend()


# As you can see how probability calibration can be helpful. You can use this probability calibration in case of Imbalance dataset as well to tweak the model performance in combination with AUC/ROC for model diagnosis.
