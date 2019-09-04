#!/usr/bin/env python
# coding: utf-8

# # Machine learning for Power System Stability Analysis

# <p style="background-color:azure;padding:10px;border:2px solid lightsteelblue"><b>Author:</b> Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# <br>
# University of Split, FESB, Department of Power Engineering <br>R. Boskovica 32, HR-21000 Split, Croatia, EU.</p>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from scipy import stats


# In[ ]:


from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


# In[ ]:


# Inline figures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Figure aesthetics
sns.set(context='notebook', style='white', font_scale=1.2)
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})


# In[ ]:


# ancilary function from: https://github.com/amueller/introduction_to_ml_with_python/blob/master/mglearn/tools.py
def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f", fontsize=14):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center", fontsize=fontsize)
    return img


# ### Transformer diagnostic data and health index values

# In[ ]:


data = pd.read_csv('GridDictionary.csv')
data.head()


# In[ ]:


# Flip ones into zeros for the "Stability" column
#data['Stability'] = 1 - data['Stability']


# In[ ]:


# Percentage of "ones" in the "Stability" column
print('There is {:.1f}% of unstable cases in the dataset!'.format(data['Stability'].sum()/float(len(data['Stability']))*100.))


# ### Select a random subset of the original data

# In[ ]:


# Select a random subset of the original dataset (without replacement)
SUBSET_SIZE = 2000
random_idx = np.random.choice(data.index, size=SUBSET_SIZE, replace=False)
data = data.iloc[random_idx]


# ### Data preprocessing and splitting

# In[ ]:


# Training dataset
no_features = len(data.columns) - 1
X_data = data.iloc[:,0:no_features]  # features
print('X_data', X_data.shape)
y_data = data['Stability']
print('y_data', y_data.shape)


# In[ ]:


# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, shuffle=True)


# In[ ]:


print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[ ]:


y_t = data[['Stability']].copy()
idx = y_test.index.values
y_t = y_t.loc[idx]
y_t.shape


# #### StandardScaler

# In[ ]:


# Standardize the input data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### LogisticRegression

# In[ ]:


# Logistic Regression (with fixed hyper-parameters)
lreg = LR(C=100.,  # fixed "C" hyper-parameter
          multi_class='ovr', solver='newton-cg', n_jobs=-1)
lreg.fit(X_train, y_train)  # fit model to data
y_lr = lreg.predict_proba(X_test)  # predict on new data


# In[ ]:


pred = lreg.predict(X_test)
labels = ['Stab', 'NotStab']
# confusion matrix
scores_image = heatmap(metrics.confusion_matrix(y_test, pred), xlabel='Predicted label', 
                       ylabel='True label', xticklabels=labels, yticklabels=labels, 
                       cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


# classification report
print(metrics.classification_report(y_test, pred, target_names=labels))


# #### GridSearchCV

# In[ ]:


# Grid-search with cross validation for optimal model hyper-parameters
parameters = {'C':[1., 10., 50., 100., 500., 1000.]}
lreg = GridSearchCV(estimator=LR(multi_class='ovr', solver='newton-cg'), 
                    param_grid=parameters, cv=3, scoring='f1',  # notice the "scoring" method!
                    refit=True, n_jobs=-1, iid=False)
lreg.fit(X_train, y_train)
# Best value of hyper-parameter "C"
best_c = lreg.best_params_['C']
print('Best value: C = {:g}'.format(best_c))


# In[ ]:


# Average classification accuracy with cross validation
scores = cross_val_score(LR(C=best_c, multi_class='ovr', solver='newton-cg'), 
                         X_train, y_train, cv=3)  # it doesn't return a model!
print('Score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# ### Feature selection with Pipeline and GridSearch

# In[ ]:


# Optimize the number of features and the classifier's hyper-parameters 
# at the same time, using pipline and grid search with cross-validation
pca = PCA()  # do NOT set "n_components" here!
logreg = LR(multi_class='ovr', solver='newton-cg')  # multinomial classification!
pipe = Pipeline([('pca',pca), ('logreg',logreg)])
param_grid = {'pca__n_components': [10, 50, 100],  # PCA
              'logreg__C': [10., 100., 500.]}      # LogisticRegression
grid_pipe = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, 
                         scoring='f1', refit=True, n_jobs=-1, iid=False)
grid_pipe.fit(X_train, y_train)
print('Best parameter (CV score = {:0.3f}):'.format(grid_pipe.best_score_))
print(grid_pipe.best_params_)


# In[ ]:


# Predict probability on test data
y_lr = grid_pipe.predict_proba(X_test)
y_t['logreg'] = y_lr.argmax(axis=1)


# In[ ]:


y_t.head()


# ### Support Vector Machine

# In[ ]:


parameters ={'C':[1., 10., 100., 500., 1000.],
             'gamma':[0.0001, 0.001, 0.01, 0.1, 1.]}
svc = GridSearchCV(estimator=svm.SVC(kernel='rbf', probability=True), 
                   param_grid=parameters, cv=3,
                   scoring='f1', refit=True, n_jobs=-1, iid=False)
svc.fit(X_train, y_train)


# In[ ]:


# Best model parameters
best_parameters = svc.best_params_
print("Best parameters from GridSearch: {}".format(svc.best_params_))


# In[ ]:


scores = cross_val_score(svm.SVC(**best_parameters), X_train, y_train, cv=3)
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[ ]:


results = pd.DataFrame(svc.cv_results_)
scores = np.array(results.mean_test_score).reshape(len(parameters['C']), len(parameters['gamma']))


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
heatmap(scores, xlabel='gamma', xticklabels=parameters['gamma'], 
        ylabel='C', yticklabels=parameters['C'], cmap="viridis", ax=ax)
plt.show()


# #### RandomizedSearchCV

# In[ ]:


parameters = {'C':stats.expon(scale=100), 'gamma':stats.expon(scale=.1)}
svc2 = RandomizedSearchCV(estimator=svm.SVC(kernel='rbf', probability=True), 
                          param_distributions=parameters, cv=3, n_iter=50,  # 50 iterations!
                          scoring='neg_log_loss',  # notice the scoring method!
                          refit=True, n_jobs=-1, iid=False)
svc2.fit(X_train, y_train)


# In[ ]:


# Best model parameters
best_parameters = svc2.best_params_
print("Best parameters from RandomSearch: {}".format(svc2.best_params_))


# In[ ]:


scores = cross_val_score(svm.SVC(**best_parameters), X_train, y_train, cv=3)
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[ ]:


y_svc2 = svc2.predict_proba(X_test)
y_t['svc'] = y_svc2.argmax(axis=1)


# #### Precision-Recall Tradeoff

# In[ ]:


y_probas = cross_val_predict(svm.SVC(**best_parameters, probability=True), 
                             X_train, y_train, cv=3, method='predict_proba')
y_scores = y_probas[:,1]  # score = probability of positive class


# In[ ]:


precisions, recalls, thresholds = metrics.precision_recall_curve(y_train, y_scores)


# In[ ]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('SVC Precision-Recall tradeof')
ax.plot(thresholds, precisions[:-1], lw=2, label='Precision')
ax.plot(thresholds, recalls[:-1], lw=2, label='Recall')
plt.vlines(0.5, 0, 1, linestyles='--', label='Threshold = 0.5')
ax.set_xlabel('Thresholds')
ax.legend(loc='best')
ax.set_ylim(ymin=0.8, ymax=1.02)
ax.grid()
fig.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.plot(precisions, recalls, lw=2, label='SVC')
default = np.argmin(np.abs(thresholds - 0.5))
ax.plot(precisions[default], recalls[default], '^', c='k', markersize=10, 
        label='Threshold = 0.5', fillstyle='none', mew=2)
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
plt.show()


# In[ ]:


# Average precision-recall score
y_test_score = svc2.predict_proba(X_test)[:,1]
average_precision = metrics.average_precision_score(y_test, y_test_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# In[ ]:


# Determine a class from the predicted probability by using 
# the user-specified threshold value (not a default of 0.5)
THRESHOLD = 0.6
preds = np.where(y_test_score > THRESHOLD, 1, 0)


# In[ ]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, preds), metrics.recall_score(y_test, preds),
                   metrics.precision_score(y_test, preds), metrics.roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"])


# ### ExtraTreesClassifier

# In[ ]:


# ExtraTreesClassifier (ensemble learner) with grid search 
# and cross-validation for hyper-parameters optimisation
parameters = {'n_estimators':[5, 10, 15, 20], 
              'criterion':['gini', 'entropy'], 
              'max_depth':[2, 5, None]}
trees = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=parameters, 
                     cv=3, scoring='neg_log_loss', refit=True, n_jobs=-1, iid=False) 
trees.fit(X_train, y_train)


# In[ ]:


# Best model parameters
best_parameters = trees.best_params_
print("Best parameters: {}".format(trees.best_params_))


# In[ ]:


scores = cross_val_score(ExtraTreesClassifier(**best_parameters), X_train, y_train, cv=3)
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[ ]:


y_trees = trees.predict_proba(X_test)
y_t['tree'] = y_trees.argmax(axis=1)


# ### RandomForest classifier (ensemble learner)

# In[ ]:


# RandomForestClassifier (ensemble learner for classification)
parameters = {'n_estimators':[10, 15, 20], 
              'criterion':['gini', 'entropy'],
              'max_features':[4, 'auto'],
              'max_depth':[2, None]}
# grid search and cross-validation for hyper-parameters optimisation
forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, 
                      cv=3, scoring='neg_log_loss', refit=True, n_jobs=-1, iid=False) 
forest.fit(X_train, y_train)


# In[ ]:


best_parameters = forest.best_params_
print("Best parameters: {}".format(forest.best_params_))


# In[ ]:


scores = cross_val_score(RandomForestClassifier(**best_parameters), X_train, y_train, cv=3)
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[ ]:


y_forest = forest.predict_proba(X_test)
y_t['forest'] = y_forest.argmax(axis=1)


# ### GradientBoosting classifier with feature importance analysis

# In[ ]:


# Train & evaluate model performance
def train_and_evaluate(model, X, y, ns=3):
    # k-fold cross validation iterator 
    cv = KFold(n_splits=ns, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')  # scoring method is f1!
    print('Average score using {:d}-fold CV: {:g} +/- {:g}'.format(ns, np.mean(scores), np.std(scores)))


# In[ ]:


# Gradient Boosting Classifier
clf_gb = GradientBoostingClassifier()
train_and_evaluate(clf_gb, X_train, y_train, 3)
clf_gb.fit(X_train, y_train)


# In[ ]:


# Feature importance
feature_importance = clf_gb.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5


# In[ ]:


# Select top features
TOP = 10
print('Most relevant {:d} features according to the GradientBoostingClassifier:'.format(TOP))
data.columns.values[sorted_idx][-TOP:][::-1]


# In[ ]:


# Plot relative feature importance
fig, ax = plt.subplots(figsize=(5,5))
ax.barh(pos[-TOP:], feature_importance[sorted_idx][-TOP:], align='center', color='magenta', alpha=0.6)
plt.yticks(pos[-TOP:], data.columns[sorted_idx][-TOP:])
ax.set_xlabel('Feature Relative Importance')
#ax.grid(which='major', axis='x')
plt.tight_layout()
plt.show()


# In[ ]:


# Correlation matrix of selected features
pearson = data[data.columns[sorted_idx][-TOP:]].corr('pearson')
pearson.iloc[-1][:-1].sort_values()
# Correlation matrix as heatmap (seaborn)
fig, ax = plt.subplots(figsize=(6.5,5.5))
sns.heatmap(pearson, annot=True, annot_kws=dict(size=9), vmin=-1, vmax=1, ax=ax)
plt.tight_layout()
plt.show()


# In[ ]:


# Predict on new data
y_gb = clf_gb.predict_proba(X_test)
y_t['gbr'] = y_gb.argmax(axis=1)


# ## Ensemble models using voting principle

# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Ensembling consists of pooling together the predictions of a set of different models, to produce better predictions. The key to making ensembling work is the diversity of the set of classifiers. Diversity is what makes ensembling work. For this reason, one should ensemble models that are as good as possible while being <b>as different as possible</b>. This typically means using very different network architectures or even different brands of machine-learning approaches. This is exactly what has been proposed here.</p>

# ### Soft voting

# In[ ]:


clf = VotingClassifier(estimators=[('logreg', lreg),     # LogisticRegression
                                   ('svm', svc2),        # SVC
                                   ('forest', forest)],  # RandomForest 
                       weights=[1, 1, 1],  # classifier relative weights
                       voting='soft')
clf = clf.fit(X_train, y_train)


# In[ ]:


y_clf = clf.predict_proba(X_test)
y_t['vote'] = y_clf.argmax(axis=1)


# In[ ]:


scores = cross_val_score(clf, X_train, y_train, cv=3)
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# #### Predictions using individual classifiers and ensembles

# In[ ]:


y_t.head(10)


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Reported model accuracy depends on the random synthetic dataset used during the learning phase, which has been generated from the original dataset (used for testing) by means of the simple "data augmentation" technique. Possibility for overfitting and underfitting should be further examined, preferably with a larger dataset.</p>

# In[ ]:


import sys, IPython, platform, sklearn, scipy
print("Notebook createad on {:s} computer running {:s} and using:      \nPython {:s}\nIPython {:s}\nScikit-learn {:s}\nPandas {:s}\nNumpy {:s}\nScipy {:s}"      .format(platform.machine(), ' '.join(platform.linux_distribution()[:2]), sys.version[:5], 
              IPython.__version__, sklearn.__version__, pd.__version__, np.__version__, scipy.__version__))


# In[ ]:




