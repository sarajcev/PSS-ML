#!/usr/bin/env python
# coding: utf-8

# # Machine learning for Power System Stability Analysis

# <p style="background-color:azure;padding:10px;border:2px solid lightsteelblue"><b>Author:</b> Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# <br>
# University of Split, FESB, Department of Power Engineering <br>R. Boskovica 32, HR-21000 Split, Croatia, EU.</p>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from scipy import stats


# In[3]:


from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


# In[4]:


import keras


# In[5]:


import warnings


# In[7]:


warnings.filterwarnings(action='ignore', category=FutureWarning)


# In[8]:


# Inline figures
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# Figure aesthetics
sns.set(context='notebook', style='white', font_scale=1.1)
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})


# In[10]:


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

# In[11]:


data = pd.read_csv('GridDictionary.csv')
data.head()


# In[9]:


#print(data.columns.values)


# In[12]:


# Percentage of "ones" in the "Stability" column
print('There is {:.1f}% of unstable cases in the dataset!'.format(data['Stability'].sum()/float(len(data['Stability']))*100.))


# ### Select a random subset of the original data

# In[11]:


# Select a random subset of the original dataset (without replacement)
#SUBSET_SIZE = 2000
#random_idx = np.random.choice(data.index, size=SUBSET_SIZE, replace=False)
#data = data.iloc[random_idx]


# ### Data preprocessing and splitting

# In[13]:


# Training dataset
no_features = len(data.columns) - 1
X_data = data.iloc[:,0:no_features]  # features
print('X_data', X_data.shape)
y_data = data['Stability']
print('y_data', y_data.shape)


# In[14]:


# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, shuffle=True)


# In[15]:


print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[16]:


print('Unstable cases in training dataset: {:.1f}%:'.format(np.sum(y_train)/float(len(y_train))*100.))
print('Unstable cases in testing dataset {:.1f}%:'.format(np.sum(y_test)/float(len(y_test))*100.))


# In[17]:


print('Unstable cases in training dataset: {:.1f}%:'.format(np.sum(y_train)/float(len(y_train))*100.))
print('Unstable cases in testing dataset {:.1f}%:'.format(np.sum(y_test)/float(len(y_test))*100.))


# In[18]:


y_t = data[['Stability']].copy()
idx = y_test.index.values
y_t = y_t.loc[idx]
y_t.shape


# #### StandardScaler

# In[19]:


# Standardize the input data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### LogisticRegression

# In[20]:


# Grid-search with cross validation for optimal model hyper-parameters
parameters = {'C':[1., 10., 50., 100., 500., 1000.]}
lreg = GridSearchCV(estimator=LR(multi_class='auto', solver='newton-cg', 
                                 class_weight='balanced'),  # class balancing
                    param_grid=parameters, cv=3, scoring='f1',  # notice the "scoring" method!
                    refit=True, n_jobs=-1, iid=False)
# In this case GridSearchCV uses StratifiedKFold (with cv=3 folds) when
# the estimator is a classifier and y is either binary or multiclass!
lreg.fit(X_train, y_train)
# Best value of hyper-parameter "C"
best_c = lreg.best_params_['C']
print('Best value: C = {:g}'.format(best_c))


# In[21]:


# Average classification accuracy with cross validation
scores = cross_val_score(lreg, X_train, y_train, cv=3, scoring='f1')  # it does not return a model!
print('Score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[22]:


pred = lreg.predict(X_test)
labels = ['Stable', 'Unstable']
# confusion matrix
scores_image = heatmap(metrics.confusion_matrix(y_test, pred), xlabel='Predicted label', 
                       ylabel='True label', xticklabels=labels, yticklabels=labels, 
                       cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()
plt.show()


# In[23]:


# classification report
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[24]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns=['Values'])


# ### Principal components analysis

# In[25]:


# Let's first reduce all features in the dataset down to 3 principal components.
pca = PCA(n_components=3)
X3_train = pca.fit_transform(X_train)
X3_test = pca.transform(X_test)
idx_stable = y_test==0


# In[26]:


# Let's see what it looks like
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax0 = fig.add_subplot(221)
ax0.scatter(X3_test[idx_stable,1], X3_test[idx_stable,2], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax0.scatter(X3_test[~idx_stable,1], X3_test[~idx_stable,2], 
              s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax0.legend(loc='upper right')
ax0.set_xlabel('Second principal component')
ax0.set_ylabel('Third principal component')
ax0.grid()
ax1 = fig.add_subplot(222, projection='3d')
ax1.scatter(X3_test[idx_stable,0], X3_test[idx_stable,1], X3_test[idx_stable,2], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax1.scatter(X3_test[~idx_stable,0], X3_test[~idx_stable,1], X3_test[~idx_stable,2],
              s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax1.legend(loc='upper right')
ax1.set_xlabel('1st comp.')
ax1.set_ylabel('2nd comp.')
#ax1.set_zlabel('3rd comp.')
ax0 = fig.add_subplot(223)
ax0.scatter(X3_test[idx_stable,0], X3_test[idx_stable,1], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax0.scatter(X3_test[~idx_stable,0], X3_test[~idx_stable,1], 
              s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax0.legend(loc='upper right')
ax0.set_xlabel('First principal component')
ax0.set_ylabel('Second principal component')
ax0.grid()
ax0 = fig.add_subplot(224)
ax0.scatter(X3_test[idx_stable,0], X3_test[idx_stable,2], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax0.scatter(X3_test[~idx_stable,0], X3_test[~idx_stable,2], 
              s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax0.legend(loc='upper right')
ax0.set_xlabel('First principal component')
ax0.set_ylabel('Third principal component')
ax0.grid()
fig.tight_layout()
plt.savefig('PCA-3D.png', dpi=600)
plt.show()


# In[27]:


# Average LR accuracy with only three principal components as features
scores = cross_val_score(LR(C=best_c, multi_class='ovr', solver='newton-cg', 
                            class_weight='balanced'), 
                         X3_train, y_train, cv=3, scoring='f1')
print('Score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# ### Autoencoder

# In[28]:


# Reducing "no_features" to 2D space with autoencoder
input_layer = keras.layers.Input(shape=(no_features,))
# "encoded" is the encoded representation of the input
x = keras.layers.Dense(128, activation='relu', 
                       kernel_initializer='glorot_normal')(input_layer)
x = keras.layers.Dense(64, activation='relu', 
                       kernel_initializer='glorot_normal')(x)
encoded = keras.layers.Dense(2, activation='relu', 
                             kernel_initializer='glorot_normal')(x)
# "decoded" is the lossy reconstruction of the input
x = keras.layers.Dense(64, activation='relu', 
                       kernel_initializer='glorot_normal')(encoded)
x = keras.layers.Dense(128, activation='relu', 
                       kernel_initializer='glorot_normal')(x)
decoded = keras.layers.Dense(no_features, activation='linear', 
                             kernel_initializer='glorot_normal')(x)
# Autoencoder
autoencoder = keras.models.Model(input_layer, decoded)
# Encoder
encoder = keras.models.Model(input_layer, encoded)
# Compile the model
autoencoder.compile(optimizer='adadelta', loss='kullback_leibler_divergence')
# Define early stopping criteria
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, 
                          shuffle=True, validation_split=0.1, 
                          callbacks=[early_stopping], verbose=0)


# In[29]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# In[30]:


encoded_data = encoder.predict(X_test)


# In[31]:


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(encoded_data[idx_stable,0], encoded_data[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(encoded_data[~idx_stable,0], encoded_data[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='best')
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.grid()
fig.tight_layout()
plt.savefig('Autoencode2D.png', dpi=600)
plt.show()


# ### Dimensionality reduction using truncated SVD

# In[32]:


svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_test)


# In[33]:


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X_svd[idx_stable,0], X_svd[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X_svd[~idx_stable,0], X_svd[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='best')
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.grid()
fig.tight_layout()
plt.show()


# ### t-distributed Stochastic Neighbor Embedding

# In[34]:


tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X_test)


# In[35]:


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X_embedded[idx_stable,0], X_embedded[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X_embedded[~idx_stable,0], X_embedded[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='best')
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.grid()
fig.tight_layout()
plt.show()


# #### Feature selection with Pipeline and GridSearch

# In[30]:


# Optimize the number of features and the classifier's hyper-parameters 
# at the same time, using pipline and grid search with cross-validation
pca = PCA()  # do NOT set "n_components" here!
logreg = LR(multi_class='ovr', solver='newton-cg', class_weight='balanced') 
pipe = Pipeline([('pca',pca), ('logreg',logreg)])
param_grid = {'pca__n_components': [10, 20, 30, 50, 100],   # PCA
              'logreg__C': [1., 10., 50., 100.]}      # LogisticRegression
grid_pipe = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, 
                         scoring='f1', refit=True, n_jobs=-1, iid=False)
grid_pipe.fit(X_train, y_train)
print('Best parameter (CV score = {:0.3f}):'.format(grid_pipe.best_score_))
print(grid_pipe.best_params_)


# In[31]:


# Predict probability on test data
y_lr = grid_pipe.predict_proba(X_test)
y_t['logreg'] = y_lr.argmax(axis=1)


# ### Support Vector Machine

# In[32]:


parameters ={'C':[1., 10., 100., 500., 1000.],
             'gamma':[0.0001, 0.001, 0.01, 0.1, 1.]}
svc = GridSearchCV(estimator=svm.SVC(kernel='rbf', probability=True, 
                                     class_weight='balanced'),  # class balancing
                   param_grid=parameters, cv=3,
                   scoring='f1', refit=True, n_jobs=-1, iid=False)
svc.fit(X_train, y_train)


# In[33]:


# Best model parameters
best_parameters = svc.best_params_
print("Best parameters from GridSearch: {}".format(svc.best_params_))


# In[34]:


scores = cross_val_score(svc, X_train, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[35]:


results = pd.DataFrame(svc.cv_results_)
scores = np.array(results.mean_test_score).reshape(len(parameters['C']), len(parameters['gamma']))


# In[36]:


fig, ax = plt.subplots(figsize=(5,5))
heatmap(scores, xlabel='gamma', xticklabels=parameters['gamma'], 
        ylabel='C', yticklabels=parameters['C'], cmap="viridis", ax=ax)
plt.show()


# #### RandomizedSearchCV

# In[37]:


parameters = {'C':stats.expon(scale=100), 'gamma':stats.expon(scale=.1)}
svc2 = RandomizedSearchCV(estimator=svm.SVC(kernel='rbf', probability=True, 
                                            class_weight='balanced'), 
                          param_distributions=parameters, cv=3, n_iter=50,  # 50 iterations!
                          scoring='f1',  # notice the scoring method!
                          refit=True, n_jobs=-1, iid=False)
svc2.fit(X_train, y_train)


# In[38]:


# Best model parameters
best_parameters = svc2.best_params_
print("Best parameters from RandomSearch: {}".format(svc2.best_params_))


# In[39]:


scores = cross_val_score(svc2, X_train, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[40]:


# classification report
pred = svc2.predict(X_test)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[41]:


y_svc2 = svc2.predict_proba(X_test)
y_t['svc'] = y_svc2.argmax(axis=1)


# #### Precision-Recall Tradeoff

# In[42]:


y_probas = cross_val_predict(svm.SVC(**best_parameters, probability=True, class_weight='balanced'), 
                             X_train, y_train, cv=3, method='predict_proba')


# In[43]:


y_scores = y_probas[:,1]  # score == probability of positive class
precisions, recalls, thresholds = metrics.precision_recall_curve(y_train, y_scores)


# In[44]:


fig, ax = plt.subplots(figsize=(6,4))
#ax.set_title('SVM Precision-Recall tradeof')
ax.plot(thresholds, precisions[:-1], lw=2, label='Precision')
ax.plot(thresholds, recalls[:-1], lw=2, label='Recall')
plt.vlines(0.5, 0, 1, linestyles='--', label='Threshold = 0.5')
ax.set_xlabel('Thresholds')
ax.legend(loc='best')
ax.set_ylim(bottom=0.85, top=1.02)
ax.grid()
fig.tight_layout()
plt.savefig('SVM-Precision-Recall.png', dpi=600)
plt.show()


# In[45]:


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


# In[46]:


# Average precision-recall score
y_test_score = svc2.predict_proba(X_test)[:,1]
average_precision = metrics.average_precision_score(y_test, y_test_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# In[48]:


# Determine a class from the predicted probability by using 
# the user-specified threshold value (not a default of 0.5)
THRESHOLD = 0.52  # Example
preds = np.where(y_test_score > THRESHOLD, 1, 0)


# In[49]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, preds), metrics.recall_score(y_test, preds),
                   metrics.precision_score(y_test, preds), metrics.roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns=['Values'])


# ### ExtraTreesClassifier

# In[50]:


# ExtraTreesClassifier (ensemble learner) with grid search 
# and cross-validation for hyper-parameters optimisation
parameters = {'n_estimators':[5, 10, 15, 20], 
              'criterion':['gini', 'entropy'], 
              'max_depth':[2, 5, None]}
trees = GridSearchCV(estimator=ExtraTreesClassifier(class_weight='balanced'), param_grid=parameters, 
                     cv=3, scoring='f1', refit=True, n_jobs=-1, iid=False) 
trees.fit(X_train, y_train)


# In[51]:


# Best model parameters
best_parameters = trees.best_params_
print("Best parameters: {}".format(trees.best_params_))


# In[52]:


scores = cross_val_score(trees, X_train, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[53]:


# classification report
pred = trees.predict(X_test)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[54]:


y_trees = trees.predict_proba(X_test)
y_t['tree'] = y_trees.argmax(axis=1)


# #### Feature importance analysis

# In[55]:


trees = ExtraTreesClassifier(**best_parameters, class_weight='balanced')
trees.fit(X_train, y_train)
trees_feature_importance = trees.feature_importances_
trees_feature_importance = 100.0 * (trees_feature_importance / trees_feature_importance.max())
sorted_idx_trees = np.argsort(trees_feature_importance)
position = np.arange(sorted_idx_trees.shape[0]) + .5


# In[56]:


# Select top features considering their relative importance
# Top features are those above some preselect threshold of relative importance
IMPORTANCE = 20.  # relative importance threshold
TOPT = np.sum(trees_feature_importance[sorted_idx_trees] > IMPORTANCE)
print(TOPT)
top_features_trees = data.columns.values[sorted_idx_trees][-TOPT:][::-1]
print('Most relevant {:d} features according to the ExtraTreesClassifier:'.format(TOPT))
print(top_features_trees)


# In[57]:


# Plot relative feature importance
fig, ax = plt.subplots(figsize=(5,7))
ax.barh(position[-TOPT:], trees_feature_importance[sorted_idx_trees][-TOPT:], 
        align='center', color='navy', alpha=0.6)
plt.yticks(position[-TOPT:], data.columns[sorted_idx_trees][-TOPT:])
ax.set_xlabel('Feature Relative Importance')
ax.grid(which='major', axis='x')
plt.tight_layout()
plt.show()


# ### RandomForest classifier (ensemble learner)

# In[58]:


# RandomForestClassifier (ensemble learner for classification)
parameters = {'n_estimators':[10, 15, 20], 
              'criterion':['gini', 'entropy'],
              'max_features':[4, 'auto'],
              'max_depth':[2, None]}
# grid search and cross-validation for hyper-parameters optimisation
forest = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced'), param_grid=parameters, 
                      cv=3, scoring='f1', refit=True, n_jobs=-1, iid=False) 
forest.fit(X_train, y_train)


# In[59]:


best_forest_parameters = forest.best_params_
print("Best parameters: {}".format(forest.best_params_))


# In[60]:


scores = cross_val_score(forest, X_train, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[61]:


# classification report
pred = forest.predict(X_test)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[62]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns=['Values'])


# In[63]:


y_forest = forest.predict_proba(X_test)
y_t['forest'] = y_forest.argmax(axis=1)


# In[64]:


forest_top = RandomForestClassifier(**best_forest_parameters, class_weight='balanced')
forest_top.fit(X_train, y_train)
forest_feature_importance = forest_top.feature_importances_
forest_feature_importance = 100.0 * (forest_feature_importance / forest_feature_importance.max())
sorted_idx_forest = np.argsort(forest_feature_importance)
position_forest = np.arange(sorted_idx_forest.shape[0]) + .5


# In[65]:


FOPT = np.sum(forest_feature_importance[sorted_idx_forest] > IMPORTANCE)
top_features_forest = data.columns.values[sorted_idx_forest][-FOPT:][::-1]
print('Most relevant {:d} features according to the RandomForestClassifier:'.format(FOPT))
print(top_features_forest)


# In[66]:


# Plot relative feature importance
fig, ax = plt.subplots(figsize=(5,7))
ax.barh(position_forest[-FOPT:], forest_feature_importance[sorted_idx_forest][-FOPT:], 
        align='center', color='navy', alpha=0.6)
plt.yticks(position_forest[-FOPT:], data.columns[sorted_idx_forest][-FOPT:])
ax.set_xlabel('Feature Relative Importance')
ax.grid(which='major', axis='x')
plt.tight_layout()
plt.savefig('Forest-Features.png', dpi=600)
plt.show()


# In[67]:


duplicate_features = set(top_features_trees).intersection(set(top_features_forest))
print(duplicate_features)


# ### GradientBoosting classifier with feature importance analysis

# In[68]:


# Train & evaluate model performance
def train_and_evaluate(model, X, y, ns=3):
    # k-fold cross validation iterator 
    cv = StratifiedKFold(n_splits=ns, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')  # scoring='f1'
    print('Average score using {:d}-fold CV: {:g} +/- {:g}'.format(ns, np.mean(scores), np.std(scores)))


# In[69]:


# Gradient Boosting Classifier
clf_gb = GradientBoostingClassifier()
train_and_evaluate(clf_gb, X_train, y_train, 3)
clf_gb.fit(X_train, y_train)


# In[70]:


# Feature importance
feature_importance = clf_gb.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5


# In[71]:


# Select top features
TOP = 10
top_features = data.columns.values[sorted_idx][-TOP:][::-1]
print('Most relevant {:d} features according to the GradientBoostingClassifier:'.format(TOP))
print(top_features)


# In[72]:


# Plot relative feature importance
fig, ax = plt.subplots(figsize=(5,5))
ax.barh(pos[-TOP:], feature_importance[sorted_idx][-TOP:], align='center', color='magenta', alpha=0.6)
plt.yticks(pos[-TOP:], data.columns[sorted_idx][-TOP:])
ax.set_xlabel('Feature Relative Importance')
ax.grid(which='major', axis='x')
plt.tight_layout()
plt.show()


# In[73]:


print(set(top_features_trees).intersection(set(top_features)))


# In[74]:


# Correlation matrix of selected features
pearson = data[data.columns[sorted_idx][-TOP:]].corr('pearson')
pearson.iloc[-1][:-1].sort_values()
# Correlation matrix as heatmap (seaborn)
fig, ax = plt.subplots(figsize=(6.5,5.5))
sns.heatmap(pearson, annot=True, annot_kws=dict(size=9), vmin=-1, vmax=1, ax=ax)
#plt.tight_layout()
plt.show()


# In[75]:


# classification report
pred = clf_gb.predict(X_test)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[76]:


# Predict on new data
y_gb = clf_gb.predict_proba(X_test)
y_t['gbr'] = y_gb.argmax(axis=1)


# #### Re-train SVM using only top features from the GradientBoosting classifier

# In[77]:


# IMPORTANT: NTOP <= TOP
NTOP = 2  # using only top 2 features!


# In[78]:


top_features_index = []
for name in top_features:
    top_features_index.append(data.columns.get_loc(name))


# In[79]:


X_train_best = X_train[:,top_features_index[:NTOP]]
X_test_best = X_test[:,top_features_index[:NTOP]]
print(X_train_best.shape)
print(X_test_best.shape)


# In[80]:


# Optimize SVM with only TOP features
parameters = {'C':stats.expon(scale=100), 'gamma':stats.expon(scale=.1)}
svc_top = RandomizedSearchCV(estimator=svm.SVC(kernel='rbf', probability=True, class_weight='balanced'), 
                             param_distributions=parameters, cv=3, n_iter=100,  # 100 iterations!
                             scoring='f1',  # notice the scoring method!
                             refit=True, n_jobs=-1, iid=False)
svc_top.fit(X_train_best, y_train)


# In[81]:


# Best model parameters
best_parameters = svc_top.best_params_
print("Best parameters from RandomSearch: {}".format(svc_top.best_params_))


# In[82]:


scores = cross_val_score(svc_top, X_train_best, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[153]:


pred = svc_top.predict(X_test_best)
pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred),
                   metrics.f1_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", 'f1-score'], columns=['Values'])


# #### Graphical visualization of the top two features

# In[83]:


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# In[84]:


idx_stable = y_test==0


# In[85]:


# Ploting data without standard scaler transformer
fig, ax = plt.subplots(figsize=(5.5,5.5))
ax.scatter(X_test_best[idx_stable,0], X_test_best[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X_test_best[~idx_stable,0], X_test_best[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='upper left')
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_xlim(-5,1)
ax.set_ylim(-6,1)
ax.grid()
# Inset figure with a zoomed-in region of interest
axins = zoomed_inset_axes(ax, zoom=3, loc='lower left')
axins.scatter(X_test_best[idx_stable,0], X_test_best[idx_stable,1], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.25, label='Stable')
axins.scatter(X_test_best[~idx_stable,0], X_test_best[~idx_stable,1], 
              s=30, c='red', marker='o', edgecolors='k', alpha=0.25, label='Unstable')
axins.set_xlim(-1,0.1)
axins.set_ylim(-0.6,0.5)
plt.xticks(visible=False)
plt.yticks(visible=False)
# draw a box of the region of the inset axes in the parent axes 
# and connecting lines between the box and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='dimgrey')
plt.show()


# #### Plot decision region for test samples with only top two features (NTOP = 2)

# In[86]:


# Axis grid with NTOP = 2
h = 0.1; delta = 0.01
x_min, x_max = X_test_best[:,0].min() - h, X_test_best[:,0].max() + h
y_min, y_max = X_test_best[:,1].min() - h, X_test_best[:,1].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, delta), np.arange(y_min, y_max, delta))
Z = svc_top.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
Z = Z.reshape(xx.shape)


# In[87]:


fig, ax = plt.subplots(figsize=(6,5))
ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
ax.scatter(X_test_best[idx_stable,0], X_test_best[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X_test_best[~idx_stable,0], X_test_best[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='upper left')
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid()
# Inset figure with a zoomed-in region of interest
axins = zoomed_inset_axes(ax, zoom=3, loc='lower left')
axins.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
axins.scatter(X_test_best[idx_stable,0], X_test_best[idx_stable,1], 
              s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
axins.scatter(X_test_best[~idx_stable,0], X_test_best[~idx_stable,1], 
              s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
axins.set_xlim(-1,0.1)
axins.set_ylim(-0.6,0.5)
plt.xticks(visible=False)
plt.yticks(visible=False)
# draw a box of the region of the inset axes in the parent axes 
# and connecting lines between the box and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='dimgrey')
plt.savefig('SVM-TOP2Features.png', dpi=600)
plt.show()


# ### Principal components with best features from different learners

# #### Unifying best features from different classifiers

# In[88]:


print('Most relevant {:d} features according to the ExtraTreesClassifier:'.format(TOPT))
print(top_features_trees)
print('Most relevant {:d} features according to the RandomForestClassifier:'.format(FOPT))
print(top_features_forest)
top_features_all = set(top_features_trees).union(set(top_features_forest))
print('Union of most relevant features {:d}:'.format(len(top_features_all)))
print(top_features_all)
features_duplex = set(top_features_trees).intersection(set(top_features_forest))
print('Features selected by both classifiers {:d}:'.format(len(features_duplex)))
print(features_duplex)


# In[89]:


top_features_index = []
for name in top_features_all:
    top_features_index.append(data.columns.get_loc(name))


# In[155]:


# Projecting best features using only two principal components
pca = PCA(n_components=2, whiten=True)
X2_train = pca.fit_transform(X_train[:,top_features_index])
X2_test = pca.transform(X_test[:,top_features_index])


# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> The union of best (unique) features selected previously by the ExtraTrees and RandomForest classifiers are reduced down to the two principal components. This essentially means that the multi-dimensional space of original features is projected into the 2D space of principal components. Now, one can easily visualize test cases and novel decision boundary in this new 2D coordinate system of principal components.</p>

# In[156]:


fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(X2_test[idx_stable,0], X2_test[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X2_test[~idx_stable,0], X2_test[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='upper left')
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.grid()
fig.tight_layout()
plt.show()


# #### Train SVM classifier using principal components

# In[157]:


# Optimize SVM with only TOP features
parameters = {'C':stats.expon(scale=100), 'gamma':stats.expon(scale=.1)}
svc_pca = RandomizedSearchCV(estimator=svm.SVC(kernel='rbf', probability=True, class_weight='balanced'), 
                             param_distributions=parameters, cv=3, n_iter=50,  # iterations!
                             scoring='f1',  # notice the scoring method!
                             refit=True, n_jobs=-1, iid=False)
svc_pca.fit(X2_train, y_train)


# In[158]:


scores = cross_val_score(svc_pca, X2_train, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[159]:


pred = svc_pca.predict(X2_test)
pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred),
                   metrics.f1_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", 'f1-score'], columns=['Values'])


# In[160]:


h = 0.1; delta = 0.01
x_min, x_max = X2_test[:,0].min() - h, X2_test[:,0].max() + h
y_min, y_max = X2_test[:,1].min() - h, X2_test[:,1].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, delta), np.arange(y_min, y_max, delta))
Z = svc_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
Z = Z.reshape(xx.shape)


# In[161]:


fig, ax = plt.subplots(figsize=(6,5))
ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
ax.scatter(X2_test[idx_stable,0], X2_test[idx_stable,1], 
           s=30, c='green', marker='o', edgecolors='k', alpha=0.5, label='Stable')
ax.scatter(X2_test[~idx_stable,0], X2_test[~idx_stable,1], 
           s=30, c='red', marker='o', edgecolors='k', alpha=0.5, label='Unstable')
ax.legend(loc='upper left')
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.grid()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
fig.tight_layout()
plt.savefig('SVM-PCA2D-BestFeatures.png', dpi=600)
plt.show()


# ## Ensemble models using voting principle

# <p style="background-color:honeydew;padding:10px;border:2px solid mediumseagreen"><b>Note:</b> Ensembling consists of pooling together the predictions of a set of different models, to produce better predictions. The key to making ensembling work is the diversity of the set of classifiers. Diversity is what makes ensembling work. For this reason, one should ensemble models that are as good as possible while being <b>as different as possible</b>. This typically means using very different network architectures or even different brands of machine-learning approaches. This is exactly what has been proposed here.</p>

# ### Soft voting

# In[96]:


clf = VotingClassifier(estimators=[('logreg', lreg),     # LogisticRegression
                                   ('svm', svc2),        # SVC
                                   ('forest', forest)],  # RandomForest 
                       weights=[1, 2, 1],  # classifier relative weights
                       voting='soft', n_jobs=-1)
clf.fit(X_train, y_train);  # train with a full set of features


# In[97]:


scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)  # scoring='f1'!
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[98]:


# confusion matrix
scores_image = heatmap(metrics.confusion_matrix(y_test, clf.predict(X_test)), 
                       xlabel='Predicted label', ylabel='True label', 
                       xticklabels=labels, yticklabels=labels, 
                       cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()
plt.show()


# In[99]:


# classification report
pred = clf.predict(X_test)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[100]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns=['Values'])


# In[101]:


y_clf = clf.predict_proba(X_test)
y_t['vote'] = y_clf.argmax(axis=1)


# #### Voting by training classifiers with only best features

# In[102]:


X_train_best_features = X_train[:,top_features_index]
X_test_best_features = X_test[:,top_features_index]
X_train_best_features.shape


# In[103]:


lreg.fit(X_train_best_features, y_train);


# In[104]:


scores = cross_val_score(lreg, X_train_best_features, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[145]:


pred = lreg.predict(X_test_best_features)
pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred),
                   metrics.f1_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1-score"], columns=['Values'])


# In[107]:


svc2.fit(X_train_best_features, y_train);


# In[108]:


scores = cross_val_score(svc2, X_train_best_features, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[146]:


pred = svc2.predict(X_test_best_features)
pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred),
                   metrics.f1_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", 'f1-score'], columns=['Values'])


# In[110]:


forest.fit(X_train_best_features, y_train);


# In[111]:


scores = cross_val_score(forest, X_train_best_features, y_train, cv=3, scoring='f1')
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[147]:


pred = forest.predict(X_test_best_features)
pd.DataFrame(data=[metrics.accuracy_score(y_test, pred), metrics.recall_score(y_test, pred),
                   metrics.precision_score(y_test, pred), metrics.roc_auc_score(y_test, pred),
                   metrics.f1_score(y_test, pred)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1-score"], columns=['Values'])


# In[113]:


clf2 = VotingClassifier(estimators=[('logreg', lreg),   # LogisticRegression
                                    ('svm', svc2),      # SVC
                                    ('forest', forest)],  # RandomForest 
                        weights=[1, 2, 1],  # classifier relative weights
                        voting='soft', n_jobs=-1)
clf2.fit(X_train_best_features, y_train);  # train with selected features


# In[114]:


scores = cross_val_score(clf2, X_train_best_features, y_train, cv=3, scoring='f1', n_jobs=-1)  # scoring='f1'!
print('Average score using 3-fold CV: {:g} +/- {:g}'.format(np.mean(scores), np.std(scores)))


# In[115]:


# classification report
pred = clf2.predict(X_test_best_features)
print(metrics.classification_report(y_test, pred, target_names=labels))


# In[163]:


# confusion matrix
scores_image = heatmap(metrics.confusion_matrix(y_test, clf2.predict(X_test_best_features)), 
                       xlabel='Predicted label', ylabel='True label', 
                       xticklabels=labels, yticklabels=labels, 
                       cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Confusion-Matrix.png', dpi=600)
plt.show()


# #### Final Calibration

# In[117]:


y_test_probas = clf2.predict_proba(X_test_best_features)
y_test_scores = y_test_probas[:,1]  # score == probability of positive class
precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, y_test_scores)


# In[144]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Ensemble Precision-Recall tradeof')
ax.plot(thresholds, precisions[:-1], lw=2, label='Precision')
ax.plot(thresholds, recalls[:-1], lw=2, label='Recall')
#plt.vlines(0.5, 0, 1, linestyles='--', label='Threshold = 0.5')
ax.set_xlabel('Thresholds')
ax.legend(loc='best')
ax.set_ylim(bottom=0.8, top=1.02)
ax.grid()
fig.tight_layout()
plt.savefig('Calibrate.png', dpi=600)
plt.show()


# In[119]:


print('Precision: {:.5f}'.format(precisions[np.where(thresholds >= 0.5)[0][0]]))
print('Recall: {:.5f}'.format(recalls[np.where(thresholds >= 0.5)[0][0]]))


# In[120]:


# Determine a class from the predicted probability by using 
# the user-specified threshold value (not a default of 0.5)
THRESHOLD = 0.65  # fine-tuning
preds = np.where(y_test_scores > THRESHOLD, 1, 0)


# In[121]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, preds), metrics.recall_score(y_test, preds),
                   metrics.precision_score(y_test, preds), metrics.roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"], columns=['Values'])


# In[129]:


clf3 = VotingClassifier(estimators=[('logreg', lreg),   # LogisticRegression
                                    ('svm', svc2),      # SVC
                                    ('forest', forest)],  # RandomForest 
                        weights=[1, 2, 3],  # classifier relative weights
                        voting='soft', n_jobs=-1)
clf3.fit(X_train_best_features, y_train);  # train with selected features


# In[149]:


y_test_probas = clf3.predict_proba(X_test_best_features)
y_test_scores = y_test_probas[:,1]  # score == probability of positive class
precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, y_test_scores)


# In[150]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Ensemble Precision-Recall tradeof')
ax.plot(thresholds, precisions[:-1], lw=2, label='Precision')
ax.plot(thresholds, recalls[:-1], lw=2, label='Recall')
plt.vlines(0.5, 0, 1, linestyles='--', label='Threshold = 0.5')
ax.set_xlabel('Thresholds')
ax.legend(loc='best')
ax.set_ylim(bottom=0.8, top=1.02)
ax.grid()
fig.tight_layout()
plt.savefig('Calibrate-VE.png', dpi=600)
plt.show()


# In[151]:


# Determine a class from the predicted probability by using 
# the user-specified threshold value (not a default of 0.5)
THRESHOLD = 0.5  # fine-tuning
preds = np.where(y_test_scores > THRESHOLD, 1, 0)


# In[152]:


pd.DataFrame(data=[metrics.accuracy_score(y_test, preds), metrics.recall_score(y_test, preds),
                   metrics.precision_score(y_test, preds), metrics.roc_auc_score(y_test, preds),
                   metrics.f1_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1-score"], columns=['Values'])


# #### Computing environment

# In[122]:


import sys, IPython, sklearn, scipy, matplotlib
print("Notebook createad with:      \nPython {:s}\nIPython {:s}\nScikit-learn {:s}\nPandas {:s}\nNumpy {:s}\nScipy {:s}\nMatplotlib {:s}"      .format(sys.version[:5], IPython.__version__, sklearn.__version__, pd.__version__, np.__version__, 
              scipy.__version__, matplotlib.__version__))


# In[ ]:




