from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,BaggingRegressor
import matplotlib.pyplot as plt 
from IPython.display import Image
import pydotplus

# ---------------------- Create Training and Testing Data Sets --------------------------
f = open('Carseats.csv','r')
feature_names = [i[1:-1:] for i in f.readline()[:-1:].split(',')[1:]]
total = [i.split(',') for i in f.read().splitlines()]
f.close()
for data in total:
    data[6], data[9], data[10] = len(data[6]), len(data[9]), len(data[10])
total = np.array([float(j) for i in total for j in i]).reshape(-1,len(feature_names)+1)
data, sales = total[:,1:],total[:,0]
data_train,data_test,sales_train,sales_test = train_test_split(data,sales,test_size=0.25,random_state=1)

# -------------------------- Training Decision Tree Model ------------------------------
depths,min_nodes = range(2,len(feature_names)+1),range(4,7)
for node in min_nodes:
    err_train,err_test = [],[]
    for depth in depths:
        print('Fitting Decision Tree. Maximum Depth:',depth)
        dtr = tree.DecisionTreeRegressor(max_depth = depth,min_samples_leaf=node,random_state = 9)
        dtr.fit(data_train, sales_train)
        err_train.append(((dtr.predict(data_train)-sales_train)**2).sum())
        err_test.append(((dtr.predict(data_test)-sales_test)**2).sum())
        
        # Showing the Tree
        dot_data = tree.export_graphviz(dtr,out_file=None,feature_names = feature_names, filled=True, impurity=False, rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.write_png(f'figures/DecisionTree/tree_d{depth}_n{node}.png'))

        # Plotting Variable Importance
        fig = plt.figure()
        plt.bar(range(1,len(feature_names)+1),dtr.feature_importances_,width=0.5)
        plt.xticks(range(1,len(feature_names)+1),feature_names,rotation=25)
        plt.xlabel('Features')
        plt.ylabel('Importance Level')
        plt.title(f'Feature Importance, min_node={node}')
        plt.savefig(f'figures/DecisionTree/impor_d{depth}_n{node}.png')
        fig.clf()
        plt.close(fig)
        fig = plt.figure()
        plt.scatter(range(1,101),dtr.predict(data_test),label='prediction')
        plt.scatter(range(1,101),sales_test,label='ground_truth')
        plt.xlabel('Test Set Data Points')
        plt.ylabel('Sales Value')
        plt.title(f'Illustration of the Prediction and Ground-truth Sales Values\n Depth:{depth}, min_node:{node}')
        plt.legend()
        plt.savefig(f'figures/DecisionTree/point_d{depth}_n{node}.png')
        fig.clf()
        plt.close(fig)
    fig = plt.figure()
    plt.plot(depths,err_train,label='Train error')
    plt.plot(depths,err_test,label='Test error')
    plt.xlabel('Tree Depth')
    plt.ylabel('Sum of Squared Error')
    plt.title(f'Training and Testing Error vs Tree Depth, min_node = {node}')
    plt.legend()
    plt.savefig(f'figures/DecisionTree/err_d{depth}_n{node}.png')
    fig.clf()
    plt.close(fig)

# ------------------------ Training Bagging Regressor Model ----------------------------
depths, nums = range(2,len(feature_names)+1),range(5,51,5)
bias,variance = [],[]
for num_trees in nums:
    var = []
    err_train,err_test = [],[]
    score_train, score_test,score_false_train,score_false_test = [],[],[],[]
    predict_test = []
    for depth in depths:
        print('Fitting Bagging Model. Maximum Depth:',depth)
        brm = BaggingRegressor(n_estimators = num_trees,max_features= depth,random_state = 1,oob_score=True)
        brm_false = BaggingRegressor(n_estimators = num_trees,max_features= depth,random_state = 1,oob_score=False)
        brm.fit(data_train, sales_train)
        brm_false.fit(data_train, sales_train)

        # Recording Variable Importance
        params = np.zeros((num_trees,len(feature_names)))
        indices = {}
        i = 0
        for index in brm.estimators_features_:
            indices[f'est{i}_dep{depth}'] = index
            i += 1
        for i in range(num_trees):
            feature = brm.estimators_[i].feature_importances_
            ind = 0
            for j in indices[f'est{i}_dep{depth}']:
                params[i][j] = feature[ind]
                ind += 1
        feature_importances = np.mean(params, axis=0)
        predict_test = brm.predict(data_test)
        err_train.append(((brm.predict(data_train)-sales_train)**2).sum())
        err_test.append(((predict_test-sales_test)**2).sum())

        score_train.append(brm.oob_score_)
        score_test.append(brm.fit(data_test,sales_test).oob_score_)
        score_false_train.append(brm_false.score(data_train,sales_train))
        score_false_test.append(brm_false.score(data_test,sales_test))

        # Plotting Variable Importance
        fig = plt.figure()
        plt.bar(range(1,len(feature_importances)+1),feature_importances,width=0.5)
        plt.xticks(range(1,len(feature_names)+1),feature_names,rotation=25)
        plt.xlabel('Features')
        plt.ylabel('Importance Level')
        plt.title('Feature Importance')
        plt.savefig(f'figures/Bagging/impor_d{depth}_n{num_trees}.png')
        fig.clf()
        plt.close(fig)

        # Plotting Prediction Value and Ground-Truth Value Distribution
        fig = plt.figure()
        plt.scatter(range(1,101),brm.predict(data_test),label='prediction')
        plt.scatter(range(1,101),sales_test,label='ground_truth')
        plt.xlabel('Test Set Data Points')
        plt.ylabel('Sales Value')
        plt.title(f'Illustration of the Prediction and Ground-truth Sales Values, depth={depth}')
        plt.legend()
        plt.savefig(f'figures/Bagging/point_d{depth}_n{num_trees}.png')
        fig.clf()
        plt.close(fig)
    fig = plt.figure()
    plt.plot(depths,score_train,label='Train R^2 score with OOB data')
    plt.plot(depths,score_test,label='Test R^2 score with OOB data')
    plt.plot(depths,score_false_train,label='Train R^2 score without OOB data')
    plt.plot(depths,score_false_test,label='Test R^2 score without OOB data')
    plt.xlabel('Tree Depth')
    plt.ylabel('R^2 Score')
    plt.title(f'Training and Testing R^2 Scores vs Mtry Values, num_of_trees = {num_trees}')
    plt.legend()
    plt.savefig(f'figures/Bagging/score_n{num_trees}.png')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(depths,err_train,label='Train error')
    plt.plot(depths,err_test,label='Test error')
    plt.xlabel('Tree Depth')
    plt.ylabel('Sum of Squared Error')
    plt.title(f'Training and Testing Error vs Tree Depth, num_of_trees = {num_trees}')
    plt.legend()
    plt.savefig(f'figures/Bagging/err_d{depth}_n{num_trees}.png')
    fig.clf()
    plt.close(fig)

    for i in range(2,30):
        new_train,new_test,sales_new_train,sales_new_test = train_test_split(data,sales,test_size=0.25,random_state=i)
        new_predict = brm.predict(new_test)
        var.append((new_predict-predict_test)**2/100)
    variance.append(np.average(var))
    bias.append(np.average(predict_test-sales_test))
fig = plt.figure()
plt.plot(nums,variance,label='variance')
plt.plot(nums,bias,label='bias')
plt.xlabel('Number of Trees')
plt.ylabel('Bias/Variance Value')
plt.title('Bias and Variance Values w.r.t. Different Number of Trees')
plt.legend()
plt.savefig(f'figures/Bagging/bias_var.png')
fig.clf()
plt.close(fig)

# -------------------------- Training Random Forest Model ------------------------------
mtrys,nums = range(2,len(feature_names)+1),range(5,51,5)
bias,variance = [],[]
for num_trees in nums:
    var=[]
    err_train,err_test = [],[]
    score_train, score_test,score_false_train,score_false_test = [],[],[],[]
    for mtry in mtrys:
        print('Fitting Random Forest Model. Mtry:',mtry)
        rfr = RandomForestRegressor(min_samples_split = 3,max_features=mtry,n_estimators = num_trees,random_state=45,oob_score=True)
        rfr_false = RandomForestRegressor(min_samples_split = 3,max_features=mtry,n_estimators = num_trees,random_state=45,oob_score=False)
        rfr.fit(data_train,sales_train)
        rfr_false.fit(data_train,sales_train)
        feature_importances = np.mean([tree.feature_importances_ for tree in rfr.estimators_], axis=0)
        predict_test = rfr.predict(data_test)
        err_train.append(((rfr.predict(data_train)-sales_train)**2).sum())
        err_test.append(((predict_test-sales_test)**2).sum())
        score_train.append(rfr.oob_score_)
        score_test.append(rfr.fit(data_test,sales_test).oob_score_)
        score_false_train.append(rfr_false.score(data_train,sales_train))
        score_false_test.append(rfr_false.score(data_test,sales_test))
        fig = plt.figure()
        plt.bar(range(1,len(feature_importances)+1),feature_importances,width=0.5)
        plt.xticks(range(1,len(feature_names)+1),feature_names,rotation=25)
        plt.xlabel('Features')
        plt.ylabel('Importance Level')
        plt.title(f'Feature Importance,mtry={mtry},num_trees={num_trees}')
        plt.savefig(f'figures/RanFor/impor_m{mtry}_n{num_trees}.png')
        fig.clf()
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(range(1,101),rfr.predict(data_test),label='prediction')
        plt.scatter(range(1,101),sales_test,label='ground_truth')
        plt.xlabel('Test Set Data Points')
        plt.ylabel('Sales Value')
        plt.title(f'Illustration of the Prediction and Ground-truth Sales Values, mtry={mtry}')
        plt.legend()
        plt.savefig(f'figures/RanFor/point_m{mtry}_n{num_trees}.png')
        fig.clf()
        plt.close(fig)
    fig = plt.figure()
    plt.plot(mtrys,score_train,label='Train R^2 score with OOB data')
    plt.plot(mtrys,score_test,label='Test R^2 score with OOB data')
    plt.plot(mtrys,score_false_train,label='Train R^2 score without OOB data')
    plt.plot(mtrys,score_false_test,label='Test R^2 score without OOB data')
    plt.xlabel('Mtry Values')
    plt.ylabel('R^2 Score')
    plt.title(f'Training and Testing R^2 Scores vs Mtry Values, num_of_trees = {num_trees}')
    plt.legend()
    plt.savefig(f'figures/RanFor/score_n{num_trees}.png')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(mtrys,err_train,label='Train error with OOB data')
    plt.plot(mtrys,err_test,label='Test error with OOB data')
    plt.xlabel('Mtry Values')
    plt.ylabel('Sum of Squared Error')
    plt.title(f'Training and Testing Error vs Mtry Values, num_of_trees = {num_trees}')
    plt.legend()
    plt.savefig(f'figures/RanFor/err_n{num_trees}.png')
    fig.clf()
    plt.close(fig)
    for i in range(2,30):
        new_train,new_test,sales_new_train,sales_new_test = train_test_split(data,sales,test_size=0.25,random_state=i)
        new_predict = rfr.predict(new_test)
        var.append((new_predict-predict_test)**2/100)
    variance.append(np.average(var))
    bias.append(np.average(predict_test-sales_test))
fig = plt.figure()
plt.plot(nums,variance,label='variance')
plt.plot(nums,bias,label='bias')
plt.xlabel('Number of Trees')
plt.ylabel('Bias/Variance Value')
plt.title('Bias and Variance Values w.r.t. Different Number of Trees')
plt.legend()
plt.savefig(f'figures/RanFor/bias_var.png')
fig.clf()
plt.close(fig)

# Tuning the Hyperparameters for Random Forest Model
mtrys = range(2,len(feature_names)+1)
part = [0.3,0.5,0.7,0.9]
fig = plt.figure()
for partial in part:
    print('Tuning Random Forest Model. Sample Fraction:',partial)
    scores = []
    for mtry in mtrys:
        rfr = RandomForestRegressor(min_samples_split = 4,max_features=mtry,max_samples=partial,n_estimators = 25,random_state=45,oob_score=True)
        rfr.fit(data_train,sales_train)
        scores.append(rfr.oob_score_)
    plt.plot(mtrys,scores,label=f'fraction={partial}')
original = []
for mtry in mtrys:
    rfr = RandomForestRegressor(min_samples_split = 4,max_features=mtry,n_estimators = 25,random_state=45,oob_score=True)
    rfr.fit(data_train,sales_train)
    original.append(rfr.oob_score_)
plt.plot(mtrys,original,label='original')
plt.xlabel('Mtry Values')
plt.ylabel('OOB R^2 Score')
plt.title('OOB Score Performance w.r.t. Different Sampling Schemes')
plt.legend()
plt.savefig('figures/RanFor/hyper_sampling.png')
fig.clf()
plt.close(fig)


# Tuning the Hyperparameters for Random Forest Model
node_size = range(2,15)
oob_scores = []
for node in node_size:
    print('Tuning Random Forest Model. Node Size:',node)
    rfr = RandomForestRegressor(min_samples_split = node,max_features=10,n_estimators = 25,random_state=45,oob_score=True)
    rfr.fit(data_train,sales_train)
    oob_scores.append(rfr.oob_score_)
fig = plt.figure()
plt.plot(node_size,oob_scores,label='OOB Scores')
plt.xlabel('Least Node Size')
plt.ylabel('OOB Scores')
plt.title('OOB Scores vs Least Node Size')
plt.legend()
plt.savefig('figures/RanFor/hyper_node.png')
fig.clf()
plt.close(fig)

# ---------------------------- Training AdaBoost Model ---------------------------------
nums = range(20,131,5)
err_train,err_test = [],[]
bias,variance = [],[]
for num_weak_classifiers in nums:
    var=[]
    print('Fitting AdaBoost Model. No. of Learners:',num_weak_classifiers)
    ada = AdaBoostRegressor(n_estimators = num_weak_classifiers,loss='square',random_state=32)
    ada.fit(data_train, sales_train)
    predict_test = ada.predict(data_test)
    err_train.append(((ada.predict(data_train)-sales_train)**2).sum())
    err_test.append(((predict_test-sales_test)**2).sum())
    fig = plt.figure()
    plt.bar(range(1,len(feature_names)+1),ada.feature_importances_,width=0.5)
    plt.xticks(range(1,len(feature_names)+1),feature_names,rotation=25)
    plt.xlabel('Features')
    plt.ylabel('Importance Level')
    plt.title(f'Feature Importance, No. of Learners:{num_weak_classifiers}')
    plt.savefig(f'figures/AdaBoost/impor_n{num_weak_classifiers}.png')
    fig.clf()
    plt.close(fig)
    fig = plt.figure()
    plt.scatter(range(1,101),ada.predict(data_test),label='prediction')
    plt.scatter(range(1,101),sales_test,label='ground_truth')
    plt.xlabel('Test Set Data Points')
    plt.ylabel('Sales Value')
    plt.title(f'Illustration of the Prediction and Ground-truth Sales Values, No. of Learners:{num_weak_classifiers}')
    plt.legend()
    plt.savefig(f'figures/AdaBoost/point_n{num_weak_classifiers}.png')
    fig.clf()
    plt.close(fig)
    for i in range(2,30):
        new_train,new_test,sales_new_train,sales_new_test = train_test_split(data,sales,test_size=0.25,random_state=i)
        new_predict = ada.predict(new_test)
        var.append((new_predict-predict_test)**2/100)
    variance.append(np.average(var))
    bias.append(np.average(predict_test-sales_test))

fig = plt.figure()
plt.plot(nums,variance,label='variance')
plt.plot(nums,bias,label='bias')
plt.xlabel('Number of Trees')
plt.ylabel('Bias/Variance Value')
plt.title('Bias and Variance Values w.r.t. Different Number of Trees')
plt.legend()
plt.savefig(f'figures/AdaBoost/bias_var.png')
fig.clf()
plt.close(fig)

fig = plt.figure()
plt.plot(nums,err_train,label='Train error')
plt.plot(nums,err_test,label='Test error')
plt.xlabel('No. of Weak Classifiers')
plt.ylabel('Sum of Squared Error')
plt.title(f'Training and Testing Error vs No. of Classifiers')
plt.legend()
plt.savefig(f'figures/AdaBoost/err.png')
fig.clf()
plt.close(fig)
