import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import csv


def sigmoid(theta,x): # sigmoid function
    return 1/(1+np.exp(-x@theta))

def gradient_descent(data_train, y_train, step,tol,choice,i,lamb):
    theta = np.zeros((data_train.shape[1],1)) # define an arbitrary initial value of theta
    iter_num = 1
    if choice == 0:
        print(f'---- Performing {i+1}-th Standard Logistic Regression... ----')
    elif choice == 1:
        print(f'---- Performing {i+1}-th L2 Logistic Regression... ----')
    print('-------------------------------------------------------')
    cost,grad = models(data_train,y_train,choice,theta,lamb)
    theta_new = theta
    theta_old = theta_new
    theta_new = theta_old - step*grad
    diff = np.abs(theta_new-theta_old)
    y=[]
    y.append(cost[0])

    # Check the updates
    #print('Iteration:',str(iter_num)+", Grad_diff:", str(np.dot(diff.transpose(),diff)[0][0])+', Cost:', str(cost[0]))

    while np.dot(diff.transpose(),diff) > tol: # Using change of norm of gradient as stopping criteria
    # while cost[0] > 0.27: # Using cost function value as stopping criteria
        cost, grad = models(data_train,y_train,choice, theta_new, lamb)
        theta_old = theta_new
        theta_new = theta_old - step*grad # update the parameter
        iter_num += 1
        diff = np.abs(theta_new-theta_old)
        y.append(cost[0])

        # Check the updates
        # print('Iteration:',str(iter_num)+", Grad_diff:", str(np.dot(diff.transpose(),diff)[0][0])+', Cost:', str(cost[0]))
    print()
    print()

    # Print the cost function decline with respect to iterations
    #x = range(iter_num)
    #plt.plot(x,y,label=f'{chr(945)} = {step}')

    return theta_new.transpose()

def models(data_train, y_train, choice, theta, lamb):
    M = data_train.shape[0]
    if choice == 0: # Standard Logistic Regression
        cost = -(y_train@np.log(sigmoid(theta,data_train))+(1-y_train)@np.log(1-sigmoid(theta,data_train)))/M
        grad = data_train.transpose()@(sigmoid(theta,data_train)-y_train.reshape(-1,1))/M
        return np.array(cost), grad
    elif choice == 1: # L2 logistic regression
        cost = -(y_train@np.log(sigmoid(theta,data_train))+(1-y_train)@np.log(1-sigmoid(theta,data_train)))/M+pd.DataFrame(lamb*theta.transpose()@theta/(2*M))
        grad = data_train.transpose()@(sigmoid(theta,data_train)-y_train.reshape(-1,1))/M+lamb*theta/M
        return np.array(cost)[0], grad


def validate(data_test, y_test, theta,i,alpha): # cross validation
    response = sigmoid(theta.transpose(),data_test)
    predict = np.floor(response*2)
    error = abs(np.array(predict)-y_test).reshape(-1,1)
    
    # plot the distribution of prediction values
    '''plt.figure()
    #plt.scatter(range(y_test.shape[0]),response,s=2)
    plt.scatter(range(y_test.shape[0]),[0.5]*y_test.shape[0],s=1)
    red = [k for k in range(len(error)) if error[k]==1]
    blue = pd.DataFrame(data=range(y_test.shape[0]))
    blue = blue.drop(red,inplace=False)
    re_red = response.drop(blue.index,inplace=False)
    re_blue = response.drop(red,inplace=False)
    plt.scatter(red,re_red,s=2)
    plt.scatter(blue,re_blue,s=2)
    plt.xlabel('Data Instances')
    plt.ylabel(f'{chr(952)}^T x')
    plt.title(f'Overlook of Model Prediction for Train Set {i} (alpha={alpha})')'''
    # plt.savefig(f'figures/l2_regularized_{alpha}_{i}_distribution.jpg') # Saving the figure

    # return sum(error)/y_test.shape[1] # Used when executing the plot_error() function
    return sum(error)/y_test.shape[0] # Used for the rest of the time


def recall(data_test, y_test, theta,i,alpha): # compute TPR (True Positive Rate) and FPR (False Positive Rate)
    response = sigmoid(theta.transpose(),data_test)

    predict = np.array(np.floor(response*2))
    correct = [max(0,i) for i in 1-predict-y_test]
    false = [max(0,i) for i in predict+y_test-1]

    TPR = sum(correct)/sum(1-y_test)
    FPR = sum(false)/sum(y_test)
    
    return TPR, FPR


def standard_logistic(trial,step):
    error_standard, timing, recalling, fpr = [], [], [], []
    # f = open('theta_standard.csv','w') # Write the theta values into a .csv file
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["Condition","Theta"])
    for alpha in step:
        alpha_error, alpha_time, alpha_recall, alpha_fpr = [], [], [], []
        for i in range(trial):
            data_train = pd.DataFrame(data=pd.read_csv(f'standard/train_{i+1}.data'))
            data_test = pd.DataFrame(data=pd.read_csv(f'standard/test_{i+1}.data'))
            y_train = np.array(pd.DataFrame(data=pd.read_csv(f'standard/train_y_{i+1}.data'))).transpose()
            y_test = np.array(pd.DataFrame(data=pd.read_csv(f'standard/test_y_{i+1}.data')))

            data_train = (data_train-data_train.mean())/data_train.std() # data normalization
            data_test = (data_test-data_test.mean())/data_test.std() # data normalization
            data_train = data_train.fillna(0)
            data_test = data_test.fillna(0)

            M = data_train.shape[0]
            t0 = time.time()
            theta = gradient_descent(data_train, y_train, alpha,tol,0,i,0)
            # csv_writer.writerow([f'Data Set {i+1}, learn rate {alpha}']+[i for i in np.array(theta)])
            t1 = time.time()

            alpha_error.append(validate(data_test, y_test,theta,i+1,alpha))
            alpha_time.append(t1-t0)
            TPR, FPR = recall(data_test, y_test, theta, i+1, alpha)
            alpha_recall.append(TPR)
            alpha_fpr.append(FPR)

        timing.append(np.average(alpha_time)) # Store the CPU time data
        error_standard.append(np.average(alpha_error)) # Store mean error rate data
        recalling.append(np.average(alpha_recall)) # Store TPR data
        fpr.append(np.average(alpha_fpr))

        #plt.figure()
        #plt.xlabel('Iteration Times')
        #plt.ylabel(f'Cost J({chr(952)})')
        #plt.title(f'Standard Logistic Regression for Dataset {i+1}')
        #plt.legend()
        #plt.savefig(f'figures/standard_{i+1}.jpg')
    
    # f.close() # close the .csv file

    #plt.figure()
    #plt.plot(step, recalling)
    #plt.xlabel(f'Learning Rate {chr(945)}')
    #plt.ylabel('Recall Rate')
    #plt.title('Recall Rate w.r.t. Learning Rate (Standard Logistic Regression)')
    #plt.savefig('figures/recall_standard_learn_rate.jpg')

    '''plt.figure()
    plt.scatter(fpr, recalling)
    plt.xlabel(f'FPR (False Positive Rate)')
    plt.ylabel('Recall (True Positive Rate)')
    plt.title('ROC: TPR vs. FPR (Standard Logistic Regression)')'''
    # plt.savefig(f'figures/roc_standard.jpg')
    return error_standard, timing, recalling

def regularized_log_regression(K,step,lamb):
    e2, timing, recalling, fpr = [], [], [], []
    # f = open('theta_regularzied.csv','w')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["Condition","Theta"])
    for alpha in step:
        alpha_error, alpha_time, alpha_recall, alpha_fpr = [], [], [], []
        for i in range(K): # cross validation

            data_train = pd.DataFrame(data=pd.read_csv(f'regularized_regression/train_{i+1}.data'))
            data_test = pd.DataFrame(data=pd.read_csv(f'regularized_regression/test_{i+1}.data'))
            y_train = np.array(pd.DataFrame(data=pd.read_csv(f'regularized_regression/train_y_{i+1}.data'))).transpose()
            y_test = np.array(pd.DataFrame(data=pd.read_csv(f'regularized_regression/test_y_{i+1}.data')))

            data_train = (data_train-data_train.mean())/data_train.std() # data normalization
            data_test = (data_test-data_test.mean())/data_test.std() # data normalization
            data_train = data_train.fillna(0)
            data_test = data_test.fillna(0)

            t0 = time.time()
            theta_2 = gradient_descent(data_train, y_train,alpha,tol,1,i,lamb)
            # csv_writer.writerow([f'Regularized Data Set {i+1}, learning rate {alpha}']+[i for i in np.array(theta_2)])
            
            t1 = time.time()

            alpha_error.append(validate(data_test, y_test,theta_2,i+1,alpha))
            alpha_time.append(t1-t0) 
            TPR, FPR = recall(data_test, y_test, theta_2, i+1, alpha)
            alpha_recall.append(TPR)
            alpha_fpr.append(FPR)
        e2.append(np.average(alpha_error)) # Store mean error rate data
        fpr.append(np.average(alpha_fpr)) # Store TPR data
        timing.append(np.average(alpha_time)) # Store the CPU time data
        recalling.append(np.average(alpha_recall))

        # plt.figure()
        # theta_2 = gradient_descent(data_train, y_train,step,tol,1,i,lamb)
        # e2.append(validate(data_test, y_test,theta_2))

        # plt.xlabel('Iteration Times')
        # plt.ylabel(f'Cost J({chr(952)})')
        # plt.title(f'L2 Logistic Regression for Dataset {i+1}')
        # plt.legend()
        #plt.savefig(f'figures/l2_regularized_{i+1}.jpg')

    # f.close()

    #plt.figure()
    #plt.plot(step, recalling)
    #plt.xlabel(f'Learning Rate {chr(945)}')
    #plt.ylabel('Recall Rate')
    #plt.title('Recall Rate w.r.t. Learning Rate (L2 Regularized Logistic Regression)')
    #plt.savefig('figures/recall_l2_learn_rate.jpg')

    plt.figure()
    plt.scatter(fpr, recalling)
    plt.xlabel(f'FPR (False Positive Rate)')
    plt.ylabel('Recall (True Positive Rate)')
    plt.title('ROC: TPR vs. FPR (L2 Regularized Logistic Regression)')
    # plt.savefig(f'figures/roc_l2_{lamb}.jpg')
    return e2, timing, recalling

def plot_error(): # Plot mean error rate on training and testing sets with respect to their fractions
    data = pd.DataFrame(data=pd.read_csv('error/data.data'))
    y = pd.DataFrame(data=pd.read_csv('error/y.data'))
    data = (data-data.mean())/data.std()
    N = data.shape[0]
    alpha = 1
    tol = 10e-4

    error_train, error_test, cputime = [],[],[]

    for i in range(1,N,10):
        data_train = data[:i]
        data_test = data.drop(range(i),inplace=False)
        y_train = np.array(y[:i]).transpose()
        y_test = np.array(y.drop(range(i),inplace=False)).transpose()

        t0 = time.time()
        theta = gradient_descent(data_train,y_train,alpha,tol,0,i,0)
        # theta = gradient_descent(data_train,y_train,alpha,tol,1,i,2)
        t1 = time.time()
    
        error_train.append(validate(data_train,y_train,theta,i,alpha))
        error_test.append(validate(data_test,y_test,theta,i,alpha))
        cputime.append(t1-t0)
    plt.figure()
    plt.plot(np.array(range(0,N-1,10)).reshape(-1,1)/N,error_train,label = 'train')
    plt.plot(np.array(range(0,N-1,10)).reshape(-1,1)/N,error_test,label='test')
    plt.xlabel('Percentage of data in the training set (%)')
    plt.ylabel('Error Rate')
    # plt.title('Error Rate w.r.t. Data Instances in Training Set (Standard Logistic Regression)')
    plt.title('Error Rate w.r.t. Data Instances in Training Set (L2 Regularized Logistic Regression)')
    plt.legend()
    plt.savefig('error/error_standard.jpg')
    # plt.savefig('error/error_l2_regularized.jpg')

    plt.figure()
    plt.plot(np.array(range(0,N-1,10))/N,np.exp(np.array(cputime)))
    plt.xlabel('Percentage of data in the training set (%)')
    plt.ylabel('CPU Time')
    # plt.title('CPU Time w.r.t. Data Instances in Training Set (Standard Logistic Regression)')
    plt.title('CPU Time w.r.t. Data Instances in Training Set (L2 Regularized Logistic Regression)')
    # plt.savefig('error/cputime_standard.jpg')
    #plt.savefig('error/cputime_l2_regularized_cost.jpg')

def error_wrt_learn_rate(K):
    error_l2 = []
    for i in range(50):
        error_l2.append(regularized_log_regression(K,(25+i)/50,2))
    plt.figure()
    plt.plot((np.array(range(1,51))+24)/50,error_l2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Error Rate')
    plt.title('Error Rate of L2 Regularized Model w.r.t. Different Learning Rates')
    plt.savefig('figures/error_wrt_learn_rate_l2')

def l2_error_wrt_lamb(K):
    lamb,error = np.array(range(100))/20,[]
    for i in lamb:
        error.append(regularized_log_regression(K,1,i))
    plt.figure()
    plt.plot(np.array(range(100))/20,error)
    plt.xlabel('Lambda')
    plt.ylabel('Error Rate')
    plt.title('Error Rate of L2 Regularized Model w.r.t. Different Lambda')
    plt.savefig('figures/error_wrt_lambda_l2')

def check_K():
    option = range(2,100)
    data = pd.DataFrame(data=pd.read_csv('error/data.data'))
    y = pd.DataFrame(data=pd.read_csv('error/y.data'))
    error = []
    lamb = 2
    alpha = 0.5
    for K in option:
        e2 = []
        size = data.shape[0]//K

        for i in range(K):
            data_test = data[size*i:size*(i+1)]
            data_train = data.drop(data.index[size*i:size*(i+1)],inplace=False)
            y_test = np.array(y[size*i:size*(i+1)])
            y_train = np.array(y.drop(data.index[size*i:size*(i+1)],inplace=False)).transpose()
            
            data_train = (data_train-data_train.mean())/data_train.std() # data normalization
            data_test = (data_test-data_test.mean())/data_test.std() # data normalization
            data_train = data_train.fillna(0)
            data_test = data_test.fillna(0)

            theta_2 = gradient_descent(data_train, y_train,alpha,tol,1,i,lamb)
            e2.append(validate(data_test, y_test,theta_2,i+1,alpha))
        error.append(np.average(e2))

    plt.figure()
    plt.plot(option,error)
    plt.xlabel('Choice of Folds K')
    plt.ylabel('Mean Error Rate')
    plt.title('Mean Error Rate w.r.t. K Folds in Cross Validation in L2')
    plt.savefig('error/error_wrt_K.jpg')

def bias_variance(train_original, y_original, train_new, y_new, i, step,l):
    size = train_original.shape[0]
    tol = 10e-4
    theta_origin = gradient_descent(train_original,y_original, step, tol, 0, i, l)
    theta_new = gradient_descent(train_new, y_new, step, tol, 0, i, l)
    
    predict_old = np.array(np.floor(sigmoid(theta_origin.transpose(), train_original)*2))
    predict_new = np.array(np.floor(sigmoid(theta_new.transpose(), train_original)*2))
    
    variance = sum(abs(predict_old-predict_new))/size
    
    bias = validate(train_original,y_original.transpose(),theta_origin,i,step)
     
    return variance, bias

def trade_off(i,step,l):
    train_original = pd.DataFrame(data=pd.read_csv(f'standard/train_{i}.data'))
    train_original = (train_original - train_original.mean())/train_original.std()
    y_original = np.array(pd.DataFrame(data=pd.read_csv(f'standard/train_y_{i}.data'))).transpose()
    all_variance = []
    for j in range(1,7):
        if i != j:
            train_new = pd.DataFrame(data=pd.read_csv(f'standard/train_{j}.data'))
            train_new = (train_new - train_new.mean())/train_new.std()
            y_new = np.array(pd.DataFrame(data=pd.read_csv(f'standard/train_y_{j}.data'))).transpose()
            variance, bias = bias_variance(train_original,y_original,train_new,y_new,j,step,l)
            all_variance.append(variance)
    return np.average(all_variance), bias

def bias_variance_wrt_learn_rate(step):
    for i in range(1,7):
        variance, bias = [], []
        for alpha in step:
            i_variance, i_bias = trade_off(i,alpha,2)
            variance.append(i_variance)
            bias.append(i_bias)

        plt.figure()
        plt.plot(step,variance,label='variance')
        plt.plot(step,bias,label='bias')
        plt.xlabel('Learning Rate')
        plt.ylabel('Variance/ Bias')
        plt.title('Variance and Bias with Respect to Learning Rate')
        plt.legend()
        plt.savefig(f'figures/bv_learn_rate_{i}.jpg')

def bias_variance_wrt_lambda(lamb):
    for i in range(1,7):
        variance, bias = [], []
        for l in lamb:
            i_variance, i_bias = trade_off(i,1,l)
            variance.append(i_variance)
            bias.append(i_bias)
            print(i_variance)

        plt.figure()
        plt.plot(lamb,variance,label='variance')
        plt.plot(lamb,bias,label='bias')
        plt.xlabel('Lambda')
        plt.ylabel('Variance/ Bias')
        plt.title(f'Variance and Bias with Respect to {chr(955)}')
        plt.legend()
        plt.savefig(f'figures/bv_lambda_{i}.jpg')   


K = 3 # split dataset into K folds for cross validation
step = [0.1,1,3,5] # step size in gradient descent of cost function
# step = np.linspace(0.1,5,30) # step size in gradient descent of cost function
lamb = np.array(range(1,10))/2
tol = 10e-04 # tolerance in gradient descent of cost function

trial = 6 # number of times to perform regression (to calculate mean error rate)

def main():
    standard_mean_error, timing, std_recall = standard_logistic(trial,step)
    l2_mean_error, timing, l2_recall = regularized_log_regression(K,step,2)

    # the two lines below print a series of mean error rates wrt learn rate (defined in line 386 or 387)
    print('Mean Error rate for standard logistic regression (for different learning rates):',standard_mean_error)
    print('Mean Error rate for l2 logistic regression (for different learning rates):',l2_mean_error)

    # the two lines below print a series of recall rates wrt learn rate (defined in line 386 or 387)
    print('Recall rate for standard logistic regression (for different learning rates):',std_recall)
    print('Recall rate for l2 logistic regression (for different learning rates):',l2_recall)
    
    # plot_error()   # please uncomment line 83 and comment line 84 when executing this line !!
    # error_wrt_learn_rate(K)
    # l2_error_wrt_lamb(K)
    # check_K()
    # bias_variance_wrt_lambda(lamb)
    # bias_variance_wrt_learn_rate(step)

main()
