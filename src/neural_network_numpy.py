import numpy as np
from matplotlib import pyplot as plt

global train_data_with_bias, train_y, test_data, test_y, loss_vs_iter, train_accuracy, test_accuracy
global train_class_accuracy, test_class_accuracy

with open('train1.txt','r') as f:
    all_data = f.readlines()[3:]
    train_data = np.array([float(j) for i in all_data for j in i[:-1:].split(' ')[:-3]]).reshape(-1,256)
    train_y = np.array([float(k) for i in all_data for k in i[:-1:].split(' ')[-3::]]).reshape(-1,3)
with open('test1.txt','r') as f:
    all_data = f.readlines()[3:]
    test_data = np.array([float(j) for i in all_data for j in i[:-1:].split(' ')[:-3]]).reshape(-1,256)
    test_y = np.array([float(k) for i in all_data for k in i[:-1:].split(' ')[-3::]]).reshape(-1,3)

def FeedForward_BackPropagation(input_data,input_y,i, g_input, g_hidden, test=False):

    # ------------------------- Forward Pass ----------------------------------
    first_layer = input_data@input_layer_weight
    ReLU_layer = np.maximum(first_layer,0)
    # add some bias term to the hidden layer
    hidden_layer = np.hstack([ReLU_layer,np.ones(ReLU_layer.shape[0]).reshape(-1,1)])
    output_layer = hidden_layer@hidden_layer_weight
    Softmax_output = np.exp(output_layer)/np.sum(np.exp(output_layer),axis=1).reshape(-1,1)

    if test==False:
        loss = CrossEntropyLoss(Softmax_output,input_y)
        loss_vs_iter.append(loss/BATCH_SIZE)
        print(f"Iteration {i}. Loss:",loss/BATCH_SIZE)
        
        # ----------------- Backward Propagation -----------------------------
        diff_Softmax = -input_y/Softmax_output
        diff_output = Softmax_output*(1-input_y+diff_Softmax*(1-Softmax_output))
        diff_hidden = diff_output@hidden_layer_weight.T
        diff_ReLU = (diff_hidden[:,:-1]*np.sign(ReLU_layer)).T
        if i == 0: # The first iteration
            grad_input_layer_weight = (diff_ReLU@input_data).T
            grad_hidden_layer_weight = (diff_output.T@hidden_layer).T
        else: # After the first iteration we sum the gradients together
            grad_input_layer_weight = (diff_ReLU@input_data).T + g_input
            grad_hidden_layer_weight = (diff_output.T@hidden_layer).T + g_hidden
        
        # ------------------- Compute Accuracy ------------------------------
        train_prediction = FeedForward_BackPropagation(train_data_with_bias,train_y,0,0,0,test=True)
        test_prediction = FeedForward_BackPropagation(test_data,test_y,0,0,0,test=True)
        train_accuracy.append(OverallAccuracy(train_prediction,train_y))
        test_accuracy.append(OverallAccuracy(test_prediction,test_y))
        train_class_accuracy.append(ClassAccuracy(train_prediction,train_y))
        test_class_accuracy.append(ClassAccuracy(test_prediction,test_y))

        return loss, grad_input_layer_weight, grad_hidden_layer_weight
    else: return Softmax_output

def SGD(BATCH_SIZE, train_data, train_y): # random sample of train data
    indices = np.random.choice(range(369),BATCH_SIZE)
    input_data = train_data[indices]
    input_y = train_y[indices]
    return input_data, input_y

def CrossEntropyLoss(prediction,ground_truth):
    log_prediction = np.log(prediction)
    element_entropy = log_prediction*ground_truth
    cross_entropy_each_dim = -np.sum(element_entropy,axis=1)
    return np.sum(cross_entropy_each_dim)

def OverallAccuracy(prediction,ground_truth): # Compute overall accuracy
    ground_truth = np.argmax(ground_truth, axis=1)
    prediction = np.argmax(prediction, axis=1)
    return 100-100*sum(ground_truth!=prediction)/ground_truth.shape[0]

def ClassAccuracy(prediction,ground_truth): # Compute accuracy on each class
    true_response = np.sum(prediction*ground_truth,axis=0)
    num = np.sum(ground_truth,axis=0)
    return true_response/num

# --------------------- Initialization --------------------------------
m = train_data.shape[1] # number of attributes in the input layer / data dimension
o = train_y.shape[1] # number of output nodes / number of classes
BATCH_SIZE = 25
# add some bias term to the input data
train_data_with_bias = np.hstack((train_data,np.ones(train_data.shape[0]).reshape(-1,1)))
test_data = np.hstack((test_data,np.ones(test_data.shape[0]).reshape(-1,1)))

num_of_hidden_nodes = [5,10,20,50,100,200,500,1000] # try different number of hidden nodes
for h in num_of_hidden_nodes:
    loss_vs_iter, train_accuracy, test_accuracy, train_class_accuracy, test_class_accuracy = [],[],[],[],[]
    input_layer_weight = np.random.randn(m+1,h)/np.sqrt(m+1) # Xavier Initialization
    hidden_layer_weight = np.random.randn(h+1,o)/np.sqrt(h+1) # Xavier Initialization
    input_data, input_y = SGD(BATCH_SIZE,train_data_with_bias,train_y)
    loss, grad_input, grad_hidden = FeedForward_BackPropagation(input_data,input_y,0,0,0)

    # --------------- Gradient Descent Iteration --------------------------
    iter_time, learn_rate = 1, 1e-2
    while loss/BATCH_SIZE > 0.06 and iter_time < 500:
        learn_rate = learn_rate*0.95 # learning rate decay
        input_layer_weight = input_layer_weight - learn_rate*grad_input # update weight layers
        hidden_layer_weight = hidden_layer_weight - learn_rate*grad_hidden # update weight layers
        input_data, input_y = SGD(BATCH_SIZE,train_data_with_bias,train_y)
        loss, grad_input, grad_hidden = FeedForward_BackPropagation(input_data,input_y,iter_time,grad_input,grad_hidden)
        iter_time += 1

    # ----------------------- Report the Performance ---------------------------------------------
    plt.figure()
    plt.plot(range(iter_time),loss_vs_iter)
    plt.xlabel('Iteration Time(s)')
    plt.ylabel('Average Cross Entropy Loss')
    plt.title(f'Decay of Loss Function vs Iterations, h={h}')
    plt.savefig(f'figures/{h}_loss_vs_iter.png')

    plt.figure()
    plt.plot(range(iter_time),train_accuracy,label='Overall Training Accuracy')
    plt.plot(range(iter_time),test_accuracy,label='Overall Testing Accuracy')
    plt.xlabel('Iteration Time(s)')
    plt.ylabel('Training and Testing Accuracy vs Iterations')
    plt.title(f'Training/Testing Accuracy vs Iterations, h={h}')
    plt.legend()
    plt.savefig(f'figures/{h}_train_test_accuracy.png')

    train_class_accuracy = np.array(train_class_accuracy).T
    test_class_accuracy = np.array(test_class_accuracy).T

    plt.figure()
    plt.plot(range(iter_time),train_class_accuracy[0],label='Accuracy on class "6"')
    plt.plot(range(iter_time),train_class_accuracy[1],label='Accuracy on class "8"')
    plt.plot(range(iter_time),train_class_accuracy[2],label='Accuracy on class "9"')
    plt.xlabel('Iteration Time(s)')
    plt.ylabel('Accuracy of Each Class')
    plt.title(f'Training Accuracy of Each Class vs Iterations, h={h}')
    plt.legend()
    plt.savefig(f'figures/{h}_train_accuracy_on_class.png')

    plt.figure()
    plt.plot(range(iter_time),test_class_accuracy[0],label='Accuracy on class "6"')
    plt.plot(range(iter_time),test_class_accuracy[1],label='Accuracy on class "8"')
    plt.plot(range(iter_time),test_class_accuracy[2],label='Accuracy on class "9"')
    plt.xlabel('Iteration Time(s)')
    plt.ylabel('Accuracy of Each Class')
    plt.title(f'Testing Accuracy of Each Class vs Iterations, h={h}')
    plt.legend()
    plt.savefig(f'figures/{h}_test_accuracy_on_class.png')

    print('Final training accuracy:', str(train_accuracy[-1])[:5]+'%,',f'h={h}')
    print('Final testing accuracy', str(test_accuracy[-1])[:5]+'%,',f'h={h}')
