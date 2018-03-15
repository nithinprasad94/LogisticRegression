import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time

sess = tf.Session()

################### PART 1: LINEAR REGRESSION ########################
#Implement Mini-batch (SGD)

#Step 1: Use the given code to download the two-class notMNIST dataset

#Function to convert numpy array to Tensor
def numpy_to_tensor(array):
    out_tensor = tf.convert_to_tensor(array, dtype=tf.float32)
    return out_tensor

#Function that computes MSE-loss on the set: W, X and Y
def MSE_linear_regression(w,data_subtensor,target_subtensor):

    #print("IN MSE LINEAR REGRESSION")

    #PRE-PROCESSING:
    # process all incoming tensors into the proper-shapes and append
    # column of 1's into X to handle the bias values
    
    #Acquire tensor shape
    tensor_shape = data_subtensor.get_shape().as_list()
    data_count = tensor_shape[0]

    #Note: X_subtensor_rs is data_count*784 ... need to add extra value of 1
    # ie. x0 = 1 for bias for each data point.
    X_subtensor_rs = tf.reshape(data_subtensor,[data_count,-1])
    b_subtensor = tf.ones([data_count,1])
    X = tf.concat([b_subtensor,X_subtensor_rs],1)
    #print("X shape: ", X.get_shape().as_list())

    y = target_subtensor
    
    #Notes:
    # w - weight-vector => let this be a row-vector (1x785 matrix)
    # X - data-matrix => let this be a matrix (eg. 500x785 matrix)
    # y - target-vector => let this be a row-vector (eg. 500x1 matrix)

    #Prime W and Y for operations
    w_prime = tf.expand_dims(w,1) #Turn W into a 785x1 col vec
    #print("w_prime shape: ", w_prime.get_shape().as_list())
    #y_prime = tf.expand_dims(y,1) #Turn Y into a 500x1 col vec
    y_prime = y
    #print("y_prime shape: ", y_prime.get_shape().as_list())
    temp_sess = tf.Session()

    #Do operations:
    op1 = tf.matmul(X,w_prime) #results in a 500x1 vector
    
    print("op1 shape: ", op1.get_shape().as_list())

    op2 = tf.subtract(op1,y_prime) #still have a 500x1 vector

    print("op2 shape: ", op2.get_shape().as_list())

    op3 = tf.square(op2) #still have a 500x1 vector

    print("op3 shape: ", op3.get_shape().as_list())

    result = tf.reduce_mean(op3)

    #print("result shape: ", result.get_shape().as_list())

    half_constant = tf.constant(0.5)

    mse_result = result*half_constant
    
    return mse_result

#Function that computes Weight-Decay Loss for Linear Regression
def WDL_linear_regression(w,weight_decay):

    #print("IN WDL LINEAR REGRESSION")

    weight_constant = tf.constant(weight_decay*0.5)
    
    wdl_error = (weight_constant)*tf.reduce_sum(tf.square(w))

    return wdl_error

#Function that computes Total Loss for Linear Regression
def TL_linear_regression(w,X,y,weight_decay):

    #print("IN TL LINEAR REGRESSION")
    total_error = MSE_linear_regression(w,X,y) + WDL_linear_regression(w,weight_decay)
    #print("returning from TL fn")
    return total_error
    
#Function to load and return data for the 2-class notMNIST problem
def load_notMNIST_two_class():

    #Load the data ...
    with np.load("notMNIST.npz") as data :

        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255
        Target = Target[dataIndx].reshape(-1,1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data,Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        return (trainData,trainTarget,validData,validTarget,testData,testTarget)

################## RUN BASIC LINEAR REGRESSION ##########################
def linear_regression_basic(functionality,inp_batch_size,inp_learning_rate, \
                            inp_weight_decay,inp_num_iterations):

    #Functionality == 1 -> do Training
    #Functionality == 2 -> do Validation
    #Functionality == 3 -> do Testing
    
    #LOAD DATA
    trainData, trainTarget, validData, validTarget, testData, testTarget = \
               load_notMNIST_two_class()

    useData = None
    useTarget = None

    #Regardless of 1 or 2 or 3, always train the weights!
    useData = trainData
    useTarget = trainTarget

    #EXAMINE DATA

    #FORMAT DATA INTO TENSORS
    #Convert all the training data into a large-tensor
    training_data_tensor = numpy_to_tensor(useData)
    tensor_shape = training_data_tensor.get_shape().as_list()
    print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    training_target_tensor = numpy_to_tensor(useTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    #784 hyper-parameters -> one for each pixel + 1 hyperparameter for bias
    hyper_param_count = data_width*data_height + 1
    
    #print("Total number of data points for training: ", data_count)

    #Constants!!!!
    batch_size = inp_batch_size #A constant that is invariant for a single run of the program
    learning_rate = inp_learning_rate
    weight_decay = inp_weight_decay
    num_iterations = inp_num_iterations
    
    epoch_size = data_count/batch_size #eg. 3500/500 = 7
    num_epochs = math.ceil(num_iterations/(epoch_size)) #eg. 20,000/7 = 2858 (rounded)

    #initialize the tf variables and place-holders
    X = tf.placeholder(tf.float32,[batch_size,data_height,data_width])

    #print("Shape of placeholder X: ", X.get_shape().as_list())
    w = tf.Variable(tf.ones(data_height*data_width+1,1)) #initialize a 785x1 matrix to all zeroes
    #print("Shape of variable w: ", w.get_shape().as_list())
    y = tf.placeholder(tf.float32,[batch_size,1])
    #print("Shape of placeholder y: ", y.get_shape().as_list())

    init_run = tf.global_variables_initializer()

    sess.run(init_run)

    total_error = TL_linear_regression(w,X,y,weight_decay)    
    
    #Now, create the optimizer!
    single_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_error)

    #Finally, set up an error-tracking metric
    error_vector = np.empty(shape=[1],dtype=float)
    
    #PERFORM SGD OVER TRAINING SET
    for iteration in range(num_iterations):

        #Progress-bar (updates every 5000 iterations)
        if (iteration%2000 == 0):
            print("iteration: ",iteration)
            #Check if weights are updating ...
            #print("weight -> w[0]",sess.run(w[0]))
            #print("weight -> w[1]",sess.run(w[1]))
            #print("weight -> w[2]",sess.run(w[2]))

        batch_no = (iteration)%epoch_size

        check_test_lin_reg_model(w,useData,useTarget,check_index)        #Note: if start_index + batch_size > data_count ... wrap back a little bit!
        start_index = int(batch_size*batch_no) #Inclusive
        
        if start_index + batch_size <= data_count:
            end_index = int(start_index + batch_size)
        else:
            start_index = data_count - batch_size #Move start_index back so that it is a batch_size away from data_count
            end_index = data_count
             
        #end_index = int(min(start_index + batch_size, data_count)) #Exclusive
    
        #Run a single instance of the optimizer
        sess.run(single_step,feed_dict={X:useData[start_index:end_index],y:useTarget[start_index:end_index]})

        #Accumulate error-values into error-metric ONCE per epoch
        if (iteration%epoch_size == epoch_size-1):
            error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:useData[start_index:end_index],y:useTarget[start_index:end_index]}))            

        #Accumulate the last value of the last epoch, since the last epoch
            # may not run to completion (ie. 20000/7 is not an integer)
        if iteration == num_iterations-1:
            if len(error_vector) < num_epochs:
                error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:trainData[start_index:end_index],y:trainTarget[start_index:end_index]}))            

    #print("Final error: ",error_vector[-1])
    
    #Return the error_vector
    return error_vector

#Basically an indicator function that determines whether or not the model
# gives the same value as the result
def check_test_lin_reg_model(w_in,x_in,y_in):

    #Convert X into a proper vector
    tensor_shape = x_in.get_shape().as_list()

    b = tf.constant([1.])
    x_flat = tf.reshape(x_in,[-1])
    x_mid = tf.concat([b,x_flat],0)
    x = tf.expand_dims(x_mid,[0])
    w_shape = w_in.get_shape().as_list()
    w = tf.expand_dims(w_in,[1])
    mult = tf.matmul(x,w)
    format_mult = tf.reshape(mult,[1]) #Remove the outer-bracket
    
    return (format_mult[0],y_in[0])

################## RUN ADVANCED LINEAR REGRESSION ##########################
def linear_regression_advanced(functionality,inp_batch_size,inp_learning_rate, \
                            inp_weight_decay,inp_num_iterations):

    #Functionality == 1 -> do Training
    #Functionality == 2 -> do Validationcheck_test_lin_reg_model(w,useData,useTarget,check_index)
    #Functionality == 3 -> do Testing

    #LOAD DATA
    trainData, trainTarget, validData, validTarget, testData, testTarget = \
               load_notMNIST_two_class()

    useData = None
    useTarget = None


    #Regardless of 1 or 2 or 3, always train the weights!
    useData = trainData
    useTarget = trainTarget

    #EXAMINE DATA

    #FORMAT DATA INTO TENSORS
    #Convert all the training data into a large-tensor
    training_data_tensor = numpy_to_tensor(useData)
    tensor_shape = training_data_tensor.get_shape().as_list()
    #print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    training_target_tensor = numpy_to_tensor(useTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    #784 hyper-parameters -> one for each pixel + 1 hyperparameter for bias
    hyper_param_count = data_width*data_height + 1
    
    #print("Total number of data points for training: ", data_count)

    #Constants!!!!
    batch_size = inp_batch_size #A constant that is invariant for a single run of the program
    learning_rate = inp_learning_rate
    weight_decay = inp_weight_decay
    num_iterations = inp_num_iterations
    
    epoch_size = data_count/batch_size #eg. 3500/500 = 7
    num_epochs = math.ceil(num_iterations/(epoch_size)) #eg. 20,000/7 = 2858 (rounded)

    #initialize the tf variables and place-holders
    X = tf.placeholder(tf.float32,[batch_size,data_height,data_width])

    #print("Shape of placeholder X: ", X.get_shape().as_list())
    w = tf.Variable(tf.ones(data_height*data_width+1,1)) #initialize a 785x1 matrix to all ones
    #print("Shape of variable w: ", w.get_shape().as_list())
    y = tf.placeholder(tf.float32,[batch_size,1])
    #print("Shape of placeholder y: ", y.get_shape().as_list())

    init_run = tf.global_variables_initializer()

    sess.run(init_run)

    total_error = TL_linear_regression(w,X,y,weight_decay)    
    
    #Now, create the optimizer!
    single_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_error)

    #Finally, set up an error-tracking metric
    error_vector = np.empty(shape=[1],dtype=float)
    
    #PERFORM SGD OVER TRAINING SET
    for iteration in range(num_iterations):

        #Progress-bar (updates every 5000 iterations)
        if (iteration%2000 == 0):
            print("iteration: ",iteration)
            #Check if weights are updating ...
            #print("weight -> w[0]",sess.run(w[0]))
            #print("weight -> w[1]",sess.run(w[1]))
            #print("weight -> w[2]",sess.run(w[2]))

        batch_no = (iteration)%epoch_size

        #Note: if start_index + batch_size > data_count ... wrap back a little bit!
        start_index = int(batch_size*batch_no) #Inclusive
        
        if start_index + batch_size <= data_count:
            end_index = int(start_index + batch_size)
        else:
            start_index = data_count - batch_size #Move start_index back so that it is a batch_size away from data_count
            end_index = data_count
             
        #end_index = int(min(start_index + batch_size, data_count)) #Exclusive
    
        #Run a single instance of the optimizer
        sess.run(single_step,feed_dict={X:useData[start_index:end_index],y:useTarget[start_index:end_index]})

        #Accumulate error-values into error-metric ONCE per epoch
        if (iteration%epoch_size == epoch_size-1):
            error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:useData[start_index:end_index],y:useTarget[start_index:end_index]}))            

        #Accumulate the last value of the last epoch, since the last epoch
            # may not run to completion (ie. 20000/7 is not an integer)
        if iteration == num_iterations-1:
            if len(error_vector) < num_epochs:
                error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:trainData[start_index:end_index],y:trainTarget[start_index:end_index]}))            

    #Print the final MSE Error
    print("Final Error: ",error_vector[-1])
    
    ### Now try to do Validation & Testing!!!!!

    #Now change useData depending on whether validation OR test set was selected
    if functionality == 2:
        useData = validData
        useTarget = validTarget

    elif functionality == 3:
        useData = testData
        useTarget = testTarget

    #Note: w contains all the weights we need to "check new data"    

    #FORMAT DATA INTO TENSORS
    #Convert all the training data into a large-tensor
    check_data_tensor = numpy_to_tensor(useData)
    tensor_shape = check_data_tensor.get_shape().as_list()
    #print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    check_target_tensor = numpy_to_tensor(useTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    correct_count = [0,0,0,0,0] #tracks the number of correct guesses by the model

    #w = tf.Variable(tf.ones(data_height*data_width+1,1)) #Temporary measure
    x_check = tf.placeholder(tf.float32,[data_height,data_width])
    y_check = tf.placeholder(tf.float32,[1])
    model_val = tf.placeholder(tf.float32,[])
    y_val = tf.placeholder(tf.float32,[])

    equality = check_test_lin_reg_model(w,x_check,y_check)

    thresholds = [0.1,0.2,0.3,0.4,0.5]
    
    #Loop through all the data
    for check_index in range(data_count):

        #print("weight -> w[0]",sess.run(w[0]))
        #print("weight -> w[1]",sess.run(w[1]))
        #print("weight -> w[2]",sess.run(w[2]))
        
        #For each data-point, run w*x == y
        model_val,output = (sess.run(equality,feed_dict={x_check:useData[check_index],y_check:useTarget[check_index]}))

        # 1) model_val could be negative, normalize about 0!
        # 2) difference between a positive model_val and the output can be positive
        #    or negative ... normalize!

        for i in range(len(thresholds)):
            #if abs(abs(model_val)-output) < thresholds[i]:
            #    correct_count[i] += 1

            if output == 1. and model_val > output-thresholds[i]:
                correct_count[i] += 1
            elif output == 0. and model_val < output+thresholds[i]:
                correct_count[i] += 1

    #Accuracy is given by percentage of correct guesses!

    #print("Total checks: ", data_count)
    #print("Correct evaluations: ", correct_response_count)
    accuracy_vector = []
    for i in range(len(correct_count)):
        accuracy_vector.append(correct_count[i]/data_count)
    #print("Accuracy: ", correct_response_count/data_count)
    
    return accuracy_vector

def MSE_normal_equation(w,data_subtensor,target_subtensor):

    print("IN MSE NORMAL EQN")

    #PRE-PROCESSING:
    # process all incoming tensors into the proper-shapes and append
    # column of 1's into X to handle the bias values
    tensor_shape = data_subtensor.get_shape().as_list()
    data_count = tensor_shape[0]

    #Note: X_subtensor_rs is data_count*784 ... need to add extra value of 1
    # ie. x0 = 1 for bias for each data point.
    X_subtensor_rs = tf.reshape(data_subtensor,[data_count,-1])
    b_subtensor = tf.ones([data_count,1])
    X = tf.concat([b_subtensor,X_subtensor_rs],1)
    #print("X shape: ", X.get_shape().as_list())

    y = target_subtensor

    #y_prime = tf.expand_dims(y,1) #Turn Y into a 500x1 col vec
    y_prime = y
    #print("y_prime shape: ", y_prime.get_shape().as_list())
    temp_sess = tf.Session()

    #Do operations:
    op1 = tf.matmul(X,w) #results in a 500x1 vector
    
    #print("op1 shape: ", op1.get_shape().as_list())

    op2 = tf.subtract(op1,y_prime) #still have a 500x1 vector

    #print("op2 shape: ", op2.get_shape().as_list())

    op3 = tf.square(op2) #still have a 500x1 vector

    #print("op3 shape: ", op3.get_shape().as_list())

    result = tf.reduce_mean(op3)

    #print("result shape: ", result.get_shape().as_list())

    half_constant = tf.constant(0.5)

    mse_result = result*half_constant
    
    return mse_result

def return_normal_weights(X_in,y):

    #Do pre-processing on X
    X_shape = X_in.get_shape().as_list()

    data_count = X_shape[0]

    ones = tf.ones([data_count,1])

    X_flat = tf.reshape(X_in,[data_count,-1])

    X = tf.concat([ones,X_flat],1)
    
    X_transpose = tf.transpose(X)
    prod = tf.matmul(X_transpose,X)
    inv = tf.matrix_inverse(prod)
    second_prod = tf.matmul(inv,X_transpose)
    result = tf.matmul(second_prod,y)

    return result

def check_normal_eqn_model(w_in,x_in,y_in):

    #Convert X into a proper vector
    tensor_shape = x_in.get_shape().as_list()

    b = tf.constant([1.])
    x_flat = tf.reshape(x_in,[-1])
    x_mid = tf.concat([b,x_flat],0)
    x = tf.expand_dims(x_mid,[0])
    w_shape = w_in.get_shape().as_list()
    #w = tf.expand_dims(w_in,[1])
    mult = tf.matmul(x,w_in)
    format_mult = tf.reshape(mult,[1]) #Remove the outer-bracket
    
    return (format_mult[0],y_in[0])

def linear_regression_normal(functionality,inp_batch_size,inp_learning_rate, \
                            inp_weight_decay,inp_num_iterations):

    #Functionality == 1 -> do Training
    #Functionality == 2 -> do Validationcheck_test_lin_reg_model(w,useData,useTarget,check_index)
    #Functionality == 3 -> do Testing

    #LOAD DATA
    trainData, trainTarget, validData, validTarget, testData, testTarget = \
               load_notMNIST_two_class()

    useData = None
    useTarget = None

    #Regardless of 1 or 2 or 3, always train the weights!
    useData = trainData
    useTarget = trainTarget


    #Convert all the training data into a large-tensor
    check_data_tensor = numpy_to_tensor(useData)
    tensor_shape = check_data_tensor.get_shape().as_list()
    #print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    check_target_tensor = numpy_to_tensor(useTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    correct_count = [0,0,0,0,0] #tracks the number of correct guesses by the model

    #w = tf.Variable(tf.ones(data_height*data_width+1,1)) #Temporary measure
    X = tf.placeholder(tf.float32,[data_count,data_height,data_width])
    y = tf.placeholder(tf.float32,[data_count,1])
    w = tf.Variable(tf.ones([data_height*data_width+1,1])) #initialize a 785x1 matrix to all ones

    computed_weights = return_normal_weights(X,y)

    weight_assigner = tf.assign(w,computed_weights)
    
    mse = MSE_normal_equation(w,X,y)

    init_run = tf.global_variables_initializer()

    sess.run(init_run)

    sess.run(weight_assigner,feed_dict = {X: useData, y: useTarget})
    #print(weights.get_shape().as_list())

    print("MSE Error: ",sess.run(mse,feed_dict = {X: useData, y: useTarget}))

    #Now change useData depending on whether validation OR test set was selected
    if functionality == 2:
        useData = validData
        useTarget = validTarget

    elif functionality == 3:
        useData = testData
        useTarget = testTarget  

    #FORMAT DATA INTO TENSORS
    #Convert all the training74.7635 data into a large-tensor
    check_data_tensor = numpy_to_tensor(useData)
    tensor_shape = check_data_tensor.get_shape().as_list()
    #print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    check_target_tensor = numpy_to_tensor(useTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    correct_count = [0,0,0,0,0] #tracks the number of correct guesses by the model

    #w = tf.Variable(tf.ones(data_height*data_width+1,1)) #Temporary measure
    x_check = tf.placeholder(tf.float32,[data_height,data_width])
    y_check = tf.placeholder(tf.float32,[1])
    model_val = tf.placeholder(tf.float32,[])
    y_val = tf.placeholder(tf.float32,[])

    print("here")
    equality = check_normal_eqn_model(w,x_check,y_check)

    thresholds = [0.1,0.2,0.3,0.4,0.5]
    
    #Loop through all the data
    for check_index in range(data_count):
        #print(check_index)

        x_feed = useData[check_index]
        y_feed = useTarget[check_index]

        #For each data-point, run w*x == y
        #print("here2")
        model_val,output = (sess.run(equality,feed_dict={x_check:x_feed,y_check:y_feed}))

        # 1) model_val could be negative, normalize about 0!
        # 2) difference between a positive model_val and the output can be positive
        #    or negative ... normalize!

        for i in range(len(thresholds)):
            #if abs(abs(model_val)-output) < thresholds[i]:
            #    correct_count[i] += 1

            if output == 1. and model_val > output-thresholds[i]:
                correct_count[i] += 1
            elif output == 0. and model_val < output+thresholds[i]:
                correct_count[i] += 1

    #Accuracy is given by percentage of correct guesses!

    print("Total checks: ", data_count)
    print("Correct evaluations: ", correct_count)
    accuracy_vector = []
    for i in range(len(correct_count)):
        accuracy_vector.append(correct_count[i]/data_count)


    return accuracy_vector
        
if __name__ == "__main__":
    
#PART 1: LINEAR REGRESSION

    #1.1: do linear regression with given parameters and vary the learning rate
##    error_vec1 = linear_regression_basic(1,500,0.005,0,20000)
##    error_vec2 = linear_regression_basic(1,500,0.001,0,20000)
##    error_vec3 = linear_regression_basic(1,500,0.0001,0,20000)    
##
##    fig, ax = plt.subplots()
##    ax.plot(range(len(error_vec1)),error_vec1, label = 'Learning Rate: 0.005')
##    ax.plotm(range(len(error_vec2)),error_vec2, label = 'Learning Rate: 0.001')
##    ax.plot(range(len(error_vec3)),error_vec3, label = 'Learning Rate: 0.0001')
##    
##    legend = ax.legend(loc = 'upper right', shadow = True)
##
##    frame = legend.get_frame()
##    frame.set_facecolor('0.90')
##
##    num_epochs = len(error_vec1)
##    plt.xlabel('Epochs')
##    plt.ylabel('Total Error')
##    plt.show()

    #Bonus run
    #error_vec = linear_regression_basic(1,500,0.005,0,20000)
    
    #1.2: Vary the batch size for a given learning rate of 0.005 (optimum)
##    start = 0.get_shape()
##    end = 0
##
##    print("Experiment 1.2 BATCH SIZE: 500")
##    start = time.time()
##    error_vec1 = linear_regression_basic(1,500,0.005,0,20000)
##    end = time.time()
##    print("Final MSE: ", error_vec1[-1])
##    print("Linear Regression Duration: ", end-start)
##
##    print("Experiment 1.2 BATCH SIZE: 1500")
##    start = time.time()
##    error_vec2 = linear_regression_basic(1,1500,0.005,0,20000)
##    end = time.time()    
##    print("Final MSE: ", error_vec2[-1])
##    print("Linear Regression Duration: ", end-start)
##
##    print("Experiment 1.2 BATCH SIZE: 3500")
##    start = time.time()
##    error_vec3 = linear_regression_basic(1,3500,0.005,0,20000)
##    end = time.time()    
##    print("Final MSE: ", error_vec3[-1])
##    print("Linear Regression Duration: ", end-start)

    #1.3: Vary the weight decay coefficient on the validation set
    # also try out the best validation accuracy on the test set
    
##    acc1_vector = linear_regression_advanced(2,500,0.005,0,20000)
##    print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    print("Accuracy for lambda = 0: ", acc1_vector)
##
##    acc2_vector = linear_regression_advanced(2,500,0.005,0.001,20000)
##    print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    print("Accuracy for lambda = 0.001: ", acc2_vector)

##    acc3_vector = linear_regression_advanced(2,500,0.005,0.1,20000)
##    print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    print("Accuracy for lambda = 0.1: ", acc3_vector)

##    acc4_vector = linear_regression_advanced(2,500,0.005,1.,20000)
##    print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    print("Accuracy for lambda = 1.: ", acc4_vector)

##    acc3_test_vector = linear_regression_advanced(3,500,0.005,1.,20000)
##    print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    print("TEST Accuracy for lambda = 0.1: ", acc3_test_vector)

    
    #1.4: Compute Linear Regression (Weights) and Assess Performance using a Normal Equation Function
##    start = time.time()
##    acc3_vector = linear_regression_advanced(2,500,0.005,0,20000)
##    #print("THRESHOLDS: 0.1,0.2,0.3,0.4,0.5")
##    #print("Accuracy for lambda = 0.1: ", acc3_vector)
##    end = time.time()
##    print("Linear Regression Duration: ", end-start)
##
##    start = time.time()
##    acc_vec = linear_regression_normal(2,3500,0.005,1.,20000)
##    end = time.time()
##    print("Linear Regression Duration: ", end-start)

###############################################


    







