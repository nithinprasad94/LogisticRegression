import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

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

    print("IN MSE LINEAR REGRESSION")

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

#Function that computes Weight-Decay Loss for Linear Regression
def WDL_linear_regression(w,weight_decay):

    print("IN WDL LINEAR REGRESSION")

    weight_constant = tf.constant(weight_decay*0.5)
    
    wdl_error = (weight_constant)*tf.reduce_sum(tf.square(w))

    return wdl_error

#Function that computes Total Loss for Linear Regression
def TL_linear_regression(w,X,y,weight_decay):

    print("IN TL LINEAR REGRESSION")
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
def linear_regression_basic():
    
    #LOAD DATA
    trainData, trainTarget, validData, validTarget, testData, testTarget = \
               load_notMNIST_two_class()

    #EXAMINE DATA
    #print(len(trainData))
    #print(type(trainData[0]));

    #FORMAT DATA INTO TENSORS
    #Convert all the training data into a large-tensor
    training_data_tensor = numpy_to_tensor(trainData)
    tensor_shape = training_data_tensor.get_shape().as_list()
    print(tensor_shape) #Tensor is 28x28 ... 784 pixels need to be operated on!
    training_target_tensor = numpy_to_tensor(trainTarget)

    #INITIALIZE REGRESSION VARIABLES
    data_count = tensor_shape[0]
    data_width = tensor_shape[1]
    data_height = tensor_shape[2]

    #784 hyper-parameters -> one for each pixel + 1 hyperparameter for bias
    hyper_param_count = data_width*data_height + 1
    
    print("Total number of data points for training: ", data_count)

    #Constants!!!!
    batch_size = 500 #A constant that is invariant for a single run of the program
    learning_rate = 0.01
    weight_decay = 0
    num_iterations = 20000
    
    num_batches = math.ceil(data_count/batch_size)
    epoch_size = data_count/batch_size
    num_epochs = math.ceil(num_iterations/(epoch_size))

    #initialize the tf variables and place-holders
    X = tf.placeholder(tf.float32,[batch_size,data_height,data_width])

    print("Shape of placeholder X: ", X.get_shape().as_list())
    w = tf.Variable(tf.ones(data_height*data_width+1,1)) #initialize a 785x1 matrix to all zeroes
    print("Shape of variable w: ", w.get_shape().as_list())
    y = tf.placeholder(tf.float32,[batch_size,1])
    print("Shape of placeholder y: ", y.get_shape().as_list())

    init_run = tf.global_variables_initializer()

    sess.run(init_run)

    total_error = TL_linear_regression(w,X,y,weight_decay)    
    
    #Now, create the optimizer!
    single_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_error)

    #Finally, set up an error-tracking metric
    error_vector = np.empty(shape=[1],dtype=float)
    
    #PERFORM SGD OVER TRAINING SET
    for iteration in range(num_iterations):

        #if (iteration%epoch_size == 0):
        #    print("iteration: ",iteration)
        if (iteration%1000 == 0):
            print("iteration: ",iteration)

        batch_no = (iteration)%num_batches
        #print("Batch number: ",batch_no)

        #Get the batch corresponding to this batch_id
        #batches = return_batch(training_data_tensor, \
        #                                training_target_tensor, batch_id, \
        #                                batch_size)
        #X_batch = batches[0]
        #y_batch = batches[1]

        start_index = batch_size*batch_no #Inclusive
        #print("Start index: ",start_index)
        
        end_index = min(start_index + batch_size, data_count)  #Exclusive
        #print("End index: ",end_index)
    
        #Run a single instance of the optimizer
        sess.run(single_step,feed_dict={X:trainData[start_index:end_index],y:trainTarget[start_index:end_index]})

        #Accumulate error-values into error-metric ONCE per epoch
        if (iteration%epoch_size == epoch_size-1):
            error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:trainData[start_index:end_index],y:trainTarget[start_index:end_index]}))            

        #Accumulate the last value of the last epoch, since the last epoch
            # may not run to completion (ie. 20000/7 is not an integer)
        if iteration == num_iterations-1:
            if len(error_vector) < num_epochs:
                error_vector = np.append(error_vector,sess.run(total_error,feed_dict={X:trainData[start_index:end_index],y:trainTarget[start_index:end_index]}))            
        
    #Plot the graph
    plt.plot(range(len(error_vector)),error_vector)
    num_epochs = num_iterations/batch_size
    plt.axis([0,num_epochs,0,np.max(error_vector)])
    plt.show()

    print("Final error: ",error_vector[-1])
    
    #No return values
    return

        
if __name__ == "__main__":

    linear_regression_basic()












