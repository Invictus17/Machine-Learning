import numpy as np
import math
import random

class NeuralNetwork(object):
    def __init__(self, epochs, learning_rate, rho, dropout):
        self.epochs = epochs
        self.learning_rate = learning_rate 
        self. learning_rate = learning_rate
        self.dropout = dropout
        self.rho = rho

    def fit(self, x_train, y_train):
        #citation:https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        #For all weight initialisations I used the last technique mentioned in this article
                    
        self.weights_1 = np.random.randn(784,512)*np.sqrt(2/(784 + 512))

        self.bias_1 = np.random.rand(512)
        self.bias_1 = self.bias_1.reshape(1, -1)
            
        self.weights_2 = np.random.randn(512,512)*np.sqrt(2/(512 + 512))
        
        self.bias_2 = np.random.rand(512)
        self.bias_2 = self.bias_2.reshape(1, -1)
        
        self.weights_3= np.random.randn(512,10)*np.sqrt(2/(512 + 10))
        
        self.bias_3 = np.random.rand(10)
        self.bias_3 = self.bias_3.reshape(1, -1)

        loss = 0
        
        batch_size = 60

        v_W1 = 0
        v_W2 = 0
        v_W3 = 0

        v_B1 = 0
        v_B2 = 0
        v_B3 = 0

        #Adding epsilon to prevent underflow at theta updates
        #Reference Andrew NG DL course RMSprop video
        epsilon = 10**-8
        rho = self.rho
        dropout = self.dropout
        alpha = self.learning_rate
        
        for m in range(self.epochs):
            print("Epoch:", m)
            L = 0
            for i in range(0,x_train.shape[0],batch_size):
                dW1 = 0
                dW2 = 0
                dW3 = 0
                dB1 = 0
                dB2 = 0
                dB3 = 0
                loss = 0
                
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                
                #Each image is a 28x28 gray scale image, thus the number of input neurons is 28*28 = 784, we have 60,000 such inputs
                ##########################Input layer to first hidden layer
                #Input matrix
                # print("Input Matrix:",x_batch.shape)

                #W1
                #Number of weights = 784(input layer) = 784*512(hidden layer)
                # print("weights 1:",self.weights_1.shape)

                #Calculate Z1
                Hidden_layer = np.matmul(x_batch, self.weights_1)
                # print("First hidden layer:", Hidden_layer.shape)

                #Adding Bias B1
                # print("bias 1 shape:",self.bias_1.shape)
                Hidden_layer = Hidden_layer + self.bias_1
                # print("after adding bias 1:", Hidden_layer.shape)
                #Applying relu activation to first hidden layer sums 
                #citing:https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
                Hidden_layer = np.maximum(Hidden_layer,0)

                # print("A1:",Hidden_layer.shape)
                # Dropout at first hidden layer
                neurons_to_exclude = math.floor(self.dropout*512)
                list_neurons_to_exclude = random.sample(range(512), neurons_to_exclude)

                for row in range(batch_size):
                    for neuron in list_neurons_to_exclude:
                        Hidden_layer[row,neuron] = 0
                
                #########################################################################################
                #First hidden layer to next hidden layer
                #Weights
                #Number of weights = 512(first hidden layer)              
                # print("weights 2:",self.weights_2.shape)

                #Z2 
                Hidden_layer_2 = np.matmul(Hidden_layer, self.weights_2)
                # print("Hidden layer 2:",Hidden_layer_2.shape)

                #Adding Bias B2
                # print("bias 2 shape:",self.bias_2.shape)
                Hidden_layer_2 = Hidden_layer_2 + self.bias_2
                # print("After adding bias 2:", Hidden_layer_2.shape)

                #Applying relu activation to second hidden layer
                #citing:https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
                Hidden_layer_2 = np.maximum(Hidden_layer_2,0)
                # print("A2:", Hidden_layer_2.shape)
                #Dropout at second hidden layer
                neurons_to_exclude = math.floor(self.dropout*512)
                list_neurons_to_exclude = random.sample(range(512), neurons_to_exclude)
                for row in range(batch_size):
                    for neuron in list_neurons_to_exclude:
                        Hidden_layer_2[row,neuron] = 0

                #####################################################################################
                #Hidden layer to output layer
                #Weights
                #Number of weights = 512(second hidden layer) = 512*10(output layer)
                # print("weights 3:",self.weights_3.shape)

                #Z3
                Output_layer = np.matmul(Hidden_layer_2, self.weights_3)
                # print("Output layer shape:", Output_layer.shape)

                #Adding Bias B3              
                # print("bias 3 shape:",self.bias_3.shape)
                Output_layer = Output_layer + self.bias_3
                # print("Afer adding bias 3:", Output_layer.shape)

                #Applying softmax activation to output sums
                # Output_layer = Output_layer.reshape(-1,1)
                # #citing : https://stats.stackexchange.com/questions/304758/softmax-overflow

                m = np.amax(Output_layer, axis = 1).reshape(-1,1)
                Output_layer = Output_layer - m
                
                Output_layer = np.exp(Output_layer)
                sum = np.sum(Output_layer, axis=1).reshape(-1,1)
                # print("sum", sum.shape)
                Output_layer = np.divide(Output_layer,sum)

                # print("Output:", Output_layer.shape)
                # print("Output_layer:",np.sum(Output_layer, axis=1))

                #Calculate loss
                y_true = y_batch
                # y_true = y_true.reshape(1,-1)
                # print("y_true:",y_true.shape)
                loss = np.sum(-np.multiply(y_true,np.log(Output_layer)), axis=1)
                loss = loss.reshape(-1,1)
                loss = np.sum(loss)/batch_size
                # print(loss.shape)
                L +=loss.copy()
                
                #Backpropogation
                #dW3
                dZ3_dW3 = Hidden_layer_2.copy()
                # print("dZ3_dW3:", dZ3_dW3.shape)
                dL_dZ3 = Output_layer - y_true
                # print("dL_dZ3:",dL_dZ3.shape)
                dW3 += np.matmul(dZ3_dW3.T, dL_dZ3)
                # print("Weight 3 derivative or dW3:", dW3.shape)    

                #dB3
                dZ3_dB3 = np.ones(batch_size).reshape(1,-1)
                # print("dZ3_dB3:", dZ3_dB3.shape)

                dB3 += np.matmul(dZ3_dB3, dL_dZ3)
                # print("Bias 3 derivative or dB3:",dB3.shape)

                #V_W3
                v_W3 = rho*v_W3 + (1-rho)*np.square(dW3)
                v_B3 = rho*v_B3 + (1-rho)*np.square(dB3)


                #dw2 
                dZ2_dW2 = Hidden_layer.copy()
                # print("dZ2_dW2:", dZ2_dW2.shape)
                dA2_dZ2 = Hidden_layer_2.copy()
                dA2_dZ2[dA2_dZ2>0] = 1
                # print("dA2_dZ2:",dA2_dZ2.shape)
                dZ3_dA2 = self.weights_3.copy()
                # print("dZ3_dA2:",dZ3_dA2.shape)

                dW2 += np.matmul(np.multiply(np.matmul(dZ3_dA2,dL_dZ3.T), dA2_dZ2.T), dZ2_dW2)
                # print("Weight 2 derivate or dW2", dW2.shape)

                #dB2
                dZ2_dB2 = np.ones(batch_size).T.reshape(1,-1)
                # print("dZ2_dB2:", dZ2_dB2.shape)
                
                dB2 += np.matmul(dZ2_dB2, np.multiply(np.matmul(dZ3_dA2,dL_dZ3.T), dA2_dZ2.T).T)
                # print("Bias 2 derivative or dB2:", dB2.shape)

                #V_W2
                v_W2 = rho*v_W2 + (1-rho)*np.square(dW2)
                v_B2 = rho*v_B2 + (1-rho)*np.square(dB2)



                #dW1
                dZ1_dW1 = x_batch.copy()
                # print("dZ1_dW1:", dZ1_dW1.shape)
                dA1_dZ1 = Hidden_layer.copy()
                dA1_dZ1[dA1_dZ1>0] = 1
                # print("dA1_dz1:", dA1_dZ1.shape)
                dZ2_dA1 = self.weights_2.copy()
                # print("dZ2_dA1:", dZ2_dA1.shape)

                dW1 += np.matmul(dZ1_dW1.T, np.multiply(dA1_dZ1, np.matmul(dZ2_dA1, np.multiply(np.matmul(dL_dZ3, dZ3_dA2.T), dA2_dZ2).T).T))
                # print("Weight 1 derivate or dW1", dW1.shape)

                dZ1_dB1 = np.ones(batch_size).reshape(1,-1)
                # print("dZ1_dB1:", dZ1_dB1.shape)
                dB1 += np.matmul(dZ1_dB1, np.multiply(dA1_dZ1, np.matmul(dZ2_dA1, np.multiply(np.matmul(dL_dZ3, dZ3_dA2.T), dA2_dZ2).T).T))
                # print("Bias 1 derivative or dB1:", dB1.shape)

                #V_W1
                v_W1 = rho*v_W1 + (1-rho)*np.square(dW1)
                v_B1 = rho*v_B1 + (1-rho)*np.square(dB1)
                
                
                #Update weights
                self.weights_1 = self.weights_1 - alpha*((dW1/batch_size)/(np.sqrt(v_W1) + epsilon))

                self.weights_2 = self.weights_2 - alpha*((dW2/batch_size)/(np.sqrt(v_W2) + epsilon))

                self.weights_3 = self.weights_3 - alpha*((dW3/batch_size)/(np.sqrt(v_W3) + epsilon))

                #Update bias
                self.bias_1 = self.bias_1 - alpha*((dB1/batch_size)/(np.sqrt(v_B1) + epsilon))

                self.bias_2 = self.bias_2 - alpha*((dB2/batch_size)/(np.sqrt(v_B2) + epsilon))

                self.bias_3 = self.bias_3 - alpha*((dB3/batch_size)/(np.sqrt(v_B3) + epsilon))
                
            print("loss:",L, "accuracy:", self.evaluate(x_train, y_train))

    def evaluate(self, x_Input, y_Input):
        correct_count = 0
        for i in range(x_Input.shape[0]):
            #####################################################Input layer to first hidden layer
            #Input matrix
            first_image = x_Input[i]
            first_image = first_image.reshape(1,-1)
            # print("Input Matrix:",first_image.shape)

            #Calculate Z1
            Hidden_layer = np.matmul(first_image, self.weights_1)
            # print("First hidden layer:", Hidden_layer.shape)

            #Adding Bias B1
            # print("bias 1 shape:",bias_1.shape)
            Hidden_layer = Hidden_layer + self.bias_1

            #Applying relu activation to first hidden layer sums 
            #citing:https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
            Hidden_layer = np.maximum(Hidden_layer,0)

            #########################################################################################
            #First hidden layer to next hidden layer
            # print("weights 2:",self.weights_2.shape)

            #Z2
            #Sum all outputs 
            Hidden_layer_2 = np.matmul(Hidden_layer, self.weights_2)
            # print("Hidden layer 2:",Hidden_layer_2.shape)

            # print("bias 2 shape:",self.bias_2.shape)
            Hidden_layer_2 = Hidden_layer_2 + self.bias_2


            #Applying relu activation to second hidden layer
            #citing:https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
            Hidden_layer_2 = np.maximum(Hidden_layer_2,0)

            #####################################################################################
            #Hidden layer to output layer
            # print("weights 3:",self.weights_3.shape)

            #Z3
            Output_layer = np.matmul(Hidden_layer_2, self.weights_3)
            # print("Output layer shape:", Output_layer.shape)

            #Adding Bias B3

            # print("bias 3 shape:",self.bias_3.shape)
            Output_layer = Output_layer + self.bias_3

            #Applying softmax activation to output sums
            Output_layer = Output_layer.reshape(-1,1)
            #citing : https://stats.stackexchange.com/questions/304758/softmax-overflow


            m = max(Output_layer)
            Output_layer = Output_layer - m
            Output_layer = np.exp(Output_layer)
            sum = np.sum(Output_layer)
            Output_layer = Output_layer/sum

            y_true = y_Input[i]
            if np.argmax(y_true) == np.argmax(Output_layer):
                correct_count+=1

        #Accuracy
        accuracy = (correct_count/x_Input.shape[0])*100
        return accuracy
