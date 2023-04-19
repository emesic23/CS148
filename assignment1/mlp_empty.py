import numpy as np
from datasets import *
from random_control import *
from losses_empty import *


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def softmax(self, inputs):
        # TODO (Part 2)
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        softmaxes = exps / np.sum(exps, axis=1, keepdims=True)
        return softmaxes

    def tanH(self, inputs):
        return np.tanh(inputs)

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def relu(self, inputs):
        # TODO (Part 5)
        rel = np.maximum(0, inputs)
        return rel

    def tanH_derivative(self, Z):
        # TODO (Part 5)
        der = 1 - np.power(np.tanh(Z), 2)
        return der

    def sigmoid_derivative(self, Z):
        sig = self.sigmoid(Z)
        der = sig * (1 - sig)
        return der

    def relu_derivative(self, Z):
        # TODO (Part 5)
        der = (Z > 0).astype(int)
        return der

    def apply_chain_rule_activation_derivative(self, q, activation_derivative):
        # TODO rename the variable q appropriately -- what should this be?
        return None

    def forward(self, inputs, weights, bias, activation):
        Z_curr = np.dot(inputs, weights.T) + bias

        if activation == 'relu':
            A_curr = self.relu(inputs=Z_curr)
        elif activation == 'sigmoid':
            A_curr = self.sigmoid(inputs=Z_curr)
        elif activation == 'tanH':
            A_curr = self.tanH(inputs=Z_curr)
        elif activation == 'softmax':
            A_curr = self.softmax(inputs=Z_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):

        # TODO each of these functions require you to compute all of the colored terms in the Part 2 Figure.
        # We will denote the partial derivative of the loss with respect to each variable as dZ, dW, db, dA
        # These variable map to the corresponding terms in the figure. Note that these are matrices and not individual
        # values, you will determine how to vectorize the code yourself. Think carefully about dimensions!
        # You can use the self.apply_chain_rule_activation_derivative() function, although there are solutions without it.
        '''
        The inputs to this function are:
            dA_curr - the partial derivative of the loss with respect to the activation of the preceding layer (l + 1).
            W_curr - the weights of the layer (l)
            Z_curr - the weighted sum of layer (l)
            A_prev - the activation of this layer (l) ... we use prev with respect to dA_curr

        The outputs are the partial derivatives with respect
            dA - the activation of this layer (l) -- needed to continue the backprop
            dW - the weights -- needed to update the weights
            db - the bias -- (needed to update the bias
        '''

        if activation == 'softmax':
            # We deal with the softmax function for you, so dZ is not needed for this one. dA_curr = dZ for this one.
            dA = dA_curr
            dW = np.dot(A_prev.T, dA_curr)
            db = np.sum(dA_curr, axis=0, keepdims=True)
            dA = np.dot(dA, W_curr)
        elif activation == 'sigmoid':
            # Computing dZ is not technically needed, but it can be used to help compute the other values.
            activation_derivative = self.sigmoid_derivative(Z_curr)
            dZ = dA_curr * activation_derivative
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, W_curr)
        elif activation == 'tanH':
            activation_derivative = self.tanH_derivative(Z_curr)
            dZ = dA_curr * activation_derivative
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, W_curr)
        elif activation == 'relu':
            activation_derivative = self.relu_derivative(Z_curr)
            dZ = dA_curr * activation_derivative
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, W_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return dA, dW, db

'''
* `MLP` is a class that represents the multi-layer perceptron with a variable number of hidden layer. 
   The constructor initializes the weights and biases for the hidden and output layers.
* `sigmoid`, `relu`, `tanh`, and `softmax` are activation function used in the MLP. 
   They should each map any real value to a value between 0 and 1.
* `forward` computes the forward pass of the MLP. 
   It takes an input X and returns the output of the MLP.
* `sigmoid_derivative`, `relu_derivative`, `tanH_derivative` are the derivatives of the activation functions. 
   They are used in the backpropagation algorithm to compute the gradients.
*  `mse_loss`, `hinge_loss`, `cross_entropy_loss` are each loss functions.
   The MLP algorithms optimizes to minimize those.
* `backward` computes the backward pass of the MLP. It takes the input X, the true labels y, 
   the predicted labels y_hat, and the learning rate as inputs. 
   It computes the gradients and updates the weights and biases of the MLP.
* `train` trains the MLP on the input X and true labels y. It takes the number of epochs 
'''

class MLP:
    def __init__(self, layer_list):
        '''
        Arguments
        --------------------------------------------------------
        layer_list: a list of numbers that specify the width of the hidden layers. 
               The dataset dimensionality (input layer) and output layer (1) 
               should not be specified.
        '''
        self.layer_list = layer_list
        self.network = []  ## layers
        self.architecture = []  ## mapping input neurons --> output neurons
        self.params = []  ## W, b
        self.memory = []  ## Z, A
        self.gradients = []  ## dW, db
        self.all_gradients = [] ## copy of all self.gradients for each epoch
        self.loss = []
        self.accuracy = []

        self.loss_func = None
        self.loss_derivative = None

        self.init_from_layer_list(self.layer_list)
        self._set_loss_function()

    # TODO read and understand the next several functions, you will need to understand them to complete the assignment.
    #  In particular, you will need to understand
    #  self.network, self.architecture, self.params, self.memory, and self.gradients. It may be helpful to write some
    #  notes about what each of these variables are and how they are used.
    def init_from_layer_list(self, layer_list):
        for layer_size in layer_list:
            self.add(Layer(layer_size))

    def add(self, layer):
        self.network.append(layer)

    def _compile(self, data, activation_func='relu'):
        self.architecture = [] 
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim': data.shape[1], 'output_dim': self.network[idx].num_neurons,
                                          'activation': activation_func})
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': activation_func})
            else:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': 'softmax'})
        return self

    def _init_weights(self, data, activation_func, seed=None):
        self.params = []
        self._compile(data, activation_func)

        if seed is None:
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})
        else:
            # For testing purposes
            fixed_generator = np.random.default_rng(seed=seed)
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': fixed_generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})

        return self

    def forward(self, data):
        A_prev = data
        A_curr = None

        for i in range(len(self.params)):
            W_curr = self.params[i]['W']
            b_curr = self.params[i]['b']
            activation = self.architecture[i]['activation']

            A_curr, Z_curr = self.network[i].forward(A_prev, W_curr, b_curr, activation)

            mem_dict = {'A_prev': A_prev, 'W_curr': W_curr, 'A_curr': A_curr, 'Z_curr': Z_curr, 'activation': activation}
            self.memory.append(mem_dict)

            A_prev = A_curr

        return A_curr


#def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
    def backward(self, predicted, actual):
        # compute the gradient on predictions
        dscores = self.loss_derivative(predicted, actual)
        dA_prev = dscores  # This is the derivative of the loss function with respect to the output of the last layer

        self.gradients = []
        for i in reversed(range(len(self.params))):
            # get the previous layer's activation values from memory
            A_prev = self.memory[i]['A_prev']
            A_curr = self.memory[i]['A_curr']
            W_curr = self.memory[i]['W_curr']
            Z_curr = self.memory[i]['Z_curr']
            activation = self.memory[i]['activation']


            # compute dZ, dW, db for this layer
            dA, dW, db = self.network[i].backward(dA_prev, W_curr, Z_curr, A_prev, activation)

            # save the gradients in a dictionary and append to self.gradients
            grad_dict = {'dW': dW, 'db': db}
            self.gradients.append(grad_dict)

            # update dA_prev for the next layer
            dA_prev = dA

        # append this epoch's gradients to all_gradients
        self.all_gradients.append(self.gradients)

        # print(self.gradients)
        

    def _update(self, lr):
        # TODO update the network parameters using the gradients and the learning rate.
        #  Recall gradients is a list of dicts, and params is a list of dicts, pay attention to the order of the dicts.
        #  Is gradients the same order as params? This might depend on your implementations of forward and backward.
        #  Should we add or subtract the deltas?
        self.gradients = self.gradients[::-1]
        for i in range(len(self.params)):
            # Update weight and bias parameters using their gradients
            self.params[i]['W'] -= lr * self.gradients[i]['dW'].T
            self.params[i]['b'] -= lr * self.gradients[i]['db']

    # Loss and accuracy functions
    def _calculate_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _calculate_loss(self, predicted, actual):
        return self.loss_func(predicted, actual)

    def _set_loss_function(self, loss_func_name='negative_log_likelihood'):
        if loss_func_name == 'negative_log_likelihood':
            self.loss_func = negative_log_likelihood
            self.loss_derivative = nll_derivative
        elif loss_func_name == 'hinge':
            self.loss_func = hinge
            self.loss_derivative = hinge_derivative
        elif loss_func_name == 'mse':
            self.loss_func = mse
            self.loss_derivative = mse_derivative
        else:
            raise Exception("Loss has not been specified. Abort")

    def get_losses(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def train(self, X_train, y_train, epochs=1000, lr=1e-4, batch_size=16, activation_func='relu', loss_func='negative_log_likelihood'):

        self.loss = []
        self.accuracy = []

        # cast to int
        y_train = y_train.astype(int)

        # initialize network weights
        self._init_weights(X_train, activation_func)

        # TODO calculate number of batches
        num_datapoints = len(X_train)
        num_batches = int(num_datapoints / batch_size) + (num_datapoints % batch_size != 0)

        # NOTE: only reset all_gradients with every call to train
        self.all_gradients = [] # reset every time train is called

        # TODO shuffle the data and iterate over mini-batches for each epoch.
        #  We are implementing mini-batch gradient descent.
        #  How you batch the data is up to you, but you should remember shuffling has to happen the same way for
        #  both X and y.

        for i in range(int(epochs)):
            # shuffle the data
            shuffled_indices = np.random.permutation(num_datapoints)
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            batch_loss = 0
            batch_acc = 0
            for batch_num in range(num_batches - 1):

                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # TODO Hint: do any variables need to be reset each pass?

                # perform forward pass
                yhat = self.forward(X_batch)

                acc = self._calculate_accuracy(yhat, y_batch)
                loss = self._calculate_loss(yhat, y_batch)
                batch_acc += acc
                batch_loss += loss

                # perform backward pass
                self.backward(yhat, y_batch)

                # update the network
                self._update(lr)

                # save gradients for this batch
                self.all_gradients.append(self.gradients)

                # Stop training if loss is NaN, why might the loss become NaN or inf?
                if np.isnan(loss) or np.isinf(loss):
                    # s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                    # print(s)
                    print("Stopping training because loss is NaN")
                    self.loss.append(9999999)
                    return

                # TODO update the network

            self.loss.append(batch_loss / num_batches)
            self.accuracy.append(batch_acc / num_batches)

            if i % 20 == 0:
                s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                print(s)

    def predict(self, X, y, loss_func='negative_log_likelihood'):
        # Cast y to int
        y = y.astype(int)

        # Forward pass
        predicted = self.forward(X)

        # Calculate loss and accuracy
        loss = self._calculate_loss(predicted, y)
        accuracy = self._calculate_accuracy(predicted, y)

        # Handle NaN or inf loss
        if np.isnan(loss) or np.isinf(loss):
            print("Loss is NaN or inf.")
        else:
            print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))

        # Store loss and accuracy for plotting
        self.test_loss = loss
        self.test_accuracy = accuracy



if __name__ == '__main__':
    # Copy of part2.py in case useful for debugging
    N = 100
    M = 100
    dims = 3
    gaus_dataset_points = generate_nd_dataset(N, M, kGaussian, dims).get_dataset()
    X = gaus_dataset_points[:, :-1]
    y = gaus_dataset_points[:, -1].astype(int)

    model = MLP([3,2])
    model.train(X, y)
