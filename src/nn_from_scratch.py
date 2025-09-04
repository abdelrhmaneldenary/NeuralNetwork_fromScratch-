import numpy as np
'''' 
okay we want to calculate the dervative of the loss with respect to w but we cannot do that directly we will do that using chain rule through three
steps 
(dl dervative of the losss, da dervative of the input after passing through activation function, dz dervative of the input after passin through the weights and biases)
1-dl_da the dervative of the loss with respect to the output(y_predc)
2-da_dz the dervative of the activation function with respect to z
3-dz_dw the dervative of the output through weights with respect to the weights
by appling chain rule dl_dw=dl_da*da_dz*dz_dw (wich is what we want)
(but there is some special cases we will get the dervative along without the chain rule if there is direct formula or something wihch is easier).
AND OFCOURSE ALL OF THAT TO GET THE GRADIENT DESCENT IN UPDATING THE WEIGHTS IN THE BACK PROPAGATION 
4-W=W-LR*DL_DW
4-B=B-LR*DL_DB
which are the final steps in the backpropagation to update the weights
'''
# -------------------------
# Activation functions
# -------------------------
# the dervatice of these function is consider step 2 da_dz
# the dimesnions of the output is (m,units) where m is the number of training examples and units the number of units per layer 
#and that if we assume z and a are in (m,units) format wich is the case   

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derv_sigmoid(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def derv_relu(a):
    return (a > 0).astype(float)

def linear(z):
    return z

def derv_linear(z):
    return np.ones_like(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability(as if z is larger than 1000 e^z will be so big (inf) so we make sure that e^z will be small)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)    # i kept the dimension for clean broadcasting in the next steps wich will be used in matrix mult

# -------------------------
# Losses
# -------------------------
# the dervative of these functions is considerd step 1 dl_da 
# a =y_pred (the same but different names)
# the dimesnions of the output of the dervative is (m,units) where m is the number of training examples and units the number of units per layer
# and the output of the loss is one number 
# the next steps of multiplication of chain rule 
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def derv_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def derv_binary_crossentropy(y_true, y_pred): #this is considerd special case dl_dz forward not dl_da
    m = y_true.shape[0]
    return (y_pred - y_true) / m

def sparse_categorical_crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    m = y_pred.shape[0]
    return -np.mean(np.log(y_pred[np.arange(m), y_true]))

def derv_sparse_categorical_crossentropy(y_true, y_pred):# i writed it here and calculated it just for the sake of consistent but in the use of gradient ddescent i used dl_dz directly
    # eps = 1e-8
    # y_pred = np.clip(y_pred, eps, 1 - eps)
    m = y_pred.shape[0]
    grad = np.zeros_like(y_pred)
    grad[np.arange(m), y_true] = -1 / (y_pred[np.arange(m), y_true] * m)
    return grad


"""
(dl_da)the product of these function will get you a numbers for each training example(value for each (a=y_pred))  and in 
the next step of partial dervative da_dz is calculated for each training example and for each neuron
so at this moment when we multiply them we have dl_dz for each neuron in the layes and for each triaining example
so what we need now is dl_dw for each weight in each neuron in the layer in each training example
wich will we get in the last step dz_dw but the result of each training example will be averaged and we will have the update in weights
for each weight in each neuron 
"""

#the step 1 is easy dz_dw wich is suggest multlying the inputs of the layer z=wx+b  dz_dw=x (of course for each input and each neuron)
# now we will just apply forward propagation to compute the loss and then apply the backpropagation to compute the gradient descent using the chain rule 
#then update the weightts 



# -------------------------
# Neural Network Class
# -------------------------
class NeuralNetwork:
    """
    Fully-connected network with configurable layers.
    layers_config example:
      [
          {"units": input_dim, "activation": None},
          {"units": 64, "activation": "relu"},
          {"units": 32, "activation": "relu"},
          {"units": 1, "activation": "sigmoid"}
      ]
    """

    def __init__(self, layers_config, loss, lr=0.001, seed=42):
        np.random.seed(seed)
        self.layers_config = layers_config
        self.loss_name = loss
        self.lr = lr

        self.activations = [cfg["activation"] for cfg in layers_config[1:]]

        # initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layers_config) - 1):
            in_dim = layers_config[i]["units"]
            out_dim = layers_config[i + 1]["units"]
            W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / max(1, in_dim))  # a matrix in (in_dim,out_dim) dimensions 
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

    # -------------------------
    # Forward Pass
    # -------------------------

    #the function f(z)=a  z is tthe output of the input after xw+b
    def _activate(self, z, act_name):
        if act_name == "relu":
            return relu(z)
        elif act_name == "sigmoid":
            return sigmoid(z)
        elif act_name == "linear" or act_name is None:
            return z
        elif act_name == "softmax":
            return softmax(z)
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    #X here is the input and it must be in the form of (m,input_units) where m is the training examples and in_units is the dimension of the input layer
    #the output will be (m,out_units)
    def forward(self, X):
        self.z_list = []
        self.a_list = [X.astype(float)]
        for W, b, act in zip(self.weights, self.biases, self.activations):
            z = np.dot(self.a_list[-1], W) + b
            a = self._activate(z, act)
            self.z_list.append(z)
            self.a_list.append(a)
        return self.a_list[-1]

    # -------------------------
    # Loss
    # -------------------------
    def compute_loss(self, y_true, y_pred):
        if self.loss_name == "binary_crossentropy":
            return binary_crossentropy(y_true, y_pred)
        elif self.loss_name == "mse":
            return mse(y_true, y_pred)
        elif self.loss_name == "sparse_categorical_crossentropy":
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

    #da_dz
    def _activation_derivative_from_a(self, a, act_name):
        if act_name == "relu":
            return derv_relu(a)
        elif act_name == "sigmoid":
            return derv_sigmoid(a)
        elif act_name == "linear" or act_name is None:
            return derv_linear(a)
        elif act_name == "softmax":
            return None  # handled separately
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    # -------------------------
    # Backward Pass
    # -------------------------
    def backward(self, y_true):
        m = self.a_list[0].shape[0]
        l = len(self.weights)
        grads_w = [None] * l
        grads_b = [None] * l

        al = self.a_list[-1]

        if self.loss_name == "binary_crossentropy":
            dl_da = derv_binary_crossentropy(y_true, al)
        elif self.loss_name == "mse":
            dl_da = derv_mse(y_true, al)
        elif self.loss_name == "sparse_categorical_crossentropy":
            dl_da = derv_sparse_categorical_crossentropy(y_true, al)
        else:
            raise ValueError(f"Unsupported loss {self.loss_name}")

        last_act = self.activations[-1]
        if last_act == "softmax" and self.loss_name == "sparse_categorical_crossentropy":
            dz = al.copy()
            dz[np.arange(m), y_true] -= 1
            dz = dz / m

        elif last_act=='sigmoid' and self.loss_name=='binary_crossentropy"':
            dz=derv_binary_crossentropy(y_true, al)
        
        else:
            act_deriv = self._activation_derivative_from_a(al, last_act)
            dz = dl_da * act_deriv  #element wise multiplication this time not the standared np.dot multiplication so the output can be in the form of (m,units_out)

        for i in reversed(range(l)):
            '''
            i will explain this part in alittle bit of details because it is hard to get but donot worry denary got u
            l is the number of weights so we will iterate in this lenght to update the weights we got and we 
            use reversed because in backpropagation the dervative chain rule is working from the last layer to the first layer.

            the second thing is the dimensions and the multiplication of the gradinets of the weights 
            a_prev is the inputs of the layer(x1,x2,x3,..) dimensions is (m,units1) where units 1 is the lenghts of the inputs (x1,..)
            dz (m,units2) where units2 is the number of neurons
            a_prev.T (units1,m) dz(m,units2) the result is (units1,units2) where the m dissapeared because we used dot product and (1/m)
            to get the average of the update of each weight and each neuron from all the training example
            the output is (units1,units2)=[[neuron1_w1,neuron2_w1],
                                           [neuron1_w2,neuron2_w2]]
            wich is the form we want and use to update the weights as we save the weights in this form (weights,neurons)
            '''
            a_prev = self.a_list[i]
            grads_w[i] = (1 / m) * np.dot(a_prev.T, dz)
            grads_b[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            #at this point we get the update of the last layer and we will processed in this process to get the update of the other layers
            #in the other iterations
            #but before that we need to update the dz that will be used in the other layers wich will be done below 
            #as we have dl_dz3 we need to 
            if i > 0:
                dz = np.dot(dz, self.weights[i].T)# here the output will be (m,units1) where units 1 is the input of the last layer
                #but it is the output of the previous layer which is what we want 
                #we have dz (dl_dz1) and we want to get dl_dz2 from it so what we did is since z1=wx(a_prev)+b
                #so we multiplied dl_dz1 by the weights wich is dz1_da 
                #an then multiplied it by the previous activation da_dz2 
                #so we got dl_dz1*dz1_da*da_dz2=dl_dz2 which we can use in the upcoming layers calculation
                prev_act = self.activations[i - 1]
                act_deriv_prev = self._activation_derivative_from_a(self.a_list[i], prev_act)
                if act_deriv_prev is None:
                    raise RuntimeError("Activation derivative unavailable.")
                dz = dz * act_deriv_prev

        # update step
        #here we update the weights of each layer
        for i in range(l):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    # -------------------------
    # Training (with mini-batch)
    # -------------------------
    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            # shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # iterate over batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]

                y_pred = self.forward(X_batch)
                self.backward(y_batch)

            if verbose and (epoch == 1 or epoch % 10 == 0):
                y_pred_full = self.forward(X)
                loss = self.compute_loss(y, y_pred_full)
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.6f}")

    # -------------------------
    # Predicsrc/nn_from_scratch.pytion
    # -------------------------
    def predict(self, X, threshold=0.5):
        a = self.forward(X)
        last_act = self.activations[-1]
        if last_act == "sigmoid":
            return (a >= threshold).astype(int)
        elif last_act == "softmax":
            return np.argmax(a, axis=1)
        else:
            return a
