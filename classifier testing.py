import sys
import numpy as np
import nnfs
import math
from nnfs.datasets import spiral_data
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import glob

nnfs.init()

## DATA
image_list = []


img0 = mpimg.imread("C:/Users/riyad/OneDrive/Desktop/Neural network/test drive/left_img.jpg") # left

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.2989, 0.5870, 0.1140]))

gray0 = rgb2gray(img0)


for filename in glob.glob('yourpath/*.gif'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

X = [gray0.transpose()]


#plt.imshow(gray, cmap=plt.get_cmap('gray'))
#plt.show()


##WEIGHTS

## BEGNNING OF NEW NEURON IN PATH
#how best tune weights and bias to achieve desired output



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) ## weights
        self.biases = 0.10*np.random.randn(1,n_neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Convolutional_Filter:
    def __init__(self,width,height):
        self.weights = 0.10*np.random.randn(width,height) # weights
        self.bias = 0.10*np.random.randn(1,1)
        self.width = width
        self.height = height
    def forward(self,input):
        input = input.astype(np.float32)/10000000.0
        x = -1
        feature_map = [[]]
        
        for i in range(0,len(input)-1-self.width,64): # rows
            x+=1
            for j in range(0,(len(input[i])-1-self.height),36): # column
                section = input[i:i+self.width,j:j+self.height]
                convulsion = np.sum(np.inner(section,self.weights)) + self.bias
                convulsion = float(convulsion)
                feature_map[x].append(convulsion)
            feature_map.append([])
        feature_map.pop()
        npft_map = np.asarray(feature_map)
        self.output = npft_map

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,10^-7,(1-(10^-7)))



        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def filter2(input_array): # searches 5by5 grid

    final_array = []
    for i in range(0,len(input_array)-1,5):
        for j in range(0,(len(input_array[i])-1),5):
            section = input_array[i:i+5,j:j+5]
            nmax = np.max(section)
            final_array.append(nmax)
    final_array = np.array(final_array)
    return final_array

def derive(y2,y1,dx):
    return (y2-y1)/dx


lossfx = Loss_CategoricalCrossentropy()



def initialize(layers):
    pass

## vars

denses = []
activations = []

## initialize layers

denses.append(Layer_Dense(9,128))
denses.append(Layer_Dense(128,256))
denses.append(Layer_Dense(256,128))
denses.append(Layer_Dense(128,3))

activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_Softmax())


precision = 0.01
## training functions
one_hot = np.array([0,1]) # EXPECTED OUTPUT, 0 LEFT, 1 CENTER, 2 RIGHT
def run(input):

    for i in range(0,len(denses)):
        if i == 0:
            denses[i].forward(input)
        else:
            denses[i].forward(activations[i-1].output)
        activations[i].forward(denses[i].output)
    loss = lossfx.calculate(activations[len(denses)-1].output,one_hot)
    return loss

def find_gradient(medium,input):
    big_gradient = []

    for i in range(0,len(medium)):
        micro_gradient = []
        for j in range(0,len(medium[i])):
            loss1 = run(input)
            medium[i][j]+=precision
            loss2 = run(input)
            medium[i][j]-=precision
            diff = derive(loss2,loss1,precision)
            micro_gradient.append(diff)
        big_gradient.append(micro_gradient)
    return big_gradient

def adjust_val(medium,input):
    d1wgrad = find_gradient(medium,input)
    for h in range(0,len(medium)):
        for l in range(0,len(medium[h])):
            original = medium[h][l]
            lose1 = run(input)
            if d1wgrad[h][l] > 0:
                medium[h][l]-=precision
            elif d1wgrad[h][l] < 0:
                medium[h][l]+=precision
            lose2 = run(input)
            if lose2 >= lose1:
                medium[h][l] = original

last_loss = 0

while True:
    iteration = -1
    losses = []
    for imagedata in X:
        iteration+=1
        filter1 = Convolutional_Filter(960,540)
        filter1.forward(imagedata)
        activation1 = Activation_ReLU()
        activation1.forward(filter1.output)

        activation2 = Activation_Softmax()
        max_pool = filter2(activation1.output) ## NEW INPUT

        adjust_val(denses[0].weights,max_pool)
        adjust_val(denses[0].biases,max_pool)
        losses.append(run(max_pool))
    losses = np.array(losses)
    print(np.mean(losses))
    




#### TRAINING ########################################################

