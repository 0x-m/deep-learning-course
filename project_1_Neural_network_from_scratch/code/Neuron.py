
#--------------IMPORTS-----------------------------------
import Sigmoid as sigmoid
import numpy as np
#--------------------------------------------------------

#abstract model for a neuron-------------------------------------
class Neuron():
   
    def __init__(self,weight,bias):
        self.weight = weight
        self.bias = bias


    #every concrete neuron implements this function in a specific manner!
    def activate(self,input): #some nonlinear operation
        pass 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#in this miniproject I used a special neuron that takes two inputs and returns one output
#LogicGateNeuron is a model for a Logic Gate!
class LogicGateNeuron(Neuron):

    def __init__(self,weight:list,bias:float):
        if(len(weight)!=2):
            raise Exception("weight muse be a 2d vector!")
        self.weight = weight
        self.bias = bias

    
    def linear_combinator(self,input):
        return np.dot(input,self.weight) + self.bias  # z = w1*x1 + w2*x2 + b
        

    #I injected inputs to the neuron through activate function 
    def activate(self,input):
        if(len(input) != 2):
            raise Exception("input muse be a 2d vector!")

        z = self.linear_combinator(input) 

        self.output = sigmoid.sigmoid_func(z) #applying sigmoid function as neuron activation 

        self.activate_prime = self.output * (1-self.output) # derivative of the activation function at current input
        
        return self.output

#-------------------------------------------------------------