
#this class takes a LogicGateNeuron and trains it!
#------------------------------------------------------------------------

#---------------IMPORTS------------
import Neuron
#----------------------------------


class Trainer():
    
    def __init__(self,neuron:Neuron.LogicGateNeuron,training_set:dict):
        self.neuron = neuron
        self.training_set = training_set
        self.trainingerr = 0
        self.testerr = 0
        self.trainerrList = list()
        
    
    #lr : learning rate
    def GD(self,lr):
        
        self.trainingerr = 0
        num = len(self.training_set) #number of data poins in training_set
        for input,target in self.training_set.items():

            self.neuron.activate(input)
            self.trainingerr += (1/num) * (self.neuron.output - target)**2 #evaluating mean square error (MSE)
           
           #updates weights and bias of the neuron-------------------------------------------------------------
            self.neuron.weight[0] -= lr * input[0] * self.neuron.activate_prime * (self.neuron.output - target)
            self.neuron.weight[1] -= lr * input[1] * self.neuron.activate_prime * (self.neuron.output - target)
            self.neuron.bias -= lr * self.neuron.activate_prime * (self.neuron.output - target)
            #--------------------------------------------------------------------------------------------------

    #epochs : number of iteration over all provided training examples
    #lr : learning rate
    def train(self,epochs,lr):
        while(epochs > 0):
            self.GD(lr)
            self.trainerrList.append(self.trainingerr)
            epochs -=1
    #------------------------------------------------------------------------------------

#-------------testing a trained neuron over new set of data points------------------
    def test(self,test_set:dict):
        num = len(test_set)
        self.testerr = 0
        for input,target in test_set.items():
            self.neuron.activate(input)
            self.testerr += (1/num) * (self.neuron.output - target)**2
         
        
    #-----------------------------------------------------------------------------


        
        

  





    
    