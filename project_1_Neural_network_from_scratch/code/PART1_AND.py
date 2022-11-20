
#miniproject PART1 - a neuron should learn logical and operation 
#-----------------IMPORTS-------------------
import Neuron
import Trainer
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------ 

#first randomly initilized weights and set bias to 1
w1 = np.random.randn()
w2 = np.random.randn()
w = [w1,w2]
b = 1
and_training_set ={(0,0):0,(0,1):0,(1,0):0,(1,1):1}

and_neuron = Neuron.LogicGateNeuron(w,b) #create an instance of LogicGateNeuron

#using a trainer to train the and_neuron !
epochs = 50
and_trainer = Trainer.Trainer(and_neuron,and_training_set)
and_trainer.train(epochs,lr = 2)

#plotting training erro vs epochs:---------------------------
plt.plot(range(0,epochs),and_trainer.trainerrList)
plt.axis([0 , epochs , 0 , 0.5])
plt.title("Traning ERROR for and_neuron")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.show()
#-------------------------------------------------------------
acc = and_trainer.trainerrList[epochs-1] #accuracy
print(acc)





