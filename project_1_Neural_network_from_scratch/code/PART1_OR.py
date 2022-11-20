
#miniproject PART1 - a neuron should learn logical or operation 
#----------------IMPORTS-------------------
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

or_training_set ={(0,0):0,(0,1):1,(1,0):1,(1,1):1}
or_neuron = Neuron.LogicGateNeuron(w,b) #create an instance of LogicGateNeuron

#using a trainer to train the or_neuron !
epochs = 50
or_trainer = Trainer.Trainer(or_neuron,or_training_set)
or_trainer.train(epochs,lr = 2)

#plotting training erro vs epochs:---------------------------
plt.plot(range(0,epochs),or_trainer.trainerrList)
plt.axis([0 , epochs , 0 , 0.5])
plt.title("Traning ERROR for or_neuron")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.show()
#-------------------------------------------------------------
acc = or_trainer.trainerrList[epochs-1] #accuracy
print(acc)