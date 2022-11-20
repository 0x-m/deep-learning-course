
#miniproject PART1 - a neuron should learn logical Xor operation 
#--------------------IMPORTS----------------
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

xor_training_set ={(0,0):0,(0,1):1,(1,0):1,(1,1):0}
xor_neuron = Neuron.LogicGateNeuron(w,b) #create an instance of LogicGateNeuron

#using a trainer to train the or_neuron !
epochs = 100
xor_trainer = Trainer.Trainer(xor_neuron,xor_training_set)
xor_trainer.train(epochs,lr = 2)

#plotting training erro vs epochs:---------------------------
plt.plot(range(0,epochs),xor_trainer.trainerrList)
plt.axis([0 , epochs , 0 , 0.5])
plt.title("Traning ERROR for xor_neuron")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.show()
#-------------------------------------------------------------
acc = xor_trainer.trainerrList[epochs-1]
print(acc)
#----------------------------------------
plt.figure()
x1 = [0,1]
y1 = [0,1]
x2 = [0,1]
y2 = [1,0]

plt.scatter(x1,y1,color='red')
plt.scatter(x2,y2,color='blue')
plt.show()
 