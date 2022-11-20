#miniproject PART2 - train a LogicGateNeuron using data set provided by make_blob

#-----------------IMPORTS----------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import Neuron
import Trainer
#---------------------------------------------
#random initialization
w1 = np.random.randn()
w2 = np.random.randn()
w = [w1,w2]
b = 1

#generate a data set consisting of points that are extracted from mixture of gaussian ditro with two centers 
no_of_samples = 100
x,y = make_blobs(no_of_samples,2,2)

#convert data set to a dictionary form so a trainer class can take it---------------
training_set = dict()
for i in range(0,no_of_samples):
    training_set[(x[i][0],x[i][1])] = y[i]
#-----------------------------------------------------------------------------------

neu = Neuron.LogicGateNeuron(w,b)

my_trainer = Trainer.Trainer(neu,training_set)
epochs = 100
my_trainer.train(epochs,2)
acc = my_trainer.trainerrList[epochs-1] #accuracy of training process 

#plotting mse vs number of epochs
plt.plot(range(0,epochs),my_trainer.trainerrList)
plt.show()
print(acc)
#-----------------------Testing process-----------------------
#in order to test the trained  neu neuron, I feed a new set of 50 points extracted from mix-gas distro 
#with 2 centers into the neu neuron then I compute the mean square of test error 

x_t,y_t = make_blobs(50,2,2)
#convert data set to a dictionary form so a trainer class can take it---------------
test_set = dict()


for i in range(0,50):
    test_set[(x[i][0],x[i][1])] = y[i]

my_trainer.test(test_set)
print(my_trainer.testerr)












