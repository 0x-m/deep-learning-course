
#--------------------------Imports-------------------------------------
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------

#----------------Define model and load the weights---------------------
model = Sequential()
model.add(Dense(32,input_dim=2,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3))
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.load_weights('train_weights12.h5') 
#-----------------------------------------------------------------------

#--------------------------Testing--------------------------------------
env = gym.make('MountainCar-v0')
c_state = env.reset().reshape(1,2)
rewards = list()
reward = 0
n_episodes = 50

for episode in range(n_episodes):
  reward = 0
  env.reset()
  for _ in range(200):
    a = np.argmax(model.predict(c_state)[0])
    s,r,done,_ = env.step(a)
    env.render()
    reward +=r
    if s[0] > 0.5:
        break
    c_state = s.reshape(1,2)
  rewards.append(reward)

#-----------Visulazation-----------------------------------------------
plt.plot(range(n_episodes),rewards)
plt.show()
#----------------------------------------------------------------------
#best reward: near -120
#----------------------------------------------------------------------
 
