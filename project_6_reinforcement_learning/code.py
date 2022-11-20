
#--------------------------------------------------------
#MINIPROJECT 5
#Mountain Car Problem
#Author:Hamze Ghaedi (9831419)
#---------------------------------------------------------

#-------------------------------------------------------
#------------------------IMPORTS------------------------
#-------------------------------------------------------
import numpy as np
import random
from collections import deque
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
#--------------------------------------------------------


#-------------------------------------------------------
#-------------------HYPERPARAMETERS---------------------
#-------------------------------------------------------

eps = 0.99 #epsilon for greedy action 
eps_decay = 0.005 #decay rate for epsilon per episode
gamma = 0.9 #discount factor
alpha = 0.01 #learning rate
n_episodes = 1500 #number of episodes
n_steps = 201 #number of steps per episode
in_dim = 2 #input dim of the networks (position,velocity)
out_dim = 3 #output dim of the networks Q-values for (0,1,2) actions
batch_size = 32 
#-----------------------------------------------------------------
#the agent saves the  gathered informations about the environment in this variable
memory = deque()

#a helper function for inserting information into agent's memory
def save_to_memory(current_state,action,reward,new_state,terminal_flag):
  memory.append((current_state,action,reward,new_state,terminal_flag))
#----------------------------------------------------------------

#-------------------------------------------------------
#-------------------DEFINE MODELS-----------------------
#-------------------------------------------------------


#loss-function: MSE
#optimizer : Adam

def generate_model(name,lr):
  model = Sequential()
  model.add(Dense(32,input_dim=2,activation='relu'))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(3))

  model.compile(loss='mse',optimizer=Adam(lr),metrics=['mse'])
  return model


model_for_train = generate_model('model_for_train',alpha)
model_for_predict = generate_model('model_for_predict',alpha)  
#-------------------------------------------------------------


#-------------------------------------------------------
#------------------------GREEDY ACTION------------------
#-------------------------------------------------------

def greedy_action(eps,state):
  q_vals = model_for_train.predict(state)
  eeps = max(eps,0.01)
  if(np.random.uniform(0,1,1)[0] > eeps):
    return np.argmax(q_vals[0])
  return env.action_space.sample()

#--------------------------------------------------------
  
#-------------------------------------------------------
#------------------------IMPORTS------------------------
#-------------------------------------------------------

def train_on_batch(batch_size):

  if len(memory)  < batch_size:
    return


  #---------training dataset--------------------
  train_batch = list() #input datapoints
  targets = list() #corresponding targets of input the datapoints
  #---------------------------------------------

  batch = random.sample(memory,batch_size) #choose a number of random samples from memory
 

  #populating training dataset---------------------
  for sample in batch:

    current_state,action,reward,new_state,terminal_flag = sample

    q_vals = model_for_train.predict(new_state)

    if terminal_flag: #in the end of the episode we have no action so no reward
      q_vals[0][action] = reward
    else:
      act = np.argmax(q_vals[0])
      y_ddqn = reward + gamma * model_for_predict.predict(new_state)[0][act]
      q_vals[0][action] = y_ddqn
    train_batch.append(current_state[0])
    targets.append(q_vals[0])

  #----------------train the model------------------
  model_for_train.train_on_batch(np.array(train_batch),np.array(targets))
  #-------------------------------------------------


#-------------------------------------------------------
#------------------------MAIN---------------------------
#-------------------------------------------------------

env = gym.make('MountainCar-v0')

#uncomment if needed----------------------------------

#model_for_train.load_weights('train_weights.h5')

#-----------------------------------------------------

for episode in range(n_episodes):

  reward_per_episode = 0
  itr =0
  current_state = env.reset().reshape(1,2)
   
  #epsilon decay---------
  eps -=eps_decay
  #----------------------

  for step in range(n_steps):

    action = greedy_action(eps,current_state)

    #below line should be commented for boosting the taining process------------
    #it does'nt work on colab!
    #env.render()
    #---------------------------------------------------------------------------

    new_state,reward,done,_ = env.step(action)
    
    save_to_memory(current_state,action,reward,new_state.reshape(1,2),done)
    reward_per_episode +=reward

    if(new_state[0] >=0.5):
      print("-*-*-*-*-*-*-*-*-*-*-*-*( TARGET HIT )-*-*-*-*-*-*-*-*-*-*-*-*-*-")
      model_for_train.save_weights("train_weights{}.h5".format(episode))

    if done:
      break

    current_state = new_state.reshape(1,2)

  print("\n episode:{}".format(episode)+"  reward_per_episode:{}".format(reward_per_episode))
  train_on_batch(32)

  #model_for_predicted is updated every 2 episodes-----------------
  if(episode %2 ==0):
    model_for_predict.set_weights(model_for_train.get_weights())
  #----------------------------------------------------------------

#results saved every 50 episode------------------------------------
  if (episode % 50 == 0):
     model_for_train.save_weights('train_weights.h5')
#------------------------------------------------------------------
  
