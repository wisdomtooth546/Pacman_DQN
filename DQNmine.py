#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:48:19 2020

@author: prem
"""
import numpy as np
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


import tensorflow as tf
from tensorflow import keras
from capture import *
from collections import deque
import random
from tensorflow.keras import losses

import game 

DISCOUNT = 0.80
REPLAY_MEM_SIZE = 100
UPDATE_EVERY = 5
MIN_REPLAY_MEM = 30
Mini_batch_size = 20
EPISODES = 50

class DQN():
    
   def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1
    
   def create_model(self) :
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(256,(3,3), input_shape = (10,10,10)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(256, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(keras.layers.Dense(64))

    model.add(keras.layers.Dense(4, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss=losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    
    return model

   def __init__(self):
       
    self.Q_global = []
    self.model = self.create_model()
    
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())
    
    self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)
    self.update_counter = 0
    
   def update_replay_memory(self,transition):
    self.replay_memory.append(transition)
    
   def get_qs(self,state):
    return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

   def train(self,terminal_state,step):
    if len(self.replay_memory)< MIN_REPLAY_MEM:
        return
    
    minibatch = random.sample(self.replay_memory, Mini_batch_size)
    current_states = np.array([transition[0] for transition in minibatch])/255
    current_qs_list = self.model.predict(current_states)
    
    new_current_states = np.array([transition[3] for transition in minibatch])/255
    future_qs_list = self.target_model.predict(new_current_states)
    
    X = []
    y = []

    for index,(current_state, action, reward, new_current_state, done) in enumerate(minibatch):
     if not done:
         max_future_q = np.max(future_qs_list[index])
         new_q = reward + DISCOUNT * max_future_q
     else:
         new_q = reward
    #update q
    
     current_qs = current_qs_list[index]
     current_qs[action] = new_q
     
     X.append(current_state)
     y.append(current_qs)

    self.model.fit(np.array(X)/255, np.array(y), batch_size = Mini_batch_size, verbose =0, shuffle=False)

    if terminal_state:
  
     self.update_counter += 1
   
    if self.update_counter > UPDATE_EVERY:
     self.target_model.set_weights(self.model.get_weights())
     self.update_counter = 0