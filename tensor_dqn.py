#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:45:49 2020

@author: prem
"""


from captureAgents import CaptureAgent
import random, util
from game import Directions
from util import nearestPoint
import math

import numpy as np

from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam

import game 
from itertools import count

DISCOUNT = 0.80
REPLAY_MEM_SIZE = 100
UPDATE_EVERY = 5
MIN_REPLAY_MEMORY_SIZE = 30

Mini_batch_size = 20
EPISODES = 50
REPLAY_MEMORY_SIZE = 100
epsilon = 0.9
EPS_END = 0.1
DECAY = 100
AGGREGATE_EVERY = 10
n_actions = 4

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DQNAgent:
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(32,32,3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(n_actions, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

       

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, Mini_batch_size)
    
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
    
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
    
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

        # If not a terminal state, get new q from future states, otherwise set it to 0
        # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

        # Update Q value for given state
        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        # And append to our training data
        X.append(current_state)
        y.append(current_qs)

    # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=Mini_batch_size, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
    
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
      return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agent = DQNAgent()
epsilon = 0.9

class ReflexCaptureAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 10.0}
  
             
class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
 
    
   
class DefensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.index = index
        self.target = None
        self.current_score = 0
        self.lastObservedFood = None
        self.epsilon = epsilon
        # This variable will store our patrol points and
        # the agent probability to select a point as target.
        self.patrolDict = {}
    
    
    def step(self,action):
        
        mypos = self.state.getAgentPosition(self.index)
        
        score = self.getScore(self.state)
        if self.target == mypos:
            self.last_reward = 50
        else: self.last_reward = 10
        #actions = self.state.getLegalActions(self.index)    
        
        new_state = self.state.generateSuccessor(self.index,action)
       
        reward = self.last_reward
        done = True if score == 200 else False
        
        return new_state, reward, done
        
      
    def select_action(self,gameState):
      #Returns an action based on the state from the policy net
        
      #Exploitatio v Exploration  
        myPos = self.state.getAgentPosition(self.index)
        enemies=[gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        inRange = filter(lambda x: x.isPacman and x.getPosition() != None,enemies)
        if len(inRange)>0:
             eneDis,enemyPac = min([(self.agent.getMazeDistance(myPos,x.getPosition()), x) for x in inRange])
             self.target=enemyPac.getPosition()
      
         
    
        
    
    
    
    
       
    def distFoodToPatrol(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        food = self.getFoodYouAreDefending(gameState).asList()
        total = 0

        # Get the minimum distance from the food to our
        # patrol points.
        for position in self.noWallSpots:
            closestFoodDist = "+inf"
            for foodPos in food:
                dist = self.getMazeDistance(position, foodPos)
                if dist < closestFoodDist:
                    closestFoodDist = dist
            # We can't divide by 0!
            if closestFoodDist == 0:
                closestFoodDist = 1
            self.patrolDict[position] = 1.0 / float(closestFoodDist)
            total += self.patrolDict[position]
        # Normalize the value used as probability.
        if total == 0:
            total = 1
        for x in self.patrolDict.keys():
            self.patrolDict[x] = float(self.patrolDict[x]) / float(total)

    def selectPatrolTarget(self):
        """
        Select some patrol point to use as target.
        """
        rand = random.random()
        sum = 0.0
        for x in self.patrolDict.keys():
            sum += self.patrolDict[x]
            if rand < sum:
                return x
            
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        # Compute central positions without walls from map layout.
        # The defender will walk among these positions to defend
        # its territory.
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWallSpots = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.noWallSpots.append((centralX, i))
        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        while len(self.noWallSpots) > (gameState.data.layout.height - 2) / 2:
            self.noWallSpots.pop(0)
            self.noWallSpots.pop(len(self.noWallSpots) - 1)
        # Update probabilities to each patrol point.
        self.distFoodToPatrol(gameState)
        
        
        
       

    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # If some of our food was eaten, we need to update
        # our patrol points probabilities.
        if self.lastObservedFood and len(self.lastObservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
            self.distFoodToPatrol(gameState)

        mypos = gameState.getAgentPosition(self.index)
        if mypos == self.target:
            self.target = None

        # If we can see an invader, we go after him.
        x = self.getOpponents(gameState)
        enemies = [gameState.getAgentState(i) for i in x]
        invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
        if len(invaders) > 0:
            positions = [agent.getPosition() for agent in invaders]
            self.target = min(positions, key=lambda x: self.getMazeDistance(mypos, x))
            
    def getAction(self,state):
        self.state = state
        for episode in range(1, EPISODES + 1):
          #self.registerInitialState(self)
          episode_reward = 0
          step = 1
          
          done = False
          while not done:

        # This part stays mostly the same, the change is to query a model for Q values
           if np.random.random() > self.epsilon:
            # Get action from Q table
             action = np.argmax(agent.get_qs(state))
        else:
            # Get random action
            action = np.random.randint(0, 4)

        new_state, reward, done = self.step(action)
          
        episode_reward += reward

        agent.update_replay_memory((state, action, reward, new_state, done))
        agent.train(done, step)
        state = new_state
        step += 1  
          

        
        if self.epsilon > EPS_END:
          self.epsilon *= DECAY
          self.epsilon = max(EPS_END, self.epsilon)
    
