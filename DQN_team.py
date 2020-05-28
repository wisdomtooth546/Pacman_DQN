#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:23:12 2020

@author: prem
"""

from captureAgents import CaptureAgent
from capture import GameState
import random, util
from game import Directions
from util import nearestPoint
import math

import torch
import torch.nn as nn

from collections import namedtuple
import torch.nn.functional as F
import torch.optim as optim

import game 
from itertools import count

DISCOUNT = 0.80
REPLAY_MEM_SIZE = 100
UPDATE_EVERY = 5
MIN_REPLAY_MEM = 30
Mini_batch_size = 20
EPISODES = 50

EPS_START = 0.9
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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  
    
memory = ReplayMemory(REPLAY_MEM_SIZE)

class DQN(nn.Module):
    
   def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Linear(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Linear(16, 32)
       # self.bn2 = nn.BatchNorm2d(32)
       # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
       # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(32, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
   def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

   
             
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
      
        game_state = self.state.getAgentState(self.index) 
        sample = random.random()
        sample = 10
        eps_threshold = EPS_END+(EPS_START-EPS_END) * \
         math.exp(-1. * self.steps_done / DECAY)
        self.steps_done += 1
        actions = gameState.getLegalActions(self.index)
        
        if sample > eps_threshold:
         with torch.no_grad():
             # t.max(1) will return largest column value of each row.
             # second column on max result is index of where max element was
             # found, so we pick action with the larger expected reward.
             a = self.policy_net(torch.tensor(game_state)).max(1)[1].view(1, 1)
             
             if a in actions:
                 return a
             else:
              return torch.tensor(actions[0], device=device, dtype=torch.long)
        else:
         return torch.tensor(actions[0], device=device, dtype=torch.long)
      
    
    
    
    def optimize_model(self):
       if len(memory) < Mini_batch_size:
              return
       transitions = memory.sample(Mini_batch_size)
       
       #converting the array of transitions into transition of arrays 
       batch = Transition(*zip(*transitions))
       
       #creating a mask of non-final states to filter out the final states values while running the simulation
       non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                           batch.next_state)), device=device, dtype=torch.bool)
       non_final_next_states = torch.cat([s for s in batch.next_state
                                                 if s is not None])
       state_batch = torch.cat(batch.state)
       action_batch = torch.cat(batch.action)
       reward_batch = torch.cat(batch.reward)
       
       #creating Q(s_t,a) from Q(s_t)
       Q_state_action_values = self.policy_net(state_batch).gather(1, action_batch)
       
       next_state_values = torch.zeros(Mini_batch_size, device=device)
       next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
       
       #computing the expected Q(s_t,a) using Bellman's equation
       expected_state_action_values = (next_state_values*DISCOUNT) + reward_batch
      
       #huber loss is used here
       loss = F.smooth_l1_loss(Q_state_action_values, expected_state_action_values.unsqueeze(1))
       #optimizing the model
       self.optimizer.zero_grad()
       loss.backward()
       for param in self.policy_net.parameters():
         param.grad.data.clamp_(-1, 1)
       self.optimizer.step() 
       
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
        
        
        self.policy_net = DQN(gameState.data.layout.height,gameState.data.layout.width,n_actions).to(device)
        self.target_net = DQN(32,32,n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
       
    
        self.steps_done = 0
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
        for episode in range(EPISODES):
          #self.registerInitialState(self)
          episode_durations = []  
          self.steps_done = 0  
          self.state = state
          for t in count():
              #selecting the best action as determined by the policy net
              action = self.select_action(self.state)
              #Deciding what happens when the agent in state, s takes this action
              if action is not None:
                  state,reward, done = self.step(action)
                  print('Steps done: %d' %self.steps_done)
                  reward = torch.tensor([reward],device =device)
              if not done:
                  next_state = self.getSuccessor(state,action)
              else: next_state = None
              
              
              
              memory.push(state,action,next_state,reward)
              
              state = next_state
              self.optimize_model()
              if done:
                  episode_durations.append(t+1)
                  break
          
          if episode % AGGREGATE_EVERY == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
