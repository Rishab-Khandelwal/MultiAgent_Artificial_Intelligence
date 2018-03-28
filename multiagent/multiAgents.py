# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        #Getting the successor Game State
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #Getting the new ghost states for the successor states 
        newGhostStates = successorGameState.getGhostStates()

        #getting all the power capsules in the successor game state
        powerPellet = successorGameState.getCapsules()

        #getting the new Pacman position in the successor game state
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        #print "successorGameState", successorGameState
        
        #print "new position is ", successorGameState.getPacmanPosition()
        
        #print "new Food is ", successorGameState.getFood().asList()
        
        #print "new GhostState is ", newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print "new newScaredTimes is ", newScaredTimes

        #Total scare time of the ghost in the successor ghost states
        TotalScaredTimes = 0

        for ghostState in newGhostStates:
          TotalScaredTimes += ghostState.scaredTimer

        #nearest Food distance in the successor state
        nearestFoodDistance = nearestDistance(newFood.asList(),newPos)
        #print "Hello Reached here"
        #nearest Pallet Distance in the successor state
        nearestPalletDistance = nearestDistance(powerPellet,newPos)
        #print "Reached here also yay"
        ghostPostion = successorGameState.getGhostPositions()
        nearestGhostDistance = nearestDistance(ghostPostion,newPos)
        #print "almost completed"
        numberOfPellets = len(powerPellet)

        if nearestFoodDistance > 0 :

          evalFunction = 1 * nearestGhostDistance + 200.0/nearestFoodDistance + 100* successorGameState.getScore() + 690 * TotalScaredTimes \
                         + 450 * successorGameState.getNumFood()  + 450 * numberOfPellets - 300 * nearestPalletDistance
        else : 
          evalFunction = 1 * nearestGhostDistance + 100 * successorGameState.getScore() + 690 * TotalScaredTimes \
                         + 450 * successorGameState.getNumFood()  + 450 *numberOfPellets - 300 * nearestPalletDistance
          "*** YOUR CODE HERE ***"
        #print "successorGameState score is ", successorGameState.getScore()
        return evalFunction


def nearestDistance(possibleStates, currentPos):
      #calculating the nearest distance of an entity from the current position
      nearestDistance = 0
      dist = {}
      curr = currentPos
      if len(possibleStates) > 0:
        for state in possibleStates:
          end = state
          if curr == end:
            break

          manhattanDistance = abs(curr[0] - end[0]) + abs(curr[1] -end[1])
          dist[end] = manhattanDistance
      else:
        return nearestDistance


      getValues = dist.values()

      if len(getValues) > 0:

        min = getValues[0]
        for value in getValues:
          if min > value:
            min = value

        return min
      else:
        return nearestDistance
    




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ActionScore = {}

        nodeIndex = self.index
        
        flag = False
        numberOfAgents = gameState.getNumAgents()
        totalDepth = self.depth * numberOfAgents

        #successors = self.getSuccessors(gameState,nodeIndex)

        legalMoves = gameState.getLegalActions(nodeIndex)

        successors = []

        for action in legalMoves:
          successors.append((action, gameState.generateSuccessor(nodeIndex,action)))
        # processing minimax for each of the successors
        for successor in successors:
          score = self.processMinimax(successor,(nodeIndex+1)% numberOfAgents,totalDepth -1)

          ActionScore[score] = successor[0]

        if nodeIndex == 0:
          keyIndex = max(ActionScore.keys())
        else:
          keyIndex = min(ActionScore.keys())


        finalAction = ActionScore[keyIndex]

        return finalAction


    def processMinimax(self, successor,nodeIndex,currentDepth):

      if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
        return self.evaluationFunction(successor[1])


      if nodeIndex == 0 :
        #maximize
        return self.minimax(currentDepth, nodeIndex, successor,flag = False)
      else:
        return self.minimax(currentDepth,nodeIndex,successor,flag = True)


    def minimax(self,currentDepth,nodeIndex,successor,flag):
      #returns the min if the flag is set to true else return the max of the node score
      if flag == True:

        nodeScore = [float("inf")]
        #childNodes = self.getSuccessors(successor[1],nodeIndex)
        legalMoves = successor[1].getLegalActions(nodeIndex)

        successors = []

        for action in legalMoves:
          successors.append((action, successor[1].generateSuccessor(nodeIndex,action)))
        childNodes = successors
        
        for childNode in childNodes:
          nodeScore.append(self.processMinimax(childNode,(nodeIndex+1)%successor[1].getNumAgents(),currentDepth - 1 ))

        bestScore = min(nodeScore)
        return bestScore
      else:

        nodeScore = [float("-inf")]

        #childNodes = self.getSuccessors(successor[1],nodeIndex)
        legalMoves = successor[1].getLegalActions(nodeIndex)

        successors = []

        for action in legalMoves:
          successors.append((action, successor[1].generateSuccessor(nodeIndex,action)))
        childNodes = successors

        for childNode in childNodes:
          nodeScore.append(self.processMinimax(childNode,(nodeIndex+1)%successor[1].getNumAgents(),currentDepth - 1 ))

        bestScore = max(nodeScore)
        return bestScore



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        ActionScore = {}

        nodeIndex = self.index
        numberOfAgents = gameState.getNumAgents()

        totalDepth = self.depth * numberOfAgents

        
        beta = float("inf")

        alpha = float("-inf")

        flag = False
        if nodeIndex == 0:
          score = float("-inf")
        else:
          score = float("+inf")

        for legalMove in gameState.getLegalActions(nodeIndex):
          successor = (legalMove,gameState.generateSuccessor(nodeIndex,legalMove))
          newScore = self.processAlphaBeta(successor,(nodeIndex+1)%numberOfAgents,totalDepth - 1 ,alpha,beta)
          ActionScore[newScore] = successor[0]

          if nodeIndex == 0:
            score = max(newScore,score)
            if score < alpha :
              break
            alpha = max(alpha,score)
          else:
            score = min(newScore,score)
            if score > beta :
              break
            beta = min(beta , score)


        if nodeIndex == 0 : 
          keyIndex = max(ActionScore.keys())
        else:
          keyIndex = min(ActionScore.keys())

        finalAction = ActionScore[keyIndex]

        return finalAction



        #util.raiseNotDefined()

    def processAlphaBeta(self,successor,nodeIndex,currentDepth,alpha,beta) :
      if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
        return self.evaluationFunction(successor[1])


      if nodeIndex == 0 :
        return self.minimax(alpha, beta , currentDepth, nodeIndex, successor,flag = False)
      else:
        return self.minimax(alpha,beta,currentDepth,nodeIndex,successor,flag = True)


    def minimax(self,alpha,beta,currentDepth,nodeIndex,successor,flag) :
      #returns the max if the flag is set to false else return the min of the node score
      if flag == False:
        currNodeScore = float("-inf")

        for childMove in successor[1].getLegalActions(nodeIndex):
          childNode = (childMove,successor[1].generateSuccessor(nodeIndex,childMove))
          childNodeScore = self.processAlphaBeta(childNode,(nodeIndex+1)% successor[1].getNumAgents(),currentDepth-1,alpha,beta)

          currNodeScore = max(currNodeScore,childNodeScore)

          if currNodeScore > beta:
            return currNodeScore

          alpha = max(alpha,currNodeScore)

        return currNodeScore

      else:
        currNodeScore = float("inf")

        for childMove in successor[1].getLegalActions(nodeIndex):
          childNode = (childMove,successor[1].generateSuccessor(nodeIndex,childMove))
          childNodeScore = self.processAlphaBeta(childNode,(nodeIndex+1)% successor[1].getNumAgents(),currentDepth-1,alpha,beta)

          currNodeScore = min(currNodeScore,childNodeScore)

          if currNodeScore < alpha:
            return currNodeScore

          beta = min(beta,currNodeScore)

        return currNodeScore




        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"


        ActionScore = {}

        nodeIndex = self.index
        numberOfAgents = gameState.getNumAgents()

        totalDepth = self.depth * numberOfAgents

        alpha = float("-inf")
        beta = float("inf")
        flag = False

        if nodeIndex == 0:
          score = float("-inf")
        else:
          score = float("-inf")

        for legalMove in gameState.getLegalActions(nodeIndex):
          successor = (legalMove,gameState.generateSuccessor(nodeIndex,legalMove))
          newScore = self.processAlphaBeta(successor,(nodeIndex+1)%numberOfAgents,totalDepth - 1 ,alpha,beta)
          ActionScore[newScore] = successor[0]

          if nodeIndex == 0:
            score = max(newScore,score)
            if score < alpha :
              break
            alpha = max(alpha,score)
          else:
            score = min(newScore,score)
            if score > beta :
              break
            beta = min(beta , score)


        if nodeIndex == 0 : 
          keyIndex = max(ActionScore.keys())
        else:
          keyIndex = min(ActionScore.keys())

        finalAction = ActionScore[keyIndex]

        return finalAction

    def processAlphaBeta(self,successor,nodeIndex,currentDepth,alpha,beta) :
      if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
        return self.evaluationFunction(successor[1])


      if nodeIndex == 0 :
        return self.minimax(alpha, beta , currentDepth, nodeIndex, successor, flag = True)
      else:
        return self.minimax(alpha,beta,currentDepth,nodeIndex,successor, flag = False)


    def minimax(self,alpha,beta,currentDepth,nodeIndex,successor,flag) :
      #returns the max if the flag is set to true else return the expected value of the node score
      if flag == True:

        currNodeScore = float("-inf")

        for childMove in successor[1].getLegalActions(nodeIndex):
          childNode = (childMove,successor[1].generateSuccessor(nodeIndex,childMove))
          childNodeScore = self.processAlphaBeta(childNode,(nodeIndex+1)% successor[1].getNumAgents(),currentDepth-1,alpha,beta)

          currNodeScore = max(currNodeScore,childNodeScore)

          if currNodeScore > beta:
            return currNodeScore

          alpha = max(alpha,currNodeScore)

        return currNodeScore
      else:


        currNodeScore = float("inf")

        scoreList = []

        for childMove in successor[1].getLegalActions(nodeIndex):
          childNode = (childMove,successor[1].generateSuccessor(nodeIndex,childMove))
          childNodeScore = self.processAlphaBeta(childNode,(nodeIndex+1)% successor[1].getNumAgents(),currentDepth-1,alpha,beta)
          scoreList.append(childNodeScore)
          #currNodeScore = min(currNodeScore,childNodeScore)

          #if currNodeScore < alpha:
          #  return currNodeScore

          #beta = min(beta,currNodeScore)
          cumScore = 0
          for score in scoreList:
            cumScore += score
          ansNodeScore = cumScore/len(scoreList)
        return ansNodeScore

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState) :


  successorGameState = currentGameState
  #print "successorGameState", successorGameState
  newPos = successorGameState.getPacmanPosition()
  #print "new position is ", successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  #print "new Food is ", successorGameState.getFood().asList()
  newGhostStates = successorGameState.getGhostStates()
  #print "new GhostState is ", newGhostStates
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  #print "new newScaredTimes is ", newScaredTimes

  powerPellet = successorGameState.getCapsules()
  TotalScaredTimes = 0

  for ghostState in newGhostStates:
    TotalScaredTimes += ghostState.scaredTimer

  nearestFoodDistance = nearestDistance(newFood.asList(),newPos)
  #print "Hello Reached here"
  nearestPalletDistance = nearestDistance(powerPellet,newPos)
  #print "Reached here also yay"
  ghostPostion = successorGameState.getGhostPositions()
  nearestGhostDistance = nearestDistance(ghostPostion,newPos)
  #print "almost completed"
  numberOfPellets = len(powerPellet)

  if nearestFoodDistance > 0 :

    evalFunction = 1 * nearestGhostDistance + 200.0/nearestFoodDistance + 100* successorGameState.getScore() + 690 * TotalScaredTimes \
                         + 450 * successorGameState.getNumFood()  + 450 * numberOfPellets - 300 * nearestPalletDistance
  else : 
    evalFunction = 1 * nearestGhostDistance + 100 * successorGameState.getScore() + 690 * TotalScaredTimes \
                         + 450 * successorGameState.getNumFood()  + 450 *numberOfPellets - 300 * nearestPalletDistance
        
  #print "successorGameState score is ", successorGameState.getScore()
  return evalFunction  

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

