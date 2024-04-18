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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"                               
        # Evaluate the proximity for food
        foodScore = 0
        if len(newFood.asList()) > 0:
            foodScore = 1 / min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])

        # Evaluate the proximity to ghosts 
        closestGhosttoPacman = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])
        if closestGhosttoPacman < 2:
            ghostReduceScore = 100
        else:
            ghostReduceScore = 0
        

        # Evaluate the impact of scared ghosts 
        # If ghosts are scared, then nearby ghosts aren't an issue (we shouldn't lose points for being close to them)
        if max(newScaredTimes) != 0: 
           closestScaredGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])
        else:
            closestScaredGhost = 0
        ghostScaredScore = sum(newScaredTimes) + closestScaredGhost

        # Evaluate total score
        newScore = successorGameState.getScore() - currentGameState.getScore()
        return newScore + foodScore - ghostReduceScore + ghostScaredScore
    

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def max_value(state, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0): # Get legal actions for Pacman
                successor = state.generateSuccessor(0, action)
                v = max(v, min_value(successor, 1, depth))
            return v

        def min_value(state, ghost_index, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(ghost_index): # Get legal actions for Ghost
                successor = state.generateSuccessor(ghost_index, action)
                if ghost_index == state.getNumAgents() - 1:
                    v = min(v, max_value(successor, depth - 1))
                else:
                    v = min(v, min_value(successor, ghost_index + 1, depth))
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = max(legalActions, key=lambda x: min_value(gameState.generateSuccessor(0, x), 1, self.depth))
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, alpha, beta, depth, agentIndex):
          depth -= 1 
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          
          v = float("-inf")
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            # Get the value of the leftmost child
            score = min_value(successor, alpha, beta, depth,agentIndex + 1)[0] 
            if score > v: # Update score if it's better than what we have. It should be since v is initialized at -inf
              v = score
              bestAction = action # After analyzing the leftmost child, the bestAction should be to take that action
            if v > beta:
              return (v, bestAction)
            alpha = max(v, alpha)
          return (v, bestAction)

        def min_value(state, alpha, beta, depth, agentIndex):
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          
          v = float("inf")
          if state.getNumAgents() - 1 > agentIndex:
             value, nextAgent = (min_value, agentIndex + 1)
          else:
             value, nextAgent = (max_value, 0)

          # Follow the skeleton code implementation 
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            score = value(successor, alpha, beta, depth,nextAgent)[0]
            if score < v: # Update score (boundary) if it's better than what we have. It should be since v is initialized at inf
              v = score
              bestAction = action
            if v < alpha:
              return (v, bestAction)
            beta = min(v, beta)          
          return (v, bestAction)
        
        return max_value(gameState, float("-inf"), float("inf"), self.depth, 0)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def evaluateAction(state, agent_num, depth, agent):
            legal_moves = state.getLegalActions(agent_num) # Collect legal moves for Pacman (0) and the ghosts (1, 2, 3, 4,...)
            # Follow given implementation for Exectimax in the CS 188 Notes 
            if agent is max:
                # Select the action that gets you the highest score 
                return max(getExpectiMaxValue(state.generateSuccessor(agent_num, action), agent_num + 1, depth - 1) for action in legal_moves) 
            elif agent is min:
                # Add the probability to implemnt ghosts of possibly making "dumb" moves
                return sum(1.0 / len(legal_moves) * getExpectiMaxValue(state.generateSuccessor(agent_num, action), agent_num + 1, depth - 1) for action in legal_moves)

        def getExpectiMaxValue(state, agent_num, depth):
            agent_num = agent_num % state.getNumAgents() 
            # Base Case
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            elif agent_num == 0: # Pacman (max agent)
                return evaluateAction(state, agent_num, depth, max)
            else: # Ghosts (min agent)
                return evaluateAction(state, agent_num, depth, min)

        legal_moves = gameState.getLegalActions(0)
        # Get a list of the scores you can get from each possible action
        scores = [getExpectiMaxValue(gameState.generateSuccessor(0, action), 1, self.depth * gameState.getNumAgents() - 1) for action in legal_moves] # Build the Expectimax tree
        best_score = max(scores)
        best_choices = [action for action, score in enumerate(scores) if score == best_score] # Create a list of most optimal actions since we're dealing with randomness
        # Randomly select one of the best choices Pacman can make sincewe don't know what action the ghosts will take (whether it's optimal or not)
        return random.choice([legal_moves[action] for action in best_choices])

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Write a better evaluation function for Pacman in the provided function betterEvaluationFunction. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did. With depth 2 search, your evaluation function should clear the smallClassic layout with one random ghost more than half the time and still run at a reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he’s winning).

    DESCRIPTION: < Features:
    - Food Score: Inverse of the distance to the closest food pellet. Higher score if less food is remaining.
    - Power Pellet Score: Inverse of the distance to the closest power pellet. Higher score if power pellets are available.
    - Ghost Score: Inverse of the distance to the closest ghost. A penalty is applied if a ghost is too close. If the ghost is scared,
      the penalty is reversed, providing a reward.

    Coefficients:
    - Adjusted weights (foodWeight, powerWeight, ghostWeight) control the importance of each feature in the final evaluation.
    - Scores for different features are combined with weights and the current game score to compute the final evaluation.
>
    """
    "*** YOUR CODE HERE ***"
    foodWeight, powerWeight, ghostWeight = 2, 5, 2

    pacmanPos, foodGrid = currentGameState.getPacmanPosition(), currentGameState.getFood()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    isScared = max(scaredTimes) != 0
    ghostPositions, powerPellets = currentGameState.getGhostPositions(), currentGameState.getCapsules()

    
    closestGhostDist = float(min([util.manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPositions]))
    closestPowerPelletDist = float(min([len(powerPellets)] + [util.manhattanDistance(powerPos, pacmanPos) for powerPos in powerPellets]))
    closestFoodDist = float(min([foodGrid.width + foodGrid.height] + [util.manhattanDistance(pacmanPos, foodPos) for foodPos in foodGrid.asList()]))

    
    powerScore = 1 if len(powerPellets) == 0 else 1 / closestPowerPelletDist
    foodScore = 1 if len(foodGrid.asList()) == 0 else 1 / closestFoodDist
    ghostScore = -999 if closestGhostDist < 1 else 1 / closestGhostDist

    if isScared and closestGhostDist < max(scaredTimes):
        ghostWeight, ghostScore = 100, abs(ghostScore)

    finalScore = foodWeight * foodScore + ghostWeight * ghostScore + powerWeight * powerScore + currentGameState.getScore()

    return finalScore

  

# Abbreviation
better = betterEvaluationFunction
