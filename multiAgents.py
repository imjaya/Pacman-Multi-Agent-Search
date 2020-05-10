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
        remaining food (all_food_pos) and Pacman position after moving (cur_pos_pacman).
        scared_time holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        all_ghost_pos = successorGameState.getGhostStates()


        all_food_pos = successorGameState.getFood()
        cur_pos_pacman = successorGameState.getPacmanPosition()


        "*** YOUR CODE HERE ***"
        #Finding the Manhattan distance between ghosts and pacman and calculating the proximity to ghost at a distance of d=1 around the pacman

        proximity_score = 0
        ghost_distance = 1

        for item in successorGameState.getGhostPositions():

            distance = util.manhattanDistance(cur_pos_pacman, item)
            ghost_distance += distance

            if(distance <= 1):
                proximity_score = proximity_score +1



        Food_list = all_food_pos.asList()
        mininum_food_distance = -1

        for food_pos in Food_list:
            space = util.manhattanDistance(cur_pos_pacman, food_pos) #Calculating the distance between the pacman and the food pallete using the manhattan distance

            if mininum_food_distance >= space or mininum_food_distance == -1:
                mininum_food_distance = space


        p=(1 / float(mininum_food_distance))                     #Taking the inverse of minimum distance to the food pallete

        q=(1 / float(ghost_distance))                            # Taking the inverse of distance to the nearby ghost



        #return (successorGameState.getScore()  - ghost_distance + mininum_food_distanceproximity_score - proximity_score)
        return (successorGameState.getScore()  -3*q +2*p - (100*proximity_score))         #returning the weighted features sum

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
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth,gameState):
            p=gameState.isLose()
            q=gameState.isWin()
            d=self.depth
            if q or p  or depth == d:  # return the utility in case the defined depth is reached or the game is won/lost.
                gd=self.evaluationFunction(gameState)
                return(gd)
            if agent == 0:  #see if the agent is pacman (as given in the description that pacman is agent 0) hence maximize it
                score=max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                return(score)

            else:  #if the agent is not equal to zero this implies it is a ghost hence we need to minimize it.

                nextAgent = agent + 1 #go to the next agent and increase depth


                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0

                if nextAgent == 0: # increasing the depth
                   depth += 1
                score=min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                return(score)


        maximum = float("-inf")        #setting the maximum score to a very large number
        action = Directions.NORTH      #going to cordinate(0,1) w.r.t. to current
        for agentState in gameState.getLegalActions(0):   # checking if the agent state is in allowed legal actions
            score = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if score > maximum or maximum == float("-inf"):  # checking if the score calculated by minimax is graterbthan or equal to maximum value

                action = agentState
                maximum = score

        return(action)     #returning the  action to be performed

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):

    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(state):
            alpha, beta = None, None         #initializing the alpha beta value to none
            value, bestAction = None, None   #initializing the variables to none


            for action in state.getLegalActions(0):
                value = max(value,  minValue(state.generateSuccessor(0,  action),    1, 1,  alpha, beta))
                if alpha is None:   #checking the condition if alpha is none
                    alpha = value   #setting to alpha to max value
                    bestAction = action
                else:
                    bestAction =  action if value > alpha else bestAction  #determining the best action when action is not in legal actions
                    alpha=max(value, alpha)         #setting the alpha value to max value returned by max function


            return(bestAction)      # returning the best action

        def maxValue(state, agentIdx, depth, alpha, beta):   #defining the max function for alpha
            value=float("-inf")   #initializing the value to - infinity
            d=depth

            if d > self.depth:   #determining the depth condition
                eval=self.evaluationFunction(state)
                return(eval)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, d, alpha, beta)
                value = max(value, succ)  #setting the max valure
                if beta is not None and value > beta:
                    return(value)
                alpha = max(alpha, value)
            if value is not None:  # if value is less than beta then nothing is returned
                return(value)
            else:
                return(self.evaluationFunction(state))

        def minValue(state, agentIdx, depth, alpha, beta):  #defining the minimum value for the beta prunning
            value=float("inf")   #initializing the value to  infinity
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1, alpha, beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                mini = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta) #determining the minimum value returned

                if value is None:
                    value = mini   #setting the value to minimum
                else:
                    value = min(value, mini)


                if alpha is not None and value < alpha:  #checking the condition for prunning
                    return(value)

                if beta is None:



                    beta = value

                else:


                    beta = min(beta, value)

            if value is not None:


                return(value)

            else:
                sta=self.evaluationFunction(state)


                return(sta)




        action = alpha_beta(gameState)   # determing the action to be taken using the defined funtion

        return(action)   # returning the action to be taken
        util.raiseNotDefined()

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
        def expectimax_search(state, agentIndex, depth):

            if agentIndex == state.getNumAgents(): #checking if the agent is a last ghost



                if depth == self.depth: #if dept is equal to max
                    state=self.evaluationFunction(state)
                    return(state)    # returning the state

                else:      # if the max depth has not yet been reached
                    search=expectimax_search(state, 0, depth + 1)
                    return(search)

            else:  #if not min layer
                percept = state.getLegalActions(agentIndex)   #getting all possible moves
                length=len(percept)


                if(length == 0): #if a move is not possible then determine the state
                    state=self.evaluationFunction(state)
                    return(state)



                minimax = (expectimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in percept)  #finding all the mini max values


                if agentIndex == 0:  #if the agent is pacman
                    r=max(minimax)   #calcualting the max value of minimax
                    return(r)

                else:   # if not agent find the expectimax valu
                    set_minimax = set(minimax)  # storing all the possible moves in a set
                    summed=sum(set_minimax)     #taking the sum of alll elements
                    length=len(set_minimax)     # finding the number of possible moves
                    q=(summed/length)           # taking the ratio
                    return(q)

        action = max(gameState.getLegalActions(0), key=lambda x: expectimax_search(gameState.generateSuccessor(0, x), 1, 1))

        return(action)# returning the action with highest minimax value

def betterEvaluationFunction(currentGameState):



    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    cur_pos_pacman = currentGameState.getPacmanPosition()  #finding the current position of pacman
    all_food_pos = currentGameState.getFood()      # finding the positions of food pallete

    all_ghost_pos = currentGameState.getGhostStates()   #finding thte current positions of ghosts
    all_capsules_pos = currentGameState.getCapsules()     #finding position of capsule


    closestGhost = min([manhattanDistance(cur_pos_pacman, ghost.getPosition()) for ghost in all_ghost_pos]) #finding the closest ghost position

    if all_capsules_pos:


        closestCapsule = min([manhattanDistance(cur_pos_pacman, caps) for caps in all_capsules_pos]) # finding the nearest capsules
    else:

        closestCapsule = 0

    if closestCapsule:


        closest_capsule = (-3 / closestCapsule)  # taking the inverse of distance to the capsules for metrics
    else:


        closest_capsule = 400

    if closestGhost:


        ghost_distance = (-2 / closestGhost)   # taking the inverse of distance for the closest ghost metrics

    else:


        ghost_distance = -500

    foodList = all_food_pos.asList()

    if foodList:



        closestFood = min([manhattanDistance(cur_pos_pacman, food) for food in foodList]) #finding the closest food pallete

    else:

        closestFood = 0


    metrics=-0.4 * closestFood + 2*ghost_distance - 10 * len(foodList)  +closest_capsule #giving the weightage to the important parameters


    return(metrics)

# Abbreviation
better = betterEvaluationFunction
