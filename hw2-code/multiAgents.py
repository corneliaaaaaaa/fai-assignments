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

import sys


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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # I referenced this github repo for this function, but did minor modifications:
        # https://github.com/karlapalem/UC-Berkeley-AI-Pacman-Project/blob/master/multiagent/multiAgents.py

        # if next step is to win state, set the score max_int
        if successorGameState.isWin():
            return sys.maxsize

        # three things that affect the step choice
        # 1. ghost: our distance with them, scared times
        # 2. food: number, our distance with them
        # 3. power pellet: number
        # so we take them as variables for the evaluation function

        # states for successors
        # 1-1: distance with ghost
        ghostPos = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistance = []
        for pos in ghostPos:
            ghostDistance.append(manhattanDistance(newPos, pos))
        # 1-2: scared times
        totalScaredTimes = sum(newScaredTimes)
        # 2-1: number of food
        foodLeft = len(newFood.asList())
        # 2-2: distance with them
        # TODO: possible improvements
        # 3-1: number of power pellets
        # TODO: possible improvement
        # powerPellets = len(successorGameState.getCapsules())

        # states for current
        # 1-1: know the current state of ghost to see if our successor gets us further from them
        ghostPosCurrent = [
            ghost.getPosition() for ghost in currentGameState.getGhostStates()
        ]
        ghostDistanceCurrent = []
        for pos in ghostPosCurrent:
            ghostDistanceCurrent.append(manhattanDistance(newPos, pos))
        # 2-1: number of food, to see if our successor reduces the food left
        foodLeftCurrent = len(currentGameState.getFood().asList())

        # compute score
        score = successorGameState.getScore() - currentGameState.getScore()
        # reward pacman to get closer to ghost if ghost is scared (uses 1-1, 1-2)
        if totalScaredTimes > 0:
            if min(ghostDistance) > min(ghostDistanceCurrent):
                score += 200
            else:
                score -= 100
        # reward pacman to get further to ghost if ghost is not scared (uses 1-1, 1-2)
        else:
            if min(ghostDistanceCurrent) < min(ghostDistance):
                score -= 100
            else:
                score += 200
        # reward if can have less food left in successor (uses 2-1)
        if foodLeft < foodLeftCurrent:
            score += 200
        # reward if can eat a power pellet in successor (similar to using 3-1)
        if newPos in currentGameState.getCapsules():
            score += 150  # TODO: possible improvement, e.g. times powerPellets

        # penalty for stop
        if action == Directions.STOP:
            score -= 10
        # penalty for each food left
        score -= 10 * foodLeft

        return score


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        # I referenced this github repo for this function, but did minor modifications:
        # https://github.com/karlapalem/UC-Berkeley-AI-Pacman-Project/blob/master/multiagent/multiAgents.py

        def maxValue(gameState, depth):
            currentDepth = (
                depth + 1
            )  # because one agent + one all-ghost move is one round
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            maxScore = -sys.maxsize
            actions = gameState.getLegalActions(0)
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                maxScore = max(maxScore, minValue(nextState, currentDepth, 1))

            return maxScore

        def minValue(gameState, depth, agentIndex):
            if (
                gameState.isWin() or gameState.isLose()
            ):  # no need to check depth == self.depth because we have checked in maxValue
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            minScore = sys.maxsize
            for action in actions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if we have changed the gameState after going through all ghosts' actions,
                # we can now get the terminal state value or give the turn to pacman
                if agentIndex == gameState.getNumAgents() - 1:
                    minScore = min(minScore, maxValue(nextState, depth))
                else:
                    minScore = min(
                        minScore, minValue(nextState, depth, agentIndex + 1)
                    )  # after letting one ghost to do an action to the current game state, we give the turn to the next ghost

            return minScore

        # start from pacman
        pacmanIndex = 0
        actions = gameState.getLegalActions(pacmanIndex)
        maxScore = -sys.maxsize
        result = ""
        for action in actions:
            nextState = gameState.generateSuccessor(pacmanIndex, action)
            score = minValue(nextState, pacmanIndex, pacmanIndex + 1)
            if score > maxScore:
                maxScore = score
                result = action

        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # I reused my code for q2 and referenced this github repo for this function but did minor modifications:
        # https://github.com/karlapalem/UC-Berkeley-AI-Pacman-Project/blob/master/multiagent/multiAgents.py

        def maxValue(gameState, depth, alpha, beta):
            currentDepth = (
                depth + 1
            )  # because one agent + one all-ghost move is one round
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            maxScore = -sys.maxsize
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                maxScore = max(
                    maxScore, minValue(nextState, currentDepth, 1, alpha1, beta)
                )

                if maxScore > beta:
                    return maxScore
                alpha1 = max(alpha1, maxScore)

            return maxScore

        def minValue(gameState, depth, agentIndex, alpha, beta):
            if (
                gameState.isWin() or gameState.isLose()
            ):  # no need to check depth == self.depth because we have checked in maxValue
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            minScore = sys.maxsize
            beta1 = beta  # prevent from changing the beta in higher levels
            # but we can reuse this beta and put it into the min() because all lower nodes will pass this minValue node, and if we want to find min, we can just ignore all values that are larger than this beta
            for action in actions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if we have changed the gameState after going through all ghosts' actions,
                # we can now get the terminal state value or give the turn to pacman
                if agentIndex == gameState.getNumAgents() - 1:
                    minScore = min(minScore, maxValue(nextState, depth, alpha, beta1))
                else:
                    minScore = min(
                        minScore,
                        minValue(nextState, depth, agentIndex + 1, alpha, beta1),
                    )  # after letting one ghost to do an action to the current game state, we give the turn to the next ghost

                if minScore < alpha:
                    return minScore

                beta1 = min(beta1, minScore)

            return minScore

        # start from pacman
        pacmanIndex = 0
        actions = gameState.getLegalActions(pacmanIndex)
        maxScore = -sys.maxsize
        alpha = -sys.maxsize
        beta = sys.maxsize
        result = ""
        for action in actions:
            nextState = gameState.generateSuccessor(pacmanIndex, action)
            score = minValue(nextState, pacmanIndex, pacmanIndex + 1, alpha, beta)
            if score > maxScore:
                maxScore = score
                result = action

            alpha = max(
                alpha, score
            )  # update the max score among successors in this level
        return result


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
        # I reused my code for q2 and referenced this github repo for this function but did minor modifications:
        # https://github.com/karlapalem/UC-Berkeley-AI-Pacman-Project/blob/master/multiagent/multiAgents.py

        def maxValue(gameState, depth):
            currentDepth = (
                depth + 1
            )  # because one agent + one all-ghost move is one round
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            maxScore = -sys.maxsize
            actions = gameState.getLegalActions(0)
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                maxScore = max(maxScore, expectedValue(nextState, currentDepth, 1))

            return maxScore

        def expectedValue(gameState, depth, agentIndex):
            if (
                gameState.isWin() or gameState.isLose()
            ):  # no need to check depth == self.depth because we have checked in maxValue
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            expectedScore = 0
            for action in actions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if we have changed the gameState after going through all ghosts' actions,
                # we can now get the terminal state value or give the turn to pacman
                if agentIndex == gameState.getNumAgents() - 1:
                    expectedScore += maxValue(nextState, depth)
                else:
                    expectedScore += expectedValue(
                        nextState, depth, agentIndex + 1
                    )  # after letting one ghost to do an action to the current game state, we give the turn to the next ghost
            expectedScore /= len(actions)

            return expectedScore

        # start from pacman
        pacmanIndex = 0
        actions = gameState.getLegalActions(pacmanIndex)
        maxScore = -sys.maxsize
        result = ""
        for action in actions:
            nextState = gameState.generateSuccessor(pacmanIndex, action)
            score = expectedValue(nextState, pacmanIndex, pacmanIndex + 1)
            if score > maxScore:
                maxScore = score
                result = action

        return result


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    There are mainly three things that affect the step choice
    1. ghost: our distance with them, scared times
    2. food: number, our distance with them
    3. power pellet: number
    So we take them as variables for the evaluation function.

    The main idea is that
    a. No matter whether the ghosts are scared, we need to get closer to food and eat more food.
    b. Only ghosts and power pellets should be acted towards differently under the two condition: "ghosts are scared", "ghosts are not scared"

    The different score computing formula that we have for "ghosts are scared or not" can be referenced in the following code.
    More details and reasoning can be found there.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # 1-1: sum of distance with ghosts
    ghostPos = [ghost.getPosition() for ghost in newGhostStates]
    ghostDistance = [manhattanDistance(pos, newPos) for pos in ghostPos]
    sumGhostDistance = sum(ghostDistance)
    # 1-2: scared times
    sumScaredTimes = sum(newScaredTimes)
    # 2-1: number of food eaten
    foodEaten = len(newFood.asList(False))
    # 2-2: sum of distance with food
    foodDistance = [manhattanDistance(pos, newPos) for pos in newFood.asList()]
    sumFoodDistance = sum(foodDistance)
    reciporcalFoodDistance = 1 / sumFoodDistance if sumFoodDistance > 0 else 0
    # 3-1: number of power pellets
    powerPellets = len(currentGameState.getCapsules())

    # use 2-1, 2-2 (we act the same towards food no matter whether ghosts are scared)
    score = currentGameState.getScore() + foodEaten + reciporcalFoodDistance
    # act differently towards 1-1, 1-2, 3-1 under different circumstances
    if sumScaredTimes > 0:
        # when ghosts are scared, it's better to
        # 1. have longer scared times
        # 2. get closer to ghosts (provide more moving flexibility)
        # 3. have less power pellets eaten (we don't want to consume it when the ghosts are already scared)
        score += sumScaredTimes - powerPellets - sumGhostDistance
    else:
        # when ghosts are not scared, it's better to
        # 1. get further from ghosts
        # 2. have more power pellets in the successor (so that we can scare ghosts later on)
        score += sumGhostDistance + powerPellets

    return score


# Abbreviation
better = betterEvaluationFunction
