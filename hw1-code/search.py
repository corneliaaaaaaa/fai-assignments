# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import math
import itertools
import copy
import sys


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    destinations = maze.getObjectives()

    # implement BFS
    fringe = [(start, [start])]  # (location, path)
    visited = []
    result = []
    while True:
        if not fringe:
            return []

        target, path = fringe.pop(0)
        if target in destinations:
            result.extend(path)
            destinations.remove(target)
            fringe = [(target, [target])]
            visited = []
            if not destinations:
                break
        else:
            if target not in visited:
                visited.append(target)
                for neighbor in maze.getNeighbors(target[0], target[1]):
                    if neighbor not in visited:
                        new = (neighbor, path + [neighbor])
                        fringe.append(new)

    return result


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    destinations = maze.getObjectives()

    fringe = PriorityQueue([(start, [start], 0)])  # (location, path, f)
    visited = []
    result = []
    while True:
        if not fringe:
            return []

        # select strategy
        target, path, cost = fringe.pop()

        # expand
        if target in destinations:
            result.extend(path)
            destinations.remove(target)
            fringe = PriorityQueue([(target, [target], 0)])  # (location, path, cost)
            visited = []
            if not destinations:
                break
        else:
            if target not in visited:
                visited.append(target)
                for neighbor in maze.getNeighbors(target[0], target[1]):
                    if neighbor not in visited:
                        f = (
                            len(path)
                            + 1
                            + heuristic(neighbor, destinations, "manhattan")
                        )  # f = real cost + heuristic

                        new = (
                            neighbor,
                            path + [neighbor],
                            f,
                        )
                        fringe.push(new)

    return result


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # for this question, the search logic is inspired by this stackoverflow post:
    # https://stackoverflow.com/questions/28570079/a-search-with-multiple-goals-python

    start = maze.getStart()
    destinations = maze.getObjectives()

    # search through all possible corner permutation
    orderCandidates = list(itertools.permutations(destinations, len(destinations)))
    candidateSteps = []
    for candidate in orderCandidates:
        visitOrder = [start] + list(candidate)
        path = []
        for i in range(1, len(visitOrder)):
            tmpMaze = copy.deepcopy(maze)
            tmpMaze.setStart(visitOrder[i - 1])
            tmpMaze.setObjectives([visitOrder[i]])
            partialPath = astar(tmpMaze)
            if i == 1:
                path += partialPath
            else:
                path += partialPath[1:]  # prevent repeating the corner nodes
        candidateSteps.append(path)

    # select the candidate with the shortest path
    result = min(candidateSteps, key=len)

    return result

    # we can also use the algorithm astar_multi to solve this problem, which will be faster
    # return astar_multi(maze)


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # The general idea of my function is inspired by this github repo:
    # https://github.com/adityavgupta/ECE448-CS440/blob/master/mp1/search.py
    # I comprehended it then use my own coding logic to implement it.
    start = maze.getStart()
    destinations = maze.getObjectives()

    # find all possible objective pairs
    objPairs = []
    for i in range(len(destinations)):
        for j in range(i + 1, len(destinations)):
            objPairs.append((i, j))

    # compute costMap, which stores the real cost (distance) it takes to go from one objective to the other using A* search algorithm
    costMap = {}
    for pair in objPairs:
        tmpMaze = copy.deepcopy(maze)
        tmpMaze.setStart(destinations[pair[0]])
        tmpMaze.setObjectives([destinations[pair[1]]])
        path = astar(tmpMaze)
        costMap[pair] = (
            len(path) - 1
        )  # cost is the gap between each step in a path, so need to - 1

    startNode = (
        start,
        [start],
        mstHeuristic(start, destinations, destinations, costMap),
        tuple(destinations),
    )
    fringe = PriorityQueue([startNode])
    costForCurrentNode = {
        (start, tuple(destinations)): 0
    }  # stores the costs that takes so far since traveling from the starting point
    result = []
    while True:
        # I think the structure is pretty much the same
        # it's just more complicated in the for-loop for neighbors and we need a new state representation
        if not fringe.queue:
            return []

        # select strategy
        target, path, cost, remainingDestinations = fringe.pop()
        currentItem = (target, tuple(remainingDestinations))

        if not remainingDestinations:
            result = path
            break

        for neighbor in maze.getNeighbors(target[0], target[1]):
            # update the remaining destinations if we will go for the neighbor node
            tmp = []
            for i in remainingDestinations:
                if neighbor != i:
                    tmp.append(i)
            rd = tuple(tmp)

            neighborItem = (neighbor, rd)
            # if the new path we found to get to the same neighbor is longer, don't need to do anything
            if (
                neighborItem in costForCurrentNode
                and costForCurrentNode[neighborItem]
                <= costForCurrentNode[currentItem] + 1
            ):
                continue
            else:
                # update cost
                costForCurrentNode[neighborItem] = costForCurrentNode[currentItem] + 1

                # update f
                prev_f = cost
                f = costForCurrentNode[neighborItem] + mstHeuristic(
                    neighbor, destinations, rd, costMap
                )

                # if the new f < previous f, the heuristic will be inconsistent.
                # Since it should never decrease, we need to use max() to adjust it.
                f = max(prev_f, f)

                new = (neighbor, path + [neighbor], f, rd)
                fringe.push(new)

    return result


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return bfs(maze)


# some classes and utility functions
class PriorityQueue:
    def __init__(self, q=[]):
        self.queue = q

    def pop(self):
        if not self.queue:
            return None
        else:
            return self.queue.pop(0)

    def push(self, item):
        self.queue.append(item)
        self.queue.sort(key=lambda x: x[2])


def heuristic(current: tuple, destinations: list, mode: str):
    result = []

    for d in destinations:
        distance = 0
        if mode == "manhattan":
            distance = manhattanDistance(current, d)
        elif mode == "euclidean":
            distance = euclideanDistance(current, d)
        result.append(distance)

    return min(result)


def manhattanDistance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def euclideanDistance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def mstHeuristic(current, allDestinations, remainingDestinations, costMap):
    """
    Since finding a shortest path that goes through all objectives is just like the
    travel salesman problem, we use Minimum Spanning Tree as a part of the heuristic
    to solve the problem.

    For the search heuristic to be admissible, it has to be the lower bound of the
    true cost required to meet goal state. Therefore, our h(n) will be:
    h(n) = min(manhattan(current + the nearest objective of current node))
    + sum of weights of the MST formed by all objectives

    An intuitive way to understand this logic: imagine that the distance from current
    node to goal state will be gapped by:
    1. Current node goes to an objective first.
    2. Starts to follow the branch of the MST to traverse all other objectives.
    """
    if not remainingDestinations:
        return 0

    mstWeights = 0
    destinationsInTree_ID = [
        allDestinations.index(remainingDestinations[0])
    ]  # use the destination's index in the list to represent itself
    restDestinations_ID = []
    for d in remainingDestinations[1:]:
        restDestinations_ID.append(allDestinations.index(d))

    # compute the sum of weights of the MST
    while len(destinationsInTree_ID) != len(remainingDestinations):
        # since we already have one destination in the MST,
        # we can use any pair and add the destination into the MST, and the MST can be maintained.
        pathWithMinCosts = []

        # we try (n - m) * m pairs each time, and only add the node with the minimum cost to the tree
        for x in destinationsInTree_ID:
            minCost = sys.maxsize
            nodeWithMinCost = None
            for y in restDestinations_ID:
                # use the correct order (small index first) for the pair expression
                pair = (x, y)
                if x > y:
                    pair = (y, x)

                if costMap[pair] < minCost:
                    minCost = costMap[pair]
                    nodeWithMinCost = y  # y is the node that is not in the tree yet
            pathWithMinCosts.append((minCost, nodeWithMinCost))

        targetDestination = min(pathWithMinCosts)
        restDestinations_ID.remove(targetDestination[1])
        destinationsInTree_ID.append(targetDestination[1])
        mstWeights += targetDestination[0]

    # compute final heuristics
    manhattan = []
    for d in remainingDestinations:
        manhattan.append(manhattanDistance(d, current))

    result = min(manhattan) + mstWeights

    return result
