from __future__ import division

import time
import math
import random

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, action, parent, level):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.level = level
        self.goodRoute = False
        self.action = action

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy, depth=3, width = 3, simLimit = 20):
        
        self.depth = depth
        self.width = width
        self.simLimit = simLimit
        self.nodeArray = []
        self.numSims = 0

        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None ,None, 0)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
                if(self.numSims >= self.simLimit):
                    self.pruneTree()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
                if(self.numSims >= self.simLimit):
                    self.pruneTree()

        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), action, node, (node.level+1))
                node.children[action] = newNode
                if(newNode.level == self.depth):
                    self.nodeArray.append(newNode)
                    self.numSims = self.numSims + 1
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        try:
            return random.choice(bestNodes)
        except:
            _, value = random.choice(list(node.children.items()))
            return value
    
    def deleteChildren(self, node):
        if not node:
            return

        if node.goodRoute:
            for child_state in list(node.children.keys()):  # Iterate over a copy of keys
                child = node.children[child_state]
                self.deleteChildren(child)
        else:
            if node.parent:
                node.parent.children.pop(node.action)  # Remove this node from its parent's children

    def pruneTree(self):
        print('Prune')
        # Sorts array in decending order
        self.nodeArray.sort(key=lambda node: node.numVisits, reverse=True)

        bestNodes = self.nodeArray[:self.width]

        #sets parents of top to be full expanded
        for node in bestNodes:
            node.goodRoute = True
            nextNode = node.parent
            while nextNode is not None:
                nextNode.isFullyExpanded = True
                nextNode.goodRoute = True
                nextNode = nextNode.parent

        self.deleteChildren(self.root)
        self.numSims = 0


