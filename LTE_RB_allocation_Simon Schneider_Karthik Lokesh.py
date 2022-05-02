# LTE_RB_allocation_Simon Schneider_Karthik Lokesh.py
#
# Performs allocation of resource blocks (RBs) in a LTE network where RBs have to be shared due to a higher number of devices than RBs
# Solution of network optimization project for Traffic Engineering lecture
# Authors: Simon Malte Schneider (21597305) and Karthik Lokesh (54937)

# README: How to run: simply type "$ python LTE_RB_allocation.py" into the console in the directory where you save the file
# You will be prompted wth the option to choose between network a, b, or randomized

from gurobipy import *
import numpy
import scipy.constants
import sys
import random
import networkx as nx
from matplotlib import pyplot as plt
from timeit import default_timer as timer

# Network layout for task 2a
def networkA():
    baseStationPosition = (50,40)
    d2dSenderPositions = [(0,80), (100,80)]
    cellularPositions = [(0,0), (100,0)]
    d2dReceiverPositions = [(0,90), (100,90)]       # NOTE: order has to be corresponding to order of senders (d2dSenderPositions[i] will communicate with d2dReceiverPositions[i])
    numRBs = 2
    return baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs

# Network layout for task 2b
def networkB():
    baseStationPosition = (50,40)
    d2dSenderPositions = [(100,0), (0,80)]
    cellularPositions = [(0,0), (100,80)]
    d2dReceiverPositions = [(100,-10), (0,90)]       # NOTE: order has to be corresponding to order of senders (d2dSenderPositions[i] will communicate with d2dReceiverPositions[i])
    numRBs = 2
    return baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs

# Random network with n nodes located in a circular network cell
def randomNetwork(n):
    diam = 1         #diameter of circle in kilometers
    numRBs = int(n / 2)
    baseStationPosition = (0,0)
    d2dSenderPositions = []
    cellularPositions = []
    d2dReceiverPositions = []

    # D2D senders
    counter = 0
    while counter < (n / 2):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        d = dist(x, y, baseStationPosition[0], baseStationPosition[1])
        if d > (diam / 2):
            continue
        else:
            d2dSenderPositions.append((x,y))
            counter += 1

    # Cellular devices
    counter = 0
    while counter < (n / 2):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        d = dist(x, y, baseStationPosition[0], baseStationPosition[1])
        if d > (diam / 2):
            continue
        else:
            cellularPositions.append((x,y))
            counter += 1

    # D2D receivers
    counter = 0
    while counter < (n / 2):
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        d = dist(x, y, baseStationPosition[0], baseStationPosition[1])
        if d > (diam / 2):
            continue
        else:
            d2dReceiverPositions.append((x,y))
            counter += 1

    return baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs

# Prints out graph
def drawGraph(G, baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs, numDevices):
    # Add nodes, put their positions into lists of each type, and add labels
    labels = {}
    G.add_node("base", pos = baseStationPosition)

    nodesCellular = []
    nodesSender = []
    nodesReceiver = []
    for i in range(len(cellularPositions)):
        G.add_node(i, pos = cellularPositions[i])
        labels[i] = ("c" + str(i))
        nodesCellular.append(i)
    for i in range(len(d2dSenderPositions)):
        G.add_node((i + len(cellularPositions)), pos = d2dSenderPositions[i])
        labels[i + len(cellularPositions)] = ("s" + str(i))
        nodesSender.append(i + len(cellularPositions))
    for i in range(len(d2dReceiverPositions)):
        G.add_node((i + len(cellularPositions) + len(d2dSenderPositions)), pos = d2dReceiverPositions[i])
        labels[i + len(cellularPositions) + len(d2dSenderPositions)] = ("r" + str(i))
        nodesReceiver.append(i + len(cellularPositions) + len(d2dSenderPositions))

    # Add edges (= communication pairs)
    edgesD2D = []
    edgesCellular = []
    for i in range(len(d2dSenderPositions)):
        G.add_edge((i + len(cellularPositions)), (i + len(cellularPositions) + len(d2dSenderPositions)))
        edgesD2D.append(((i + len(cellularPositions)), (i + len(cellularPositions) + len(d2dSenderPositions))))
    for i in range(len(cellularPositions)):
        G.add_edge("base", i)
        edgesCellular.append(("base", i))
    pos = nx.get_node_attributes(G, 'pos')

    # Adding arcs for RB
    ax = plt.gca()
    for r in range(numRBs):
        for i in range(numDevices):
            for j in range(i, numDevices):
                if y[i,j,r].X == 1:
                    ax.annotate("", xy = pos[i], xycoords = 'data', xytext = pos[j], textcoords = 'data', arrowprops = dict(arrowstyle="-", color = "orangered", shrinkA = 10, shrinkB = 10, patchA = None, patchB = None, connectionstyle = "arc3, rad = 0.2",),)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist = ["base"], node_color = "black")
    nx.draw_networkx_nodes(G, pos, nodelist = nodesCellular, node_color = '#566573')
    nx.draw_networkx_nodes(G, pos, nodelist = nodesSender, node_color = '#1371ca')
    nx.draw_networkx_nodes(G, pos, nodelist = nodesReceiver, node_color = '#7cb6ec')

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist = edgesD2D, edge_color = '#c4e0fa', width = 1.0)
    nx.draw_networkx_edges(G, pos, edgelist = edgesCellular, edge_color = '#cfd1d2', width = 1.0)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size = 8)

    # Draw circle if random network
    if numRBs > 2:
        circle = plt.Circle((0,0), 500, color = 'black', fill = False, clip_on = False)
        ax.add_artist(circle)
    ax.axis("off")
    plt.show()
    return

# Calculates distance between two nodes with coordinates (x1,y1) and (x2,y2). Output in kilometers
def dist(x1, y1, x2, y2):
    d = math.sqrt(((x1 - x2) ** 2) + ((y1 -y2) ** 2))   # in meters
    d *= 0.001                                          # in kilometers
    return d

# Calculates free space path loss over distance d (in kilometers). Output in dB
def fspl(d):
    f = 2100000000000              # 2.1GHz
    loss = 20 * math.log10(d) + 20 * math.log10(f) + 20 * math.log10((4 * scipy.constants.pi) / scipy.constants.c)
    return loss

# Calculates the interference occuring when two senders share a resource block. Senders given by indice in D. Output in mW
def interference(i, j):
    loss = 0
    distance = 0
    transPower = 26                 # in dBm

    # Setting correct senders and receivers
    if (i == j):
        return 0
    if deviceType[i] == 1:          # sender is cellular, so receiver is d2d
        sender = devicePositions[i]
        receiver = d2dReceiverPositions[j % numCellular]
    elif deviceType[i] == 0:        # sender is D2d
        if deviceType[j] == 1:      # second device is cellular, so receiver is base station
            sender = devicePositions[i]
            receiver = baseStationPosition
        elif deviceType[j] == 0:
            sender = devicePositions[i]
            receiver = d2dReceiverPositions[j % numCellular]

    # Actual calculation
    distance = dist(sender[0], sender[1], receiver[0], receiver[1])
    loss = fspl(distance)
    interference = transPower - loss            # in dBm
    interference = convert(interference)        # in mW
    return interference

# Converts value v from dBm into mW
def convert(v):
    v = 10 ** (v / 10)
    return v

# Heuristic approach, greedy algorithm
def heuristic(numDevices, numRBs, numCellular, numD2D):
    t0 = timer()
    totalInterferenceHeuristic = 0
    heuristicSolution = numpy.zeros((numDevices, numRBs), dtype = int)
    availableRB = list(range(numRBs))

    # First, assign an RB to each cellular user
    for i in range(numCellular):
        heuristicSolution[i, i] = 1

    # For each D2D user, check all available RBs for the one with lowest additional interference
    for i in range(numD2D):
        lowestInterference = interference((numCellular + i), availableRB[0]) + interference(availableRB[0], (numCellular + i))
        bestRB = None
        for r in availableRB:
            currentInterference = interference((numCellular + i), r) + interference(r, (numCellular + i))
            if currentInterference <= lowestInterference:
                lowestInterference = currentInterference
                bestRB = r

        # Remove best RB from available ones, mark pair in solution, and add interference to total interference
        availableRB.remove(bestRB)
        heuristicSolution[(numCellular + i), bestRB] = 1
        totalInterferenceHeuristic += lowestInterference

    t1 = timer()
    executionTime = t1 - t0

    # Print out like Gurobis solution
    print("\n\tAllocation of resource blocks by heuristic:")
    print("\n\t\t    ", end = "")
    for i in range(numCellular):
        print("|c" + str(i) + " ", end = "")
    for i in range(numD2D):
        print("|s" + str(i) + " ", end = "")
    print("|", end = "")
    for i in range(numRBs):
        if i < 10:
            print("\n\t\tb" + str(i) + "  |", end = "")
        else:
            print("\n\t\tb" + str(i) + " |", end = "")
        for j in range(numDevices):
            if heuristicSolution[j, i] == 0:
                print("   |", end = "")
            else:
                print(" X |", end = "")
    print("\n\n\tTotal interference: " + str(totalInterferenceHeuristic))
    print("\tRuntime: " + str(executionTime))

    return executionTime, totalInterferenceHeuristic

if __name__ == "__main__":
### Data ########################
    # User interface
    dec = input("\n\tWhich network do you want to inspect?\n\n \
                Network for task 2a -> enter a\n \
                Network for task 2b -> enter b\n \
                Random network -> enter r\n")
    if dec == "a":
        baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs = networkA()
    elif dec == "b":
        baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs = networkB()
    elif dec == "r":
        n = int(input("\n\tPlease enter the number of nodes: "))
        baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs = randomNetwork(n)
    else:
        print("\tInput incorrect, please run again.\n")
        sys.exit()

    # Values for number of devices
    devicePositions = d2dSenderPositions + cellularPositions
    numCellular = len(cellularPositions)
    numD2D = len(d2dSenderPositions)
    numDevices = len(devicePositions)
    numReceivers = len(d2dReceiverPositions)

    # Set D of device types
    deviceType= []
    for i in range(numD2D):
        deviceType.append(0)
    for i in range(numCellular):
        deviceType.append(1)

    G = nx.Graph()
    m = Model("LTE resource block allocation")

### Decision variables #######################################
    # NOTE: the matrix of decision variables is of the following form, where dn/cn is D2D/cellular sender n, RBn is resource block n, and x_ir is 1 if device i is allocated to RBr and 0 otherwise
    # _____________________________________
    #|    |  c0  c1  ... cn d0   d1 ... dn |
    #| RB0| x_ir                           |
    #| RB1|                                |
    #| ...|                                |
    #|_RBn|________________________________|

    x = tupledict()
    for i in range(numDevices):
        for r in range(numRBs):
            x[i,r] = m.addVar(vtype=GRB.BINARY, name='x[d' + str(i) + ', b' + str(r) + ']')   # 1 if RB r is allocated to device i, 0 otherwise

    y = tupledict()
    for i in range(numDevices):
        for j in range(numDevices):
            for r in range(numRBs):
                y[i,j,r] = m.addVar(vtype=GRB.BINARY, name='y[d' + str(i) + ', d' + str(j) + ", b" + str(r) + ']')   # product of x[i,r] and x[j,r]

### Objective function ########################################
    totalInterference = 0;
    for i in range(numDevices):
        for j in range(numDevices):
            for r in range(numRBs):
                totalInterference += y[i,j,r] * interference(i,j)

    m.setObjective(totalInterference, GRB.MINIMIZE)

### Constraints ##################################################
    # Assign at most 2 devices to a single RB
    for r in range(numRBs):
        assignedDevices = 0
        for i in range(numDevices):
            assignedDevices += x[i,r]
        m.addConstr(assignedDevices <= 2)

    # Assign at most one cellular user to a single RB
    for r in range(numRBs):
        assignedCellular = 0
        for i in range(numDevices):
            assignedCellular += x[i,r] * deviceType[i]
        m.addConstr(assignedCellular <= 1)

    # Assign at least one RB to each device
    for i in range(numDevices):
        assignedRBs = 0
        for r in range(numRBs):
            assignedRBs += x[i,r]
        m.addConstr(assignedRBs >= 1)

    # Constraints for linearization
    for r in range(numRBs):
        for i in range(numDevices):
            for j in range(numDevices):
                m.addConstr(y[i,j,r] == x[i,r] * x[j,r])
                m.addConstr(y[i,j,r] <= x[i,r])
                m.addConstr(y[i,j,r] <= x[j,r])
                m.addConstr(y[i,j,r] >= x[i,r] + x[j,r] - 1)

    m.optimize()

# Print out solution
    print("\n\tAllocation of resource blocks by Gurobi:")
    print("\n\t\t    ", end = "")
    for i in range(numCellular):
        print("|c" + str(i) + " ", end = "")
    for i in range(numD2D):
        print("|s" + str(i) + " ", end = "")
    print("|", end = "")
    for i in range(numRBs):
        if i < 10:
            print("\n\t\tb" + str(i) + "  |", end = "")
        else:
            print("\n\t\tb" + str(i) + " |", end = "")
        for j in range(numDevices):
            if x[j,i].X == 0:
                print("   |", end = "")
            else:
                print(" X |", end = "")
    print("\n\n\tTotal interference: " + str(m.objVal))
    print("\tRuntime: " + str(m.Runtime))

    executionTimeHeuristic, totalInterferenceHeuristic = heuristic(numDevices, numRBs, numCellular, numD2D)
    print("\n\tDifference in total interference: " + str(m.objVal - totalInterferenceHeuristic))
    print("\tDifference in execution time: " + str(m.Runtime - executionTimeHeuristic))

# Print out graph
    drawGraph(G, baseStationPosition, d2dSenderPositions, cellularPositions, d2dReceiverPositions, numRBs, numDevices)
