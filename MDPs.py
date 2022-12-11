import numpy as np

# Arguments
REWARDS = [[-0.1, -0.1, -0.05], [-0.1, -0.1, 1]]
#REWARDS = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
DISCOUNT = 0.999
#DISCOUNT = 0.9
EPSILON = 10e-3

# Set up the initial environment
NUM_ACTIONS = 5
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1), (0, 0)] # [SOUTH, WEST, NORTH, EAST, DO NOTHING]
Q = [["SOUTH", 0], ["WEST", 0], ["NORTH", 0], ["EAST", 0], ["DO NOTHING", 0]]
initU = [[0, 0, 0], [0, 0, 1]]
#initU = [[0, 0, 0, 1], [0, "x", 0, -1], [0, 0, 0, 0]]
NUM_ROW = len(initU)
NUM_COL = len(initU[0]) # or length of any other row

# Visualization
def printEnvironment(arr, policy=False):
    if policy:
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if initU[r][c] == 1 or initU[r][c] == -1:
                    arr[r][c] = "GOAL"
                elif initU[r][c] == "x":
                    arr[r][c] = "WALL"
                else:
                    arr[r][c] = ["SOUTH", "WEST", "NORTH", "EAST", "DO NOTHING"][arr[r][c]]
    print(np.array(arr))

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or U[newR][newC] == "x": # Collide with the boundary or the wall
        return U[r][c] #return nextU[r][c]
    else:
        return U[newR][newC] #return nextU[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, action):
    if action == 4:
        u = 0
        u += 1 * DISCOUNT * getU(U, r, c, action)
        Q[action][1] = round(u, 2)
        u += REWARDS[r][c]
    else:
        u = 0
        u += 0.05 * DISCOUNT * getU(U, r, c, (action-1)%4)
        #u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
        u += 0.9 * DISCOUNT * getU(U, r, c, action)
        #u += 0.8 * DISCOUNT * getU(U, r, c, action)
        u += 0.05 * DISCOUNT * getU(U, r, c, (action+1)%4)
        #u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
        Q[action][1] = round(u, 2)
        u += REWARDS[r][c]
    return u

def valueIteration(U):
    iterationNum = 0
    print("During the value iteration:\n")
    while True:
        iterationNum += 1
        print("Iteration " + str(iterationNum) + ":")
        nextU = [[0, 0, 0], [0, 0, 1]]
        #nextU = [[0, 0, 0, 1], [0, "x", 0, -1], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if initU[r][c] == 1 or initU[r][c] == -1 or initU[r][c] == "x":
                    continue
                # Bellman update
                else:
                    nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)])
                print("Expected Utilities for taking each Action in state (" + str(r) + ", " + str(c) + "):")
                print(np.array(Q))
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        print("The U is:")
        printEnvironment(U)
        print("The policy is:")
        policy = getPolicy(U); printEnvironment(policy, True)
        print()  # Print a blank line just for better visualization
        if error < EPSILON * (1-DISCOUNT) / DISCOUNT:
            break
    return U

# Get the optimal policy from U
def getPolicy(U):
    policy = [[0 for j in range(NUM_COL)] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if U[r][c] == 1 or U[r][c] == -1 or U[r][c] == "x":
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy

def get_state_utilities(epsilon, rewards):
    EPSILON = epsilon
    REWARDS = rewards

    # Print the initial environment
    print("The initial U is:")
    printEnvironment(initU)
    print() # Print a blank line just for better visualization
    print() # Print a blank line just for better visualization

    # Value iteration
    U = valueIteration([[0, 0, 0], [0, 0, 1]])
    #U = valueIteration([[0, 0, 0, 1], [0, "x", 0, -1], [0, 0, 0, 0]])

    # Get the optimal policy from U and print it
    policy = getPolicy(U)
    print()  # Print a blank line just for better visualization
    print("The optimal policy is:")
    printEnvironment(policy, True)

if __name__ == "__main__":
    get_state_utilities(0.1, [[-0.1, -0.1, -0.05], [-0.1, -0.1, 1]])
    #get_state_utilities(10**(-3), [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]])
