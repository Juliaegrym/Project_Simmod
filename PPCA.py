import random as rnd
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


# 1 = prey, 2 = low_pred, 3 = top_pred
# Vector of desired species
species = [0, 1, 2, 3]
BP = [None, 0.3, 0.4, 0.2]
DP = [None, None, 0.001, 0.001]
CP = [None, 0.9, 0.7, None]

frames = 100
dimensions = 25


class Cell:
    def __init__(self, kind):
        self.kind = kind    # 0 = empty, 1 = prey, 2 = low_pred, 3 = top_pred
        self.state = 1      # States of cell, 1 = before attack, 2 = after top attack, 3 = after low attack
        self.birthprob = None
        self.deathprob = None
        self.catchprob = None

        if kind == 0:
            self.birthprob = BP[0]  # is decided from name
            self.deathprob = DP[0]  # is decided from name
            self.catchprob = CP[0]  # is decided from name
        elif kind == 1:
            self.birthprob = BP[1]  # is decided from name
            self.deathprob = DP[1]  # is decided from name
            self.catchprob = CP[1]  # is decided from name
        elif kind == 2:
            self.birthprob = BP[2]  # is decided from name
            self.deathprob = DP[2]  # is decided from name
            self.catchprob = CP[2]  # is decided from name
        elif kind == 3:
            self.birthprob = BP[3]  # is decided from name
            self.deathprob = DP[3]  # is decided from name
            self.catchprob = CP[3]  # is decided from name


def healthy(place):
    if place.kind is 0:
        assert place.birthprob == BP[0]
        assert place.deathprob == DP[0]
    if place.kind is 1:
        assert place.birthprob == BP[1]
        assert place.deathprob == DP[1]
    if place.kind is 2:
        assert place.birthprob == BP[2]
        assert place.deathprob == DP[2]
    if place.kind is 3:
        assert place.birthprob == BP[3]
        assert place.deathprob == DP[3]


def check_neighbours(i, j, grid):

    prey = 0
    prey_indices = []
    low_pred = 0
    low_pred_indices = []
    top_pred = 0
    top_pred_indices = []

    dim = len(grid)

    for k in range(3):
        for m in range(3):
            if (k == 2) and (m == 2):
                spec = 0                        # This is the cell in considerations
            else:
                s = i-1+k
                t = j-1+m

                # Checks if index is out of bounds --> index dim = 1
                # The grid is modelled as a torus, the outer bounds are connected
                if s == dim:
                    if t == dim:
                        s = 0
                        t = 0
                    else:
                        s = 0
                elif t == dim:
                    t = 0
                spec = grid[s][t].kind  # Checks the species of the cell

                if spec == 1:
                    prey += 1
                    prey_indices.append([s, t])
                elif spec == 2:
                    low_pred += 1
                    low_pred_indices.append([s, t])
                elif spec == 3:
                    top_pred += 1
                    top_pred_indices.append([s, t])

    # print(prey_indices, low_pred_indices, top_pred_indices)

    return prey, low_pred, top_pred, prey_indices, low_pred_indices, top_pred_indices


def check_eaten_neighbours(i, j, grid):

    low_eaten = 0
    top_eaten = 0
    food = 0

    dim = len(grid)

    for k in range(3):
        for m in range(3):
            if (k == 2) and (m == 2):
                spec = 0                        # Defines the species of the cell
            else:
                s = i-1+k
                t = j-1+m
                if s == dim:
                    if t == dim:
                        spec = grid[1][1].kind  # Checks the species of the cell
                        food = grid[1][1].state  # Checks the state of the cell
                    else:
                        spec = grid[1][j-1+m].kind  # Checks the species of the cell
                        food = grid[1][j-1+m].state  # Checks the state of the cell

                elif t == dim:
                    spec = grid[i-1+k][1].kind  # Checks the species of the cell
                    food = grid[i-1+k][1].state  # Checks the state of the cell

                else:
                    spec = grid[i-1+k][j-1+m].kind      # Checks the species of the cell
                    food = grid[i-1+k][j-1+m].state     # Checks the state of the cell

            if (spec == 2) and (food == 3):
                low_eaten += 1
            elif (spec == 3) and (food == 2):
                top_eaten += 1

    return low_eaten, top_eaten


def counter(grid, dim):

    # Before counting all n.o. species are set to zero
    counting_vector = [0, 0, 0, 0]      # [empty, prey, low_pred, top_pred]

    # Iterates over grid to count different species
    for i in range(dim):
        for j in range(dim):
            num = grid[i][j].kind       # Index of counting vector = species reference number
            counting_vector[num] += 1   # Ticks up when a species of that index/kind is found

    return counting_vector


def attack(place, i, j, grid):

    # Checks what animals are inhabiting surrounding cells
    [prey, low_pred, top_pred, prey_indices, low_pred_indices, top_pred_indices] = check_neighbours(i, j, grid)

    # Cell contains a prey
    if (place.kind == 1) and (low_pred > 0):

        # Probability of prey surviving low level predator
        survive_prob = (1 - Cell(1).catchprob) ** low_pred

        # checks if low predator eats prey
        # assigns state = 3 to eaten by low level predator
        r = rnd.random()
        if survive_prob < r:
            place = Cell(0)
            place.state = 3

    # Cell contains a low level predator
    elif place.kind == 2:

        # Probability of prey surviving top level predator
        survive_prob = (1 - Cell(2).catchprob) ** top_pred

        # checks if top level predator eats low level predator
        # assigns state = 2 to eaten by top level predator
        r = rnd.random()
        if (top_pred > 0) and (survive_prob < r):
            place = Cell(0)
            place.state = 2

        # Checks if low level predator attacks prey in neighbourhood
        if (1 - ((1 - Cell(1).catchprob) ** prey) >= r) and (place.kind == 2) and (prey > 0):
            place.state = 3         # Indicates the predator has eaten

            # Removes one of the preys (that has now been eaten by the low predator)
            remove_index = int(rnd.random() * len(prey_indices))

            # Changes a random prey grid cell in neighbourhood to an empty cell with state = 3
            grid[prey_indices[remove_index][0]][prey_indices[remove_index][1]] = Cell(0)
            grid[prey_indices[remove_index][0]][prey_indices[remove_index][1]].state = 3

    # Cell contains top level predator
    elif place.kind == 3:
        r = rnd.random()
        if (1 - (1 - Cell(2).catchprob) ** low_pred) >= r:
            place.state = 2

            # Removes one of the low predators (that has now been eaten by the top predator)
            remove_index = int(rnd.random() * len(low_pred_indices))

            # Changes a random prey grid cell in neighbourhood to an empty cell with state = 3
            grid[low_pred_indices[remove_index][0]][low_pred_indices[remove_index][1]] = Cell(0)
            grid[low_pred_indices[remove_index][0]][low_pred_indices[remove_index][1]].state = 2

    healthy(place)

    return place, grid


def reproduction(place, i, j, grid):

    # Checks what animals are inhabiting surrounding cells
    [prey, low_pred, top_pred, prey_indices, low_pred_indices, top_pred_indices] = check_neighbours(i, j, grid)
    [low_eaten, top_eaten] = check_eaten_neighbours(i, j, grid)

    if place.kind == 2:

        # Checks if low level predator dies by natural causes
        r = rnd.random()
        if Cell(2).deathprob >= r:
            place = Cell(0)
            place.state = 1

    elif place.kind == 3:

        # Checks if top level predator dies by natural causes
        r = rnd.random()
        if Cell(3).deathprob >= r:
            place = Cell(0)
            place.state = 1

    if place.kind == 0:

        # Checks if the empty cell was empty from the start
        # If no low level predators, preys can reproduce with some probability
        if place.state == 1:
            repr_prob = 1 - (1 - Cell(1).birthprob) ** prey
            r = rnd.random()
            if (prey > 0) and (low_pred == 0) and (repr_prob >= r):
                place = Cell(1)     # Sets cells species/kind to prey = 1

        # Checks if the cell is empty due to top level predator attack (state = 2)
        # Top level predators that have eaten can reproduce with some probability
        elif place.state == 2:
            repr_prob = 1 - (1 - Cell(3).birthprob) ** top_eaten
            r = rnd.random()
            if (top_eaten > 0) and (repr_prob >= r):
                place = Cell(3)     # Sets cells species/kind to top level predator = 3

        # Checks if the cell is empty due to low level predator attack (state = 3)
        # If no top level predators, low level predators that have eaten can reproduce with some probability
        elif place.state == 3:
            repr_prob = 1 - (1 - Cell(2).birthprob) ** low_eaten
            r = rnd.random()
            if (low_eaten > 0) and (top_pred == 0) and (repr_prob >= r):
                place = Cell(2)     # Sets cells species/kind to low level predator = 2

            else:
                place.state = 1

    healthy(place)

    if place.state != 1:
        place.state = 1

    return place


def define_grid(dim):
    # Grid of Cellular Automaton
    grid = [[0]*dim]*dim

    for i in range(dim):
        for j in range(dim):
            num = int(rnd.random()*4)
            grid[i][j] = Cell(int(num))

    return grid


def update(dim, grid):
    # Performs one sweep of updating all cells in grid, based on different rules

    new_grid = [[0]*dim]*dim

    for i in range(dim):
        for j in range(dim):
            state = grid[i][j]
            if state.kind != 0:
                [state, grid] = attack(state, i, j, grid)
            new_grid[i][j] = state

    for k in range(dim):
        for l in range(dim):
            state = new_grid[k][l]
            if state.kind != 1:
                state = reproduction(state, k, l, grid)
            new_grid[k][l] = state

    return new_grid


def cellular_automaton(dim, steps):
    # Function to initialize a grid and call CA-update function
    # Returns the grid after steps amount of steps

    grid = define_grid(dim)
    # ans = counter(grid, dim)
    # print('Starts with: ' + str(ans))

    for i in range(steps):
        grid = update(dim, grid)
        # ans = counter(grid, dim)
        # print('After returning value: ' + str(ans))

    return grid


def run_several_ca():
    # Iterates over tries ans performs cellular automaton many times for same initial values
    tries = 50
    i = 0
    prey_list = np.zeros(tries)
    low_pred_list = np.zeros(tries)
    top_pred_list = np.zeros(tries)

    for tri in range(tries):
        ca = cellular_automaton(dimensions, frames)
        ans = counter(ca, dimensions)
        prey_list[tri] = ans[1]
        low_pred_list[tri] = ans[2]
        top_pred_list[tri] = ans[3]
        print(i)
        i += 1

    plot_statistics(prey_list, low_pred_list, top_pred_list, tries)


def plot_statistics(prey_list, low_pred_list, top_pred_list, x_value):
    # Plots animal populations
    plt.plot(prey_list, label = 'Population of prey')
    plt.plot(low_pred_list, label = 'Population of low level predators')
    plt.plot(top_pred_list, label = 'Population of top level predators')
    plt.xlabel(str(x_value))
    plt.ylabel('N.o. animals')
    plt.legend()
    plt.show()


run_several_ca()
