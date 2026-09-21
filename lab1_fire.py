"""
Lab 1: Spread of Forest Fires and Infectious Disease

This script contains the forest-fire model, validation tests,
wildfire experiments, controlled-burn experiment, and disease model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Cell states
BARE = 1
FOREST = 2
FIRE = 3

# Task 1

def initialize_forest(nx, ny):
    """
    Create a forest grid with all cells forested.
    """
    forest = np.full((ny, nx), FOREST, dtype=int)

    # Start a fire in the center
    center_y = ny // 2
    center_x = nx // 2
    forest[center_y, center_x] = FIRE

    return forest


def fire_step(forest, prob_spread):
    """
    Advance the forest-fire model by one time step.
    Fire spreads only to the four orthogonal neighbors.
    """
    ny, nx = forest.shape

    # Copy the current forest so newly ignited cells do not spread
    # until the next iteration
    new_forest = forest.copy()

    for y in range(ny):
        for x in range(nx):

            if forest[y, x] == FIRE:

                # Burning cell becomes bare/burnt
                new_forest[y, x] = BARE

                # Check the four neighboring cells
                neighbors = [
                    (y + 1, x),     # up
                    (y - 1, x),     # down
                    (y, x - 1),     # left
                    (y, x + 1)      # right
                ]

                for ny_neighbor, nx_neighbor in neighbors:

                    # Make sure neighbor is inside the grid
                    if (
                        0 <= ny_neighbor < ny
                        and 0 <= nx_neighbor < nx
                    ):

                        # Fire can only spread to forested cells
                        if forest[ny_neighbor, nx_neighbor] == FOREST:

                            if np.random.random() < prob_spread:
                                new_forest[ny_neighbor, nx_neighbor] = FIRE

    return new_forest


def plot_forest(forest, iteration):
    """
    Plot the forest grid using the coordinate convention
    shown in the lab: (0,0) is the lower-left cell.
    """

    forest_cmap = ListedColormap(
        ['tan', 'darkgreen', 'firebrick']
    )

    fig, ax = plt.subplots(1, 1)

    ax.pcolor(
        forest,
        cmap=forest_cmap,
        vmin=1,
        vmax=3,
        edgecolors='black',
        linewidths=1
    )

    # Put tick labels at the centers of cells
    ax.set_xticks(np.arange(forest.shape[1]) + 0.5)
    ax.set_yticks(np.arange(forest.shape[0]) + 0.5)

    # Label cells using their Python indices
    ax.set_xticklabels(np.arange(forest.shape[1]))
    ax.set_yticklabels(np.arange(forest.shape[0]))

    ax.set_title(f"Iteration {iteration}")

    ax.set_aspect('equal')


# Validation: 3*3 grid

nx = 3
ny = 3

prob_spread = 1.0

# Iteration 0
forest = initialize_forest(nx, ny)

print("Iteration 0")
print(forest)

plot_forest(forest, 0)


# Iteration 1
forest = fire_step(forest, prob_spread)

print("\nIteration 1")
print(forest)

plot_forest(forest, 1)


# Iteration 2
forest = fire_step(forest, prob_spread)

print("\nIteration 2")
print(forest)

plot_forest(forest, 2)


# Additional validation: 3*5 grid

nx = 5
ny = 3

prob_spread = 1.0

# Iteration 0
forest = initialize_forest(nx, ny)

print("\n3x5 - Iteration 0")
print(forest)

plot_forest(forest, 0)


# Iteration 1
forest = fire_step(forest, prob_spread)

print("\n3x5 - Iteration 1")
print(forest)

plot_forest(forest, 1)


# Iteration 2
forest = fire_step(forest, prob_spread)

print("\n3x5 - Iteration 2")
print(forest)

plot_forest(forest, 2)

plt.show()


# Task 2

def initialize_random_forest(nx, ny, prob_bare, prob_ignite):
    """
    Initialize a forest for Task 2.

    """
    # Start with all cells forested
    forest = np.full((ny, nx), FOREST, dtype=int)

    # Randomly create bare cells
    is_bare = np.random.rand(ny, nx) < prob_bare
    forest[is_bare] = BARE

    # Randomly ignite cells that are still forested
    is_fire = (
        (np.random.rand(ny, nx) < prob_ignite)
        & (forest == FOREST)
    )

    forest[is_fire] = FIRE

    return forest

# nx = 20
# ny = 20

# prob_bare = 0.2
# prob_ignite = 0.05

# forest = initialize_random_forest(
#     nx,
#     ny,
#     prob_bare,
#     prob_ignite
# )

# print("\nTask 2 initial forest:")
# print(forest)

# plot_forest(forest, 0)
# plt.show()

# iteration = 0

# while np.any(forest == FIRE):

#     forest = fire_step(
#         forest,
#         prob_spread=0.5
#     )

#     iteration += 1

# print(f"\nFire stopped after {iteration} iterations.")

# plot_forest(
#     forest,
#     iteration
# )

# plt.show()

# Task 2a

def run_fire_simulation(
    nx,
    ny,
    prob_spread,
    prob_bare,
    prob_ignite
):
    """
    Run one complete wildfire simulation.
    - number of iterations
    - fraction of initially burnable cells that burned
    """

    # Initialize forest
    forest = initialize_random_forest(
        nx,
        ny,
        prob_bare,
        prob_ignite
    )

    # Number of cells that could burn initially
    initial_burnable = np.sum(
        (forest == FOREST) | (forest == FIRE)
    )

    iteration = 0

    # Continue until no cells are burning
    while np.any(forest == FIRE):

        forest = fire_step(
            forest,
            prob_spread
        )

        iteration += 1

    # Forest cells remaining after the fire
    final_forest = np.sum(
        forest == FOREST
    )

    # Fraction of burnable forest that burned
    if initial_burnable > 0:
        burned_fraction = (
            initial_burnable - final_forest
        ) / initial_burnable
    else:
        burned_fraction = 0.0

    return iteration, burned_fraction

def run_until_fire_stops(forest, prob_spread):
    """
    Run the fire until no burning cells remain.
    """

    iteration = 0

    while np.any(forest == FIRE):

        forest = fire_step(
            forest,
            prob_spread
        )

        iteration += 1

    return forest, iteration

# iterations, burned_fraction = run_fire_simulation(
#     nx=20,
#     ny=20,
#     prob_spread=0.5,
#     prob_bare=0.2,
#     prob_ignite=0.05
# )

# print(
#     "\nTest simulation:"
# )

# print(
#     f"Iterations = {iterations}"
# )

# print(
#     f"Fraction burned = {burned_fraction:.3f}"
# )

# Task 2a Experiment 1
# Effect of P_spread

nx = 20
ny = 20

prob_bare = 0.0
prob_ignite = 0.02

# Test P_spread from 0 to 1
spread_values = np.linspace(0, 1, 11)

mean_burned = []

n_runs = 20

for prob_spread in spread_values:

    burned_results = []

    for run in range(n_runs):

        iterations, burned_fraction = run_fire_simulation(
            nx,
            ny,
            prob_spread,
            prob_bare,
            prob_ignite
        )

        burned_results.append(burned_fraction)

    mean_burned.append(
        np.mean(burned_results)
    )

    print(
        f"P_spread = {prob_spread:.1f}, "
        f"mean fraction burned = {np.mean(burned_results):.3f}"
    )


# Plot result
fig, ax = plt.subplots()

ax.plot(
    spread_values,
    mean_burned,
    marker="o"
)

ax.set_xlabel("Probability of Spread")
ax.set_ylabel("Mean Fraction of Forest Burned")

ax.set_title(
    "Effect of Fire Spread Probability"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()

# Task 2a Experiment 2
# Effect of P_bare

nx = 20
ny = 20

# Keep these fixed while changing P_bare
prob_spread = 0.5
prob_ignite = 0.02

# Test P_bare from 0 to 1
bare_values = np.linspace(0, 1, 11)

mean_burned = []

n_runs = 20

for prob_bare in bare_values:

    burned_results = []

    for run in range(n_runs):

        iterations, burned_fraction = run_fire_simulation(
            nx,
            ny,
            prob_spread,
            prob_bare,
            prob_ignite
        )

        burned_results.append(burned_fraction)

    mean_burned.append(
        np.mean(burned_results)
    )

    print(
        f"P_bare = {prob_bare:.1f}, "
        f"mean fraction burned = {np.mean(burned_results):.3f}"
    )


# Plot result
fig, ax = plt.subplots()

ax.plot(
    bare_values,
    mean_burned,
    marker="o"
)

ax.set_xlabel("Probability of Initially Bare Cells")
ax.set_ylabel("Mean Fraction of Forest Burned")

ax.set_title("Effect of Initial Bare Fraction on Wildfire Spread")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()

# Task 2b
# Controlled burn / firebreak demonstration

nx = 20
ny = 20

prob_spread = 1.0


# --------------------------
# Case 1: No controlled burn
# --------------------------

forest_no_burn = np.full(
    (ny, nx),
    FOREST,
    dtype=int
)

# Start fire on the left side
forest_no_burn[ny // 2, 1] = FIRE

final_no_burn, iterations_no_burn = run_until_fire_stops(
    forest_no_burn,
    prob_spread
)

print(
    f"No controlled burn: "
    f"{iterations_no_burn} iterations"
)

plot_forest(
    final_no_burn,
    iterations_no_burn
)


# --------------------------
# Case 2: Controlled burn
# --------------------------

forest_controlled = np.full(
    (ny, nx),
    FOREST,
    dtype=int
)

# Create a vertical strip of bare land
forest_controlled[:, nx // 2] = BARE

# Start fire at the same location
forest_controlled[ny // 2, 1] = FIRE

final_controlled, iterations_controlled = run_until_fire_stops(
    forest_controlled,
    prob_spread
)

print(
    f"Controlled burn: "
    f"{iterations_controlled} iterations"
)

plot_forest(
    final_controlled,
    iterations_controlled
)

plt.show()


# Task 3
# Disease / Zombie Model

DEAD = 0
IMMUNE = 1
HEALTHY = 2
SICK = 3

def initialize_disease(
    nx,
    ny,
    prob_vaccine,
    prob_sick
):
    """
    Initialize the disease model.

    0 = dead
    1 = immune
    2 = healthy
    3 = sick
    """

    # Start with everyone healthy
    population = np.full(
        (ny, nx),
        HEALTHY,
        dtype=int
    )

    # Vaccinated people begin immune
    is_immune = (
        np.random.rand(ny, nx)
        < prob_vaccine
    )

    population[is_immune] = IMMUNE

    # Randomly infect people who are still healthy
    is_sick = (
        (np.random.rand(ny, nx) < prob_sick)
        & (population == HEALTHY)
    )

    population[is_sick] = SICK

    return population

def disease_step(
    population,
    prob_spread,
    prob_fatal
):
    """
    Advance the disease model by one iteration.

    0 = dead
    1 = immune
    2 = healthy
    3 = sick
    """

    ny, nx = population.shape

    # Copy the current population so newly infected
    # people do not spread disease until the next iteration
    new_population = population.copy()

    for y in range(ny):
        for x in range(nx):

            # Only process people who are currently sick
            if population[y, x] == SICK:

                # Check the four orthogonal neighbors
                neighbors = [
                    (y + 1, x),     # up
                    (y - 1, x),     # down
                    (y, x - 1),     # left
                    (y, x + 1)      # right
                ]

                for neighbor_y, neighbor_x in neighbors:

                    # Make sure neighbor is inside the grid
                    if (
                        0 <= neighbor_y < ny
                        and 0 <= neighbor_x < nx
                    ):

                        # Disease can spread only to healthy people
                        if (
                            population[
                                neighbor_y,
                                neighbor_x
                            ] == HEALTHY
                        ):

                            if (
                                np.random.rand()
                                < prob_spread
                            ):
                                new_population[
                                    neighbor_y,
                                    neighbor_x
                                ] = SICK

                # After spreading disease,
                # determine whether the sick person survives
                if np.random.rand() < prob_fatal:
                    new_population[y, x] = DEAD
                else:
                    new_population[y, x] = IMMUNE

    return new_population

def plot_disease(population, iteration, title=None):
    """
    Plot the disease model.

    0 = dead
    1 = immune
    2 = healthy
    3 = sick
    """

    disease_cmap = ListedColormap(
        ["black", "lightblue", "darkgreen", "firebrick"]
    )

    fig, ax = plt.subplots(1, 1)

    ax.pcolor(
        population,
        cmap=disease_cmap,
        vmin=0,
        vmax=3,
        edgecolors="black",
        linewidths=0.5
    )

    ax.set_xticks(
        np.arange(population.shape[1]) + 0.5
    )
    ax.set_yticks(
        np.arange(population.shape[0]) + 0.5
    )

    ax.set_xticklabels(
        np.arange(population.shape[1])
    )
    ax.set_yticklabels(
        np.arange(population.shape[0])
    )

    if title is None:
        ax.set_title(f"Iteration {iteration}")
    else:
        ax.set_title(title)

    ax.set_aspect("equal")

def run_disease_simulation(
    nx,
    ny,
    prob_vaccine,
    prob_sick,
    prob_spread,
    prob_fatal
):
    """
    Run one complete disease simulation.

    Returns:
    - number of iterations
    - fraction of initially susceptible people infected
    - fraction of total population that died
    """

    population = initialize_disease(
        nx,
        ny,
        prob_vaccine,
        prob_sick
    )

    # People who could be infected at the beginning
    initial_susceptible = np.sum(
        (population == HEALTHY) | (population == SICK)
    )

    iteration = 0

    while np.any(population == SICK):

        population = disease_step(
            population,
            prob_spread,
            prob_fatal
        )

        iteration += 1

    # Healthy people remaining were never infected
    final_healthy = np.sum(
        population == HEALTHY
    )

    # Fraction of initially susceptible people that were infected
    if initial_susceptible > 0:
        infected_fraction = (
            initial_susceptible - final_healthy
        ) / initial_susceptible
    else:
        infected_fraction = 0.0

    # Fraction of entire population that died
    dead_fraction = (
        np.sum(population == DEAD)
        / (nx * ny)
    )

    return (
        iteration,
        infected_fraction,
        dead_fraction
    )

# nx = 20
# ny = 20

# prob_vaccine = 0.2
# prob_sick = 0.02

# population = initialize_disease(
#     nx,
#     ny,
#     prob_vaccine,
#     prob_sick
# )

# prob_spread = 0.5
# prob_fatal = 0.2

# iteration = 0

# while np.any(population == SICK):

#     population = disease_step(
#         population,
#         prob_spread,
#         prob_fatal
#     )

#     iteration += 1

# print(
#     f"\nDisease stopped after {iteration} iterations."
# )

# print(
#     f"Healthy = {np.sum(population == HEALTHY)}"
# )

# print(
#     f"Immune = {np.sum(population == IMMUNE)}"
# )

# print(
#     f"Dead = {np.sum(population == DEAD)}"
# )

# plot_disease(
#     population,
#     iteration,
#     title="Final Disease Population"
# )

# plt.show()

# Task 3 Experiment 1
# Effect of vaccination rate

nx = 20
ny = 20

prob_sick = 0.02
prob_spread = 0.5
prob_fatal = 0.2

vaccine_values = np.linspace(0, 1, 11)

mean_infected = []
mean_dead = []

n_runs = 20

for prob_vaccine in vaccine_values:

    infected_results = []
    dead_results = []

    for run in range(n_runs):

        (
            iterations,
            infected_fraction,
            dead_fraction
        ) = run_disease_simulation(
            nx,
            ny,
            prob_vaccine,
            prob_sick,
            prob_spread,
            prob_fatal
        )

        infected_results.append(
            infected_fraction
        )

        dead_results.append(
            dead_fraction
        )

    mean_infected.append(
        np.mean(infected_results)
    )

    mean_dead.append(
        np.mean(dead_results)
    )

    print(
        f"P_vaccine = {prob_vaccine:.1f}, "
        f"infected = {np.mean(infected_results):.3f}, "
        f"dead = {np.mean(dead_results):.3f}"
    )

fig, ax = plt.subplots()

ax.plot(
    vaccine_values,
    mean_infected,
    marker="o"
)

ax.set_xlabel("Vaccination Rate")
ax.set_ylabel("Mean Fraction Infected")

ax.set_title(
    "Effect of Vaccination Rate on Disease Spread"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()

# Task 3 Experiment 2
# Effect of mortality rate

nx = 20
ny = 20

# Keep these fixed
prob_vaccine = 0.2
prob_sick = 0.02
prob_spread = 0.5

# Test P_fatal from 0 to 1
fatal_values = np.linspace(0, 1, 11)

mean_infected = []
mean_dead = []

n_runs = 100

for prob_fatal in fatal_values:

    infected_results = []
    dead_results = []

    for run in range(n_runs):

        (
            iterations,
            infected_fraction,
            dead_fraction
        ) = run_disease_simulation(
            nx,
            ny,
            prob_vaccine,
            prob_sick,
            prob_spread,
            prob_fatal
        )

        infected_results.append(
            infected_fraction
        )

        dead_results.append(
            dead_fraction
        )

    mean_infected.append(
        np.mean(infected_results)
    )

    mean_dead.append(
        np.mean(dead_results)
    )

    print(
        f"P_fatal = {prob_fatal:.1f}, "
        f"infected = {np.mean(infected_results):.3f}, "
        f"dead = {np.mean(dead_results):.3f}"
    )

fig, ax = plt.subplots()

ax.plot(
    fatal_values,
    mean_infected,
    marker="o",
    label="Fraction Infected"
)

ax.plot(
    fatal_values,
    mean_dead,
    marker="s",
    label="Fraction Dead"
)

ax.set_xlabel("Mortality Probability (P_fatal)")
ax.set_ylabel("Mean Fraction of Population")

ax.set_title(
    "Effect of Mortality Rate on Disease Outcomes"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.legend()

plt.show()