import random
import numpy as np

# Check if the solution is feasible or not
def isFeasible(solution):
    x1, x2, x3 = solution
    return (
            x1 + x2 + x3 <= 1500 and
            0.3 * x1 + 0.5 * x2 + 0.2 * x3 >= 500 and
            0.15 * x1 + 0.25 * x2 + 0.1 * x3 <= 600
    )

#Objective Function with penalty added if the solution violates at least one constraint
def objective(solution):
    x1, x2, x3 = solution
    if isFeasible(solution):
        return round(0.4 * x1 + 0.2 * x2 + 0.3 * x3, 2)
    else:
        penalty = (max(0, (x1 + x2 + x3 - 1500)) +
                   abs(min(0, 0.3 * x1 + 0.5 * x2 + 0.2 * x3 - 500)) +
                   max(0, 0.15 * x1 + 0.25 * x2 + 0.1 * x3 - 600))
        penaltyFactor = 1000  # Increased penalty factor
        return round((0.4 * x1 + 0.2 * x2 + 0.3 * x3) + penaltyFactor * penalty, 2)

#Return the fitness of population passed to it
def fitnessFunction(pop):
    fitness = np.zeros(len(pop))
    for i in range(len(pop)):
        fitness[i] = objective(pop[i])
    return fitness


# Repair function repairs the solution values x1, x2 and x3 if one of them exceed its boundary
def repair(solution):
    boundaries = [(0, 1000), (0, 640), (0, 420)]
    for i in range(3):
        lower, upper = boundaries[i]
        solution[i] = max(lower, min(solution[i], upper))

#Mutation Operation
def mutation(pop, solutions, mRate):
    # The three solutions p1, p2, p3
    p1, p2, p3 = [pop[i] for i in solutions]
    mutantVector = [round(p1[i] + mRate * (p2[i] - p3[i]), 2) for i in range(len(p1))]
    return mutantVector

#Crossover Operation
def crossover(targetVector, mutantVector, cRate):
    solutionLength = len(targetVector)
    trialVector = np.zeros(solutionLength, dtype=float)
    for i in range(len(targetVector)):
        if random.random() < cRate:
            trialVector[i] = mutantVector[i]
        else:
            trialVector[i] = targetVector[i]
    return list(trialVector)


def differential_evolution(pop, iterations, cRate, mRate):
    copyPopulation = pop
    bestFitness = []
    count = 1

    for i in range(iterations):
        '''
        This condition is the stopping criteria of the algorithm 
        if there is no big change over the 30 iterations then stop the algorithm
        '''
        if i >= 30:
            if abs(bestFitness[i - 30] - bestFitness[i - 1]) < bestFitness[i - 30] * 0.01:
                break
        
        #Rest of the algorithm below

        nextGeneration = np.zeros((len(pop), 3), dtype=float)
        fitness = fitnessFunction(copyPopulation)#Get the fitness of current generation

        # Do Elitism for the algorithm take the best two in the current population and keep them in the new generation
        sortedFitnessIndices = np.argsort(fitness)
        sortedPopulation = copyPopulation[sortedFitnessIndices]
        bestFitness.append(np.min(fitness))
        nextGeneration[0] = sortedPopulation[0]
        nextGeneration[1] = sortedPopulation[1]

        for j in range(2, len(copyPopulation)):  # Iterates on all solutions in the population
            # pop[j] is the target vector
            # Choose three random unique solutions distinct with the current solution
            #Get three random unique solutions not equal to the current solution
            #from the population to make mutation operation
            solutions = []
            while len(solutions) != 3:
                randomNum = random.randint(0, len(copyPopulation) - 1)
                if randomNum not in solutions:
                    if randomNum == j:
                        continue
                    else:
                        solutions.append(randomNum)

            # Get the mutant vector via mutation operation
            mutantVector = mutation(copyPopulation, solutions, mRate)

            # Pass the target and mutant vector to crossover operation to return the trial vector
            trialVector = crossover(copyPopulation[j], mutantVector, cRate)

            repair(trialVector)#If the new Solution violates the boundary repair it
            # If the trial vector is better than the target vector then take it else keep the target vector
            # Minimization Problem
            if objective(trialVector) < fitness[j]:
                nextGeneration[j] = trialVector
            else:
                nextGeneration[j] = copyPopulation[j]

        # Update the population with the new generation
        copyPopulation = nextGeneration

        count +=1#Count the number of iteration that the algorithm will take

    return copyPopulation, bestFitness , count

#Parameters
popSize = 50
numberOfIterations = 500
decisionVariables = 3  # X1, X2, X3
F = 0.7  # Mutation Factor
crossoverRate = 0.9 #Crossover Probability








# Randomly initialized x1, x2 and x3 values in the population
initialPopulation = [
    [round(random.uniform(0, 1000), 2),
     round(random.uniform(0, 640), 2),
     round(random.uniform(0, 420), 2)] for _ in range(popSize)
    ]

FinalGeneration, BestFitness, NumberOfIterations = differential_evolution(np.array(initialPopulation), numberOfIterations, crossoverRate, F)
FinalGenerationFitness = fitnessFunction(FinalGeneration)
bestSolutionValue = np.min(FinalGenerationFitness)
X1,X2,X3 = FinalGeneration[np.where(FinalGenerationFitness == bestSolutionValue)[0][0]]#Return the solution of the best solution fitness of final generation


np.set_printoptions(suppress=True, precision=2) #it helps print numpy arrays values normally

print(f"Algorithm takes {NumberOfIterations} iterations to find the solution")
print(f"Final Generation:\n{FinalGeneration}")
print(f"Final Generation Fitness:\n {FinalGenerationFitness}")
print(f"Best Fitness along the algorithm iterations:\n{np.array(BestFitness)}")


print(f"Best Solution is: X1 = {X1}, X2 = {X2} , X3={X3}")
print(f"Optimal Objective Function = {np.min(BestFitness)}")