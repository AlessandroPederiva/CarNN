import numpy as np
import random

class Genotype:
    def __init__(self, parameters=None):
        """
        Initialize a genotype with given parameters or empty if none provided.
        
        Args:
            parameters: Array of float values representing neural network weights
        """
        self.parameters = parameters if parameters is not None else []
        self.fitness = 0.0
        self.evaluated = False
    
    def __lt__(self, other):
        """Enable sorting by fitness (descending)."""
        return self.fitness > other.fitness


class GeneticAlgorithm:
    def __init__(self, parameter_count, population_size):
        """
        Initialize the genetic algorithm with specified parameters.
        
        Args:
            parameter_count: Number of parameters per genotype (total neural network weights)
            population_size: Size of the population
        """
        # Default parameters
        self.init_param_min = -1.0
        self.init_param_max = 1.0
        self.cross_swap_prob = 0.7  # Increased from 0.6
        self.mutation_prob = 0.5    # Increased from 0.3
        self.mutation_amount = 3.0  # Increased from 2.0
        self.mutation_perc = 1.0
        
        # Create initial population
        self.population_size = population_size
        self.current_population = [Genotype(np.zeros(parameter_count)) for _ in range(population_size)]
        self.generation_count = 1
        self.running = False
    
    def start(self):
        """Start the genetic algorithm by initializing the population."""
        self.running = True
        self.initialize_population()
        return self.current_population
    
    def initialize_population(self):
        """Initialize population with random parameter values."""
        for genotype in self.current_population:
            genotype.parameters = np.random.uniform(
                self.init_param_min, 
                self.init_param_max, 
                len(genotype.parameters)
            )
            genotype.evaluated = False
    
    def evaluation_finished(self):
        """Process the evaluated population and create the next generation."""
        # Sort population by fitness
        self.current_population.sort()
        
        # Select parents for next generation
        intermediate_population = self.selection()
        
        # Create new population through recombination
        new_population = self.recombination(intermediate_population)
        
        # Apply mutation to new population
        self.mutation(new_population)
        
        # Update current population and generation count
        self.current_population = new_population
        self.generation_count += 1
        
        return self.current_population
    
    def selection(self):
        """Select the best individuals for reproduction."""
        # Default implementation: select top 2 individuals
        return self.current_population[:2]
    
    def recombination(self, intermediate_population):
        """Create new population by recombining parents."""
        new_population = []
        
        # Create population_size new individuals
        for _ in range(self.population_size):
            # Select two parents from intermediate population
            parent1, parent2 = intermediate_population[0], intermediate_population[1]
            
            # Create child through crossover
            child_params = []
            for i in range(len(parent1.parameters)):
                # Randomly choose parameter from either parent
                if random.random() < self.cross_swap_prob:
                    child_params.append(parent1.parameters[i])
                else:
                    child_params.append(parent2.parameters[i])
            
            # Add new child to population
            new_population.append(Genotype(np.array(child_params)))
        
        return new_population
    
    def mutation(self, population):
        """Apply mutation to the population."""
        # Calculate how many individuals to mutate
        mutation_count = int(len(population) * self.mutation_perc)
        
        # Mutate random individuals (excluding the first one to preserve best solution)
        for i in range(1, min(mutation_count + 1, len(population))):
            genotype = population[i]
            
            # Mutate random parameters
            for j in range(len(genotype.parameters)):
                if random.random() < self.mutation_prob:
                    # Apply mutation by adding a random value
                    mutation = random.uniform(-self.mutation_amount, self.mutation_amount)
                    genotype.parameters[j] += mutation
            
            genotype.evaluated = False


# Test the genetic algorithm
if __name__ == "__main__":
    # Create a genetic algorithm with 10 parameters and population size of 20
    ga = GeneticAlgorithm(10, 20)
    
    # Start the algorithm
    population = ga.start()
    
    # Simulate evaluation
    for genotype in population:
        genotype.fitness = random.random()  # Random fitness for testing
        genotype.evaluated = True
    
    # Create next generation
    new_population = ga.evaluation_finished()
    
    print(f"Generation: {ga.generation_count}")
    print(f"Population size: {len(new_population)}")
    print(f"Best fitness: {new_population[0].fitness}")
