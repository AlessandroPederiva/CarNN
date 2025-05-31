"""
Improved Genetic Algorithm with multiple selection methods and elitism.
"""

import numpy as np
import random
import logging
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class SelectionMethod(Enum):
    """Available selection methods."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITISM = "elitism"

@dataclass
class Genotype:
    """
    Enhanced genotype class with additional metrics and metadata.
    """
    parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness: float = 0.0
    evaluated: bool = False
    age: int = 0
    parent_ids: Tuple[int, int] = field(default_factory=lambda: (-1, -1))
    generation: int = 0
    id: int = field(default_factory=lambda: random.randint(0, 1000000))
    
    def __lt__(self, other: 'Genotype') -> bool:
        """Enable sorting by fitness (descending)."""
        return self.fitness > other.fitness
    
    def clone(self) -> 'Genotype':
        """Create a deep copy of this genotype."""
        return Genotype(
            parameters=self.parameters.copy(),
            fitness=self.fitness,
            evaluated=self.evaluated,
            age=self.age,
            parent_ids=self.parent_ids,
            generation=self.generation,
            id=random.randint(0, 1000000)  # New ID for clone
        )

class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select_parents(self, population: List[Genotype], num_parents: int) -> List[Genotype]:
        """Select parents from population."""
        pass

class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select_parents(self, population: List[Genotype], num_parents: int) -> List[Genotype]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament = random.sample(population, 
                                     min(self.tournament_size, len(population)))
            
            # Select best from tournament
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents

class RouletteSelection(SelectionStrategy):
    """Roulette wheel selection strategy."""
    
    def select_parents(self, population: List[Genotype], num_parents: int) -> List[Genotype]:
        """Select parents using roulette wheel selection."""
        # Ensure all fitness values are positive
        min_fitness = min(ind.fitness for ind in population)
        adjusted_fitness = [ind.fitness - min_fitness + 1e-6 for ind in population]
        
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            # If all fitness values are zero, select randomly
            return random.choices(population, k=num_parents)
        
        # Calculate selection probabilities
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        # Select parents based on probabilities
        parents = np.random.choice(population, size=num_parents, 
                                 p=probabilities, replace=True).tolist()
        
        return parents

class RankSelection(SelectionStrategy):
    """Rank-based selection strategy."""
    
    def select_parents(self, population: List[Genotype], num_parents: int) -> List[Genotype]:
        """Select parents using rank-based selection."""
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Assign ranks (higher rank = better fitness)
        ranks = list(range(len(sorted_pop), 0, -1))
        total_rank = sum(ranks)
        
        # Calculate selection probabilities based on ranks
        probabilities = [rank / total_rank for rank in ranks]
        
        # Select parents
        parents = np.random.choice(sorted_pop, size=num_parents,
                                 p=probabilities, replace=True).tolist()
        
        return parents

class ElitismSelection(SelectionStrategy):
    """Elitism selection - select top individuals."""
    
    def select_parents(self, population: List[Genotype], num_parents: int) -> List[Genotype]:
        """Select top individuals as parents."""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:num_parents]

@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    elite_size: int = 2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    adaptive_mutation: bool = True
    diversity_threshold: float = 0.01
    max_generations: int = 1000
    convergence_threshold: float = 1e-6
    convergence_generations: int = 50

class GeneticAlgorithm:
    """
    Enhanced genetic algorithm with multiple selection strategies and advanced features.
    """
    
    def __init__(self, parameter_count: int, config: Optional[GeneticAlgorithmConfig] = None):
        """
        Initialize genetic algorithm.
        
        Args:
            parameter_count: Number of parameters per genotype
            config: Configuration object
        """
        self.parameter_count = parameter_count
        self.config = config or GeneticAlgorithmConfig()
        
        # Population management
        self.current_population: List[Genotype] = []
        self.generation_count = 0
        self.running = False
        
        # Statistics tracking
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.best_individual: Optional[Genotype] = None
        self.stagnation_counter = 0
        
        # Selection strategy
        self.selection_strategy = self._create_selection_strategy()
        
        logger.info(f"Initialized GA with {parameter_count} parameters, "
                   f"population size: {self.config.population_size}")
    
    def _create_selection_strategy(self) -> SelectionStrategy:
        """Create selection strategy based on configuration."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return TournamentSelection(self.config.tournament_size)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return RouletteSelection()
        elif self.config.selection_method == SelectionMethod.RANK:
            return RankSelection()
        elif self.config.selection_method == SelectionMethod.ELITISM:
            return ElitismSelection()
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def start(self) -> List[Genotype]:
        """Initialize and start the genetic algorithm."""
        self.running = True
        self.generation_count = 1
        self.current_population = self._initialize_population()
        
        logger.info(f"Started GA with generation 1, population size: {len(self.current_population)}")
        return self.current_population
    
    def _initialize_population(self) -> List[Genotype]:
        """Create initial population with random parameters."""
        population = []
        
        for i in range(self.config.population_size):
            genotype = Genotype(
                parameters=np.random.uniform(-1.0, 1.0, self.parameter_count),
                generation=self.generation_count,
                id=i
            )
            population.append(genotype)
        
        return population
    
    def evaluation_finished(self, best_params=None) -> List[Genotype]:
        """Process evaluated population and create next generation."""
        if not self.current_population:
            raise RuntimeError("No population to evaluate")
        self._update_statistics()
        if self._check_convergence():
            logger.info(f"Convergence reached at generation {self.generation_count}")
            return self.current_population
        new_population = self._create_next_generation(best_params=best_params)
        self.generation_count += 1
        self.current_population = new_population
        logger.debug(f"Created generation {self.generation_count}")
        return self.current_population
    
    def _update_statistics(self):
        """Update fitness and diversity statistics."""
        # Sort population by fitness
        self.current_population.sort(reverse=True)
        
        # Update best individual
        current_best = self.current_population[0]
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.clone()
            self.stagnation_counter = 0
            logger.info(f"New best fitness: {current_best.fitness:.6f} at generation {self.generation_count}")
        else:
            self.stagnation_counter += 1
        
        # Track fitness history
        avg_fitness = np.mean([ind.fitness for ind in self.current_population])
        self.fitness_history.append(avg_fitness)
        
        # Calculate diversity
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        logger.debug(f"Generation {self.generation_count}: "
                    f"Best={current_best.fitness:.4f}, "
                    f"Avg={avg_fitness:.4f}, "
                    f"Diversity={diversity:.4f}")
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.current_population) < 2:
            return 0.0
        
        # Calculate pairwise distances between parameter vectors
        distances = []
        for i in range(len(self.current_population)):
            for j in range(i + 1, len(self.current_population)):
                dist = np.linalg.norm(
                    self.current_population[i].parameters - 
                    self.current_population[j].parameters
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged."""
        if len(self.fitness_history) < self.config.convergence_generations:
            return False
        
        # Check fitness improvement
        recent_fitness = self.fitness_history[-self.config.convergence_generations:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        if fitness_improvement < self.config.convergence_threshold:
            return True
        
        # Check diversity
        if self.diversity_history[-1] < self.config.diversity_threshold:
            return True
        
        return False
    
    def _create_next_generation(self, best_params=None) -> List[Genotype]:
        """Create next generation using selection, crossover, and mutation. Usa best_params come genitore privilegiato in metà dei crossover."""
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = min(self.config.elite_size, len(self.current_population))
        elites = sorted(self.current_population, reverse=True)[:elite_count]
        
        for elite in elites:
            elite_copy = elite.clone()
            elite_copy.generation = self.generation_count + 1
            elite_copy.age += 1
            new_population.append(elite_copy)
        
        # Generate offspring for remaining population
        offspring_needed = self.config.population_size - elite_count
        
        use_best_as_parent = (best_params is not None)
        crossover_count = 0
        while len(new_population) < self.config.population_size:
            if use_best_as_parent and crossover_count % 2 == 0:
                # Forza best ever come uno dei genitori
                parent1 = Genotype(parameters=best_params.copy())
                parent2 = random.choice(self.current_population)
                parents = [parent1, parent2]
            else:
                parents = self.selection_strategy.select_parents(self.current_population, 2)
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parents[0], parents[1])
            else:
                child1, child2 = parents[0].clone(), parents[1].clone()
            
            # Mutation
            self._mutate(child1)
            self._mutate(child2)
            
            # Add to new population
            for child in [child1, child2]:
                if len(new_population) < self.config.population_size:
                    child.generation = self.generation_count + 1
                    child.evaluated = False
                    new_population.append(child)
            crossover_count += 1
        
        return new_population
    
    def _crossover(self, parent1: Genotype, parent2: Genotype) -> Tuple[Genotype, Genotype]:
        """Perform crossover between two parents."""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Uniform crossover
        mask = np.random.random(self.parameter_count) < 0.5
        
        child1.parameters[mask] = parent2.parameters[mask]
        child2.parameters[~mask] = parent1.parameters[~mask]
        
        # Set parent information
        child1.parent_ids = (parent1.id, parent2.id)
        child2.parent_ids = (parent1.id, parent2.id)
        
        return child1, child2
    
    def _mutate(self, individual: Genotype):
        """Apply mutation to an individual."""
        # Adaptive mutation rate based on diversity
        mutation_rate = self.config.mutation_rate
        if self.config.adaptive_mutation and self.diversity_history:
            diversity = self.diversity_history[-1]
            if diversity < self.config.diversity_threshold:
                mutation_rate *= 2.0  # Increase mutation when diversity is low
        
        # Apply mutation
        mutation_mask = np.random.random(self.parameter_count) < mutation_rate
        
        if np.any(mutation_mask):
            # Gaussian mutation
            mutations = np.random.normal(0, self.config.mutation_strength, 
                                       self.parameter_count)
            individual.parameters[mutation_mask] += mutations[mutation_mask]
            
            # Clamp to reasonable bounds
            individual.parameters = np.clip(individual.parameters, -10.0, 10.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        return {
            'generation': self.generation_count,
            'population_size': len(self.current_population),
            'best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'avg_fitness': np.mean([ind.fitness for ind in self.current_population]),
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'stagnation_counter': self.stagnation_counter,
            'convergence_check': self._check_convergence()
        }
    
    def save_best_individual(self, filepath: str):
        """Save the best individual to file."""
        if self.best_individual is None:
            raise RuntimeError("No best individual to save")
        
        np.savez(filepath,
                parameters=self.best_individual.parameters,
                fitness=self.best_individual.fitness,
                generation=self.best_individual.generation)
        
        logger.info(f"Saved best individual to {filepath}")
    
    def load_individual(self, filepath: str) -> Genotype:
        """Load an individual from file."""
        data = np.load(filepath)
        
        individual = Genotype(
            parameters=data['parameters'],
            fitness=float(data['fitness']),
            generation=int(data['generation']),
            evaluated=True
        )
        
        logger.info(f"Loaded individual from {filepath}")
        return individual


if __name__ == "__main__":
    # Test the improved genetic algorithm
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = GeneticAlgorithmConfig(
        population_size=20,
        elite_size=2,
        selection_method=SelectionMethod.TOURNAMENT,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # Create GA
    ga = GeneticAlgorithm(parameter_count=10, config=config)
    
    # Start algorithm
    population = ga.start()
    
    # Simulate several generations
    for generation in range(5):
        # Simulate evaluation (random fitness)
        for individual in population:
            individual.fitness = np.random.random() + np.sum(individual.parameters**2) * 0.001
            individual.evaluated = True
        
        # Create next generation
        population = ga.evaluation_finished()
        
        # Print statistics
        stats = ga.get_statistics()
        print(f"Generation {stats['generation']}: "
              f"Best={stats['best_fitness']:.4f}, "
              f"Avg={stats['avg_fitness']:.4f}, "
              f"Diversity={stats['diversity']:.4f}")
    
    print("Genetic algorithm test completed successfully!")
