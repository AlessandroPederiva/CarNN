import pygame
import time
import numpy as np
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm, Genotype
from car import Car
from course import Course

class Simulation:
    def __init__(self, screen_width=800, screen_height=600):
        """Initialize the simulation environment."""
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("2D Car Evolution Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Simulation parameters
        self.population_size = 40  # Increased from 20 for more diversity
        self.neural_network_topology = [5, 4, 3, 2]  # Input, hidden1, hidden2, output
        self.camera_offset = [0, 0]
        self.scale = 10.0  # Pixels per unit
        self.generation_finished = False
        
        # Create course
        self.create_course()
        
        # Initialize genetic algorithm
        self.initialize_genetic_algorithm()
    
    def create_course(self):
        """Create a simple course for the simulation."""
        # Define obstacles (walls)
        obstacles = [
            # Outer walls - rectangle with a gap at the right
            [(0, 0), (0, 60), (5, 60), (5, 0)],  # Left wall
            [(0, 0), (100, 0), (100, 5), (0, 5)],  # Top wall
            [(0, 55), (100, 55), (100, 60), (0, 60)],  # Bottom wall
            [(95, 0), (100, 0), (100, 60), (95, 60)],  # Right wall
            
            # Inner obstacle
            [(40, 20), (60, 20), (60, 40), (40, 40)]  # Box in the middle
        ]
        
        # Define checkpoints
        checkpoints = [
            ((20, 5), (20, 55)),   # First checkpoint
            ((50, 5), (50, 20)),   # Second checkpoint (above the box)
            ((50, 40), (50, 55)),  # Third checkpoint (below the box)
            ((80, 5), (80, 55))    # Fourth checkpoint
        ]
        
        # Define start position and rotation
        start_position = (10, 30)
        start_rotation = 0  # Facing right
        
        # Create course
        self.course = Course(obstacles, checkpoints, start_position, start_rotation)
    
    def initialize_genetic_algorithm(self):
        """Initialize the genetic algorithm and create initial population."""
        # Calculate total weights in neural network
        nn = NeuralNetwork(self.neural_network_topology)
        weight_count = nn.weight_count
        
        # Create genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(weight_count, self.population_size)
        
        # Start algorithm to initialize population
        self.population = self.genetic_algorithm.start()
        
        # Create cars with neural networks
        self.create_cars()
    
    def create_cars(self):
        """Create cars with neural networks based on current population."""
        self.course.cars = []
        
        for genotype in self.population:
            # Create neural network with topology
            nn = NeuralNetwork(self.neural_network_topology)
            
            # Set weights from genotype parameters
            nn.set_weights_flattened(genotype.parameters)
            
            # Create car with neural network
            car = Car(nn, self.course.start_position, self.course.start_rotation)
            self.course.add_car(car)
    
    def update(self, delta_time):
        """Update simulation state."""
        # Update all cars
        all_dead = True
        for i, car in enumerate(self.course.cars):
            car.update(self.course, delta_time)
            
            # Update corresponding genotype fitness
            self.population[i].fitness = car.fitness
            self.population[i].evaluated = not car.alive
            
            if car.alive:
                all_dead = False
        
        # If all cars are dead, create new generation
        if all_dead and not self.generation_finished:
            self.generation_finished = True
            self.create_new_generation()
        
        # Update camera to follow best car
        self.update_camera()
    
    def create_new_generation(self):
        """Create a new generation of cars using the genetic algorithm."""
        # Process current generation and create new one
        self.population = self.genetic_algorithm.evaluation_finished()
        
        # Create new cars with updated neural networks
        self.create_cars()
        
        self.generation_finished = False
    
    def update_camera(self):
        """Update camera position to follow the best car."""
        # Find the best car (highest fitness)
        best_car = None
        best_fitness = -1
        
        for car in self.course.cars:
            if car.alive and car.fitness > best_fitness:
                best_fitness = car.fitness
                best_car = car
        
        # If found, center camera on best car
        if best_car:
            target_x = best_car.position[0] - self.screen.get_width() / (2 * self.scale)
            target_y = best_car.position[1] - self.screen.get_height() / (2 * self.scale)
            
            # Smooth camera movement
            self.camera_offset[0] += (target_x - self.camera_offset[0]) * 0.1
            self.camera_offset[1] += (target_y - self.camera_offset[1]) * 0.1
    
    def draw(self):
        """Draw the current simulation state."""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw course and cars
        self.course.draw(self.screen, self.camera_offset, self.scale)
        
        # Draw UI information
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw UI elements with simulation information."""
        # Find best car
        best_car = None
        best_fitness = -1
        
        for car in self.course.cars:
            if car.fitness > best_fitness:
                best_fitness = car.fitness
                best_car = car
        
        # Draw generation count
        font = pygame.font.Font(None, 24)
        gen_text = font.render(f"Generation: {self.genetic_algorithm.generation_count}", True, (0, 0, 0))
        self.screen.blit(gen_text, (10, 10))
        
        # Draw best fitness
        if best_car:
            fitness_text = font.render(f"Best Fitness: {best_fitness:.2f}", True, (0, 0, 0))
            self.screen.blit(fitness_text, (10, 40))
            
            # Draw neural network outputs if car is alive
            if best_car.alive:
                outputs = best_car.neural_network.process_inputs(best_car.get_sensor_readings(self.course))
                output_text = font.render(f"Engine: {outputs[0]:.2f} Turn: {outputs[1]:.2f}", True, (0, 0, 0))
                self.screen.blit(output_text, (10, 70))
                
                # Draw alive cars count
                alive_count = sum(1 for car in self.course.cars if car.alive)
                alive_text = font.render(f"Alive: {alive_count}/{len(self.course.cars)}", True, (0, 0, 0))
                self.screen.blit(alive_text, (10, 100))
    
    def run(self):
        """Run the main simulation loop."""
        last_time = time.time()
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = min(current_time - last_time, 0.1)  # Cap delta time to avoid physics issues
            last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Update simulation
            self.update(delta_time)
            
            # Draw simulation
            self.draw()
            
            # Cap frame rate
            self.clock.tick(60)
        
        pygame.quit()
