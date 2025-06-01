import pygame
import time
import numpy as np
from neural_network import TorchNeuralNetwork
from genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig, SelectionMethod
from car import Car
from course import Course

class Simulation:
    def __init__(self, screen_width=800, screen_height=600, custom_course=None):
        """Initialize the simulation environment."""
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("2D Car Evolution Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Simulation parameters
        self.population_size = 40  # Increased from 20 for more diversity
        self.neural_network_topology = [5, 12, 8, 1]  # Input, hidden1, hidden2, output (sterzata continua)
        self.camera_offset = [0, 0]
        self.scale = 10.0  # Pixels per unit
        self.generation_finished = False
        self.orientation_random_range = 30  # +/- gradi di variazione orientamento iniziale
        
        # Create course or use custom course
        if custom_course:
            self.course = custom_course
            print(f"Using custom course with {len(self.course.lines)} lines and {len(self.course.checkpoints)} checkpoints")
        else:
            self.create_default_course()
        
        # Initialize genetic algorithm
        self.initialize_genetic_algorithm()
        
        # Best car ever
        self.best_car_ever = None
        self.best_fitness_ever = float('-inf')
    
    def create_default_course(self):
        """Create a default S-shaped course with line obstacles."""
        wall_thickness = 5
        width = 60
        segment_length = 120
        
        # Primo rettilineo orizzontale (basso)
        base_y1 = 10
        base_y2 = base_y1 + width
        
        # Secondo rettilineo orizzontale (alto)
        top_y1 = base_y1 + width + 60
        top_y2 = top_y1 + width
        
        # Create lines for the S-shaped track
        lines = []
        
        # Bottom horizontal segment - top wall
        lines.append([(0, base_y1), (segment_length, base_y1)])
        # Bottom horizontal segment - bottom wall
        lines.append([(0, base_y2), (segment_length, base_y2)])
        
        # Right curve - left side
        lines.append([(segment_length, base_y1), (segment_length + width, top_y1)])
        # Right curve - right side
        lines.append([(segment_length, base_y2), (segment_length + width, top_y2)])
        
        # Top horizontal segment - top wall
        lines.append([(segment_length + width, top_y1), (2 * segment_length + width, top_y1)])
        # Top horizontal segment - bottom wall
        lines.append([(segment_length + width, top_y2), (2 * segment_length + width, top_y2)])
        
        # Left curve - left side
        lines.append([(2 * segment_length + width, top_y1), (2 * segment_length, base_y1)])
        # Left curve - right side
        lines.append([(2 * segment_length + width, top_y2), (2 * segment_length, base_y2)])
        
        # Bottom final segment - top wall
        lines.append([(2 * segment_length, base_y1), (3 * segment_length, base_y1)])
        # Bottom final segment - bottom wall
        lines.append([(2 * segment_length, base_y2), (3 * segment_length, base_y2)])
        
        # Checkpoint lungo la S
        checkpoints = []
        # Primo rettilineo
        for x in range(20, segment_length, 40):
            checkpoints.append(((x, base_y1), (x, base_y2)))
        
        # Curva destra
        for i in range(0, 5):
            t = i / 4
            cx = segment_length + t * width
            cy1 = base_y1 + t * (top_y1 - base_y1)
            cy2 = base_y2 + t * (top_y2 - base_y2)
            checkpoints.append(((cx, cy1), (cx, cy2)))
        
        # Secondo rettilineo
        for x in range(segment_length + width + 20, 2 * segment_length + width, 40):
            checkpoints.append(((x, top_y1), (x, top_y2)))
        
        # Curva sinistra
        for i in range(0, 5):
            t = i / 4
            cx = 2 * segment_length + width - t * width
            cy1 = top_y1 - t * (top_y1 - base_y1)
            cy2 = top_y2 - t * (top_y2 - base_y2)
            checkpoints.append(((cx, cy1), (cx, cy2)))
        
        # Terzo rettilineo
        for x in range(2 * segment_length + 20, 3 * segment_length, 40):
            checkpoints.append(((x, base_y1), (x, base_y2)))
        
        start_position = (15, (base_y1 + base_y2) // 2)
        start_rotation = 0  # Facing right
        
        self.course = Course(lines, checkpoints, start_position, start_rotation)
    
    def initialize_genetic_algorithm(self):
        """Initialize the genetic algorithm and create initial population."""
        nn = TorchNeuralNetwork(self.neural_network_topology, activation='tanh', input_norm=False, dropout_rate=0.1)
        weight_count = nn.weight_count
        # Use improved config object
        config = GeneticAlgorithmConfig(
            population_size=self.population_size,
            elite_size=2,
            selection_method=SelectionMethod.TOURNAMENT,
            tournament_size=3,
            crossover_rate=0.8,
            mutation_rate=0.1,
            mutation_strength=0.1
        )
        self.genetic_algorithm = GeneticAlgorithm(weight_count, config)
        self.population = self.genetic_algorithm.start()
        self.create_cars()
    
    def create_cars(self):
        """Create cars with neural networks based on current population."""
        self.course.cars = []
        
        for genotype in self.population:
            nn = TorchNeuralNetwork(self.neural_network_topology, activation='tanh', input_norm=False, dropout_rate=0.1)
            nn.set_random_weights(-1.0, 1.0)
            nn.set_weights_flattened(genotype.parameters)
            
            # Variazione casuale orientamento iniziale
            import random
            start_rot = self.course.start_rotation + random.uniform(-self.orientation_random_range, self.orientation_random_range)
            
            # Create car with neural network
            car = Car(nn, self.course.start_position, start_rot)
            self.course.add_car(car)
    
    def update(self, delta_time):
        """Update simulation state."""
        # Update all cars
        all_dead = True
        for i, car in enumerate(self.course.cars):
            # First check for line collisions (new method)
            if car.alive and self.course.car_collides_with_lines(car):
                car.alive = False
            
            # Then update the car normally
            car.update(self.course, delta_time)
            
            # Update corresponding genotype fitness
            self.population[i].fitness = car.fitness
            self.population[i].evaluated = not car.alive
            
            # --- AGGIUNTA: behavior descriptor per Hall of Fame/Novelty ---
            # Esempio: posizione finale e numero checkpoint passati
            checkpoints_passed = 0
            for j, checkpoint in enumerate(self.course.checkpoints):
                if self.course.has_passed_checkpoint(car.position, checkpoint):
                    checkpoints_passed = j + 1
            behavior_descriptor = np.array([
                car.position[0],
                car.position[1],
                car.rotation,
                checkpoints_passed
            ], dtype=np.float32)
            self.population[i].behavior_descriptor = behavior_descriptor
            
            if car.alive:
                all_dead = False
        
        # If all cars are dead, create new generation
        if all_dead and not self.generation_finished:
            self.generation_finished = True
            self.create_new_generation()
        
        # Update camera to follow best car
        self.update_camera()
        
        # Aggiorna best car di sempre
        for car in self.course.cars:
            if car.fitness > self.best_fitness_ever:
                self.best_fitness_ever = car.fitness
                # Salva una copia profonda del best car
                import copy
                self.best_car_ever = copy.deepcopy(car)
    
    def create_new_generation(self):
        """Create a new generation of cars using the genetic algorithm."""
        # Process current generation and create new one
        best_params = None
        if self.best_car_ever is not None:
            best_params = self.best_car_ever.neural_network.get_weights_flattened()
        self.population = self.genetic_algorithm.evaluation_finished(best_params=best_params)
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
        
        # Disegna il best car di sempre (in verde scuro, anche se morto)
        if self.best_car_ever is not None:
            self.best_car_ever.draw(self.screen, self.camera_offset, self.scale, force_green=True)
        
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
        
        # Miglior fitness di sempre
        best_ever_text = font.render(f"Best Ever Fitness: {self.best_fitness_ever:.2f}", True, (0, 100, 0))
        self.screen.blit(best_ever_text, (10, 40))
        
        # Draw best fitness
        if best_car:
            fitness_text = font.render(f"Best Fitness: {best_fitness:.2f}", True, (0, 0, 0))
            self.screen.blit(fitness_text, (10, 70))
            
            # Draw neural network outputs if car is alive
            if best_car.alive:
                outputs = best_car.neural_network.process_inputs(
                    best_car.sensors.get_readings(best_car.position, best_car.rotation, self.course)
                )
                output_text = font.render(f"Steering: {outputs[0]:.2f}", True, (0, 0, 0))
                self.screen.blit(output_text, (10, 100))
                
                # Draw alive cars count
                alive_count = sum(1 for car in self.course.cars if car.alive)
                alive_text = font.render(f"Alive: {alive_count}/{len(self.course.cars)}", True, (0, 0, 0))
                self.screen.blit(alive_text, (10, 130))
    
    def run(self):
        """Run the main simulation loop."""
        last_time = time.time()
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = min(current_time - last_time, 0.1) * 2  # Raddoppia il tempo che passa
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
