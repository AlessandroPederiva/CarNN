import pygame
import math
import numpy as np

class Car:
    def __init__(self, neural_network, start_position, start_rotation):
        """
        Initialize a car with a neural network.
        
        Args:
            neural_network: The neural network controlling this car
            start_position: Initial position (x, y)
            start_rotation: Initial rotation angle in degrees
        """
        self.neural_network = neural_network
        self.position = list(start_position)
        self.rotation = start_rotation
        self.velocity = 0.0
        self.sensor_count = 5
        self.sensor_range = 10.0  # Maximum sensor range
        self.sensor_spread = 90  # Degrees spread of sensors
        self.alive = True
        self.fitness = 0.0
        self.distance_traveled = 0.0
        self.last_position = list(start_position)
        self.course = None  # Reference to the course this car belongs to
        
        # Car dimensions and properties
        self.width = 2.0
        self.length = 4.0
        self.max_speed = 15.0  # Tripled from 5.0
        self.max_turn_rate = 9.0  # Tripled from 3.0
    
    def update(self, course, delta_time):
        """Update car state based on neural network and environment."""
        if not self.alive:
            return
        
        # Get sensor readings
        sensor_readings = self.get_sensor_readings(course)
        
        # Process sensor readings through neural network
        outputs = self.neural_network.process_inputs(sensor_readings)
        
        # Extract control values from neural network output
        engine_force = outputs[0] * 2.0 - 1.0  # Map from [0,1] to [-1,1]
        turning_force = outputs[1] * 2.0 - 1.0  # Map from [0,1] to [-1,1]
        
        # Apply physics
        self.apply_physics(engine_force, turning_force, delta_time)
        
        # Check for collision
        if self.check_collision(course):
            self.alive = False
        
        # Update fitness (based on distance traveled)
        current_distance = math.sqrt((self.position[0] - self.last_position[0])**2 + 
                                    (self.position[1] - self.last_position[1])**2)
        self.distance_traveled += current_distance
        self.last_position = list(self.position)
        self.fitness = self.calculate_fitness(course)
    
    def get_sensor_readings(self, course):
        """Get readings from car sensors."""
        readings = []
        
        # Calculate angle between each sensor
        angle_step = self.sensor_spread / (self.sensor_count - 1)
        
        # Calculate readings for each sensor
        for i in range(self.sensor_count):
            # Calculate sensor angle relative to car
            sensor_angle = self.rotation - self.sensor_spread/2 + i * angle_step
            
            # Cast ray and get distance to nearest obstacle
            distance = self.cast_ray(course, sensor_angle)
            
            # Normalize reading to [0,1] range
            normalized_distance = min(distance / self.sensor_range, 1.0)
            readings.append(normalized_distance)
        
        return readings
    
    def cast_ray(self, course, angle):
        """Cast a ray from car position and return distance to nearest obstacle."""
        ray_dir_x = math.cos(math.radians(angle))
        ray_dir_y = math.sin(math.radians(angle))
        
        # Check for intersection with course boundaries
        for step in range(int(self.sensor_range * 10)):
            dist = step / 10.0
            check_x = self.position[0] + ray_dir_x * dist
            check_y = self.position[1] + ray_dir_y * dist
            
            # Check if point is inside an obstacle
            if course.is_point_in_obstacle(check_x, check_y):
                return dist
        
        return self.sensor_range
    
    def apply_physics(self, engine_force, turning_force, delta_time):
        """Apply physics to update car position and rotation."""
        # Set a minimum base velocity if the car is alive and velocity is too low
        min_base_velocity = 2.0
        # Add a larger base speed to engine force to ensure movement
        engine_force += 0.6  # Increased from 0.2
        
        # Update velocity based on engine force (tripled acceleration)
        self.velocity += engine_force * delta_time * 6.0  # Tripled from 2.0
        self.velocity = max(-self.max_speed, min(self.velocity, self.max_speed))
        # Ensure minimum base velocity
        if self.alive and abs(self.velocity) < min_base_velocity:
            self.velocity = min_base_velocity * (1 if self.velocity >= 0 else -1)
        
        # Apply less drag/friction to maintain higher speeds
        self.velocity *= 0.97  # Reduced drag from 0.95
        
        # Update rotation based on turning force and velocity
        turn_amount = turning_force * self.velocity * delta_time * self.max_turn_rate
        self.rotation += turn_amount
        
        # Update position based on velocity and rotation
        self.position[0] += math.cos(math.radians(self.rotation)) * self.velocity * delta_time
        self.position[1] += math.sin(math.radians(self.rotation)) * self.velocity * delta_time
    
    def check_collision(self, course):
        """Check if car collides with any obstacle in the course."""
        # This is a simplified collision check
        # A more accurate implementation would check the car's corners
        return course.is_point_in_obstacle(self.position[0], self.position[1])
    
    def calculate_fitness(self, course):
        """Calculate fitness based on distance traveled and course completion."""
        # Basic fitness is distance traveled
        fitness = self.distance_traveled
        
        # Bonus for course completion percentage
        completion = course.get_completion_percentage(self.position)
        fitness *= (1.0 + completion)
        
        # Penalize cars that don't move
        if self.distance_traveled < 0.5:
            fitness *= 0.1
        
        return fitness
    
    def draw(self, screen, camera_offset, scale):
        """Draw the car on the screen."""
        if not self.alive:
            return
        
        # Calculate screen position
        screen_x = (self.position[0] - camera_offset[0]) * scale
        screen_y = (self.position[1] - camera_offset[1]) * scale
        
        # Create car polygon
        car_points = [
            (-self.length/2, -self.width/2),
            (self.length/2, -self.width/2),
            (self.length/2, self.width/2),
            (-self.length/2, self.width/2)
        ]
        
        # Rotate and translate points
        rotated_points = []
        for x, y in car_points:
            # Rotate
            rot_x = x * math.cos(math.radians(self.rotation)) - y * math.sin(math.radians(self.rotation))
            rot_y = x * math.sin(math.radians(self.rotation)) + y * math.cos(math.radians(self.rotation))
            
            # Translate and scale
            screen_point_x = int(screen_x + rot_x * scale)
            screen_point_y = int(screen_y + rot_y * scale)
            
            rotated_points.append((screen_point_x, screen_point_y))
        
        # Draw car body - determine if this is the best car
        is_best_car = False
        try:
            # Find the maximum fitness among all cars
            max_fitness = max([c.fitness for c in self.course.cars])
            is_best_car = (self.fitness == max_fitness)
        except:
            # If there's any error (like course not being accessible), default to not best
            pass
            
        pygame.draw.polygon(screen, (0, 255, 0) if is_best_car else (200, 200, 100), rotated_points)
        
        # Draw sensors
        self.draw_sensors(screen, camera_offset, scale, self.course)
    
    def draw_sensors(self, screen, camera_offset, scale, course):
        """Draw the car's sensors."""
        screen_x = (self.position[0] - camera_offset[0]) * scale
        screen_y = (self.position[1] - camera_offset[1]) * scale
        
        angle_step = self.sensor_spread / (self.sensor_count - 1)
        
        for i in range(self.sensor_count):
            sensor_angle = self.rotation - self.sensor_spread/2 + i * angle_step
            ray_dir_x = math.cos(math.radians(sensor_angle))
            ray_dir_y = math.sin(math.radians(sensor_angle))
            
            # Get sensor reading
            distance = self.cast_ray(course, sensor_angle)
            
            # Draw sensor line
            end_x = int(screen_x + ray_dir_x * distance * scale)
            end_y = int(screen_y + ray_dir_y * distance * scale)
            
            pygame.draw.line(screen, (255, 0, 0), (int(screen_x), int(screen_y)), (end_x, end_y), 1)
            
            # Draw sensor endpoint
            pygame.draw.circle(screen, (255, 255, 255), (end_x, end_y), 3)
