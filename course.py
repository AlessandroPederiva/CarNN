import pygame
import numpy as np

class Course:
    def __init__(self, obstacles, checkpoints, start_position, start_rotation):
        """
        Initialize a course with obstacles and checkpoints.
        
        Args:
            obstacles: List of obstacle polygons
            checkpoints: List of checkpoint lines in order
            start_position: Starting position for cars
            start_rotation: Starting rotation for cars
        """
        self.obstacles = obstacles
        self.checkpoints = checkpoints
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.cars = []
    
    def add_car(self, car):
        """Add a car to the course."""
        car.course = self  # Set the car's reference to this course
        self.cars.append(car)
    
    def is_point_in_obstacle(self, x, y):
        """Check if a point is inside any obstacle."""
        point = (x, y)
        
        for obstacle in self.obstacles:
            if self.point_in_polygon(point, obstacle):
                return True
        
        return False
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_completion_percentage(self, position):
        """Calculate how far a car has progressed through the course."""
        # Find the furthest checkpoint the car has passed
        passed_checkpoints = 0
        
        for i, checkpoint in enumerate(self.checkpoints):
            if self.has_passed_checkpoint(position, checkpoint):
                passed_checkpoints = i + 1
        
        return passed_checkpoints / max(1, len(self.checkpoints))
    
    def has_passed_checkpoint(self, position, checkpoint):
        """Check if a position has passed a checkpoint line."""
        # Simple implementation - just check if position is past the checkpoint's x-coordinate
        # This assumes checkpoints are vertical lines and the course runs left to right
        start, end = checkpoint
        return position[0] > start[0]
    
    def draw(self, screen, camera_offset, scale):
        """Draw the course on the screen."""
        # Draw obstacles
        for obstacle in self.obstacles:
            points = [(int((x - camera_offset[0]) * scale), int((y - camera_offset[1]) * scale)) 
                     for x, y in obstacle]
            pygame.draw.polygon(screen, (100, 100, 100), points)
        
        # Draw checkpoints
        for checkpoint in self.checkpoints:
            start, end = checkpoint
            start_screen = (int((start[0] - camera_offset[0]) * scale), 
                           int((start[1] - camera_offset[1]) * scale))
            end_screen = (int((end[0] - camera_offset[0]) * scale), 
                         int((end[1] - camera_offset[1]) * scale))
            pygame.draw.line(screen, (0, 255, 255), start_screen, end_screen, 2)
        
        # Draw cars
        for car in self.cars:
            car.draw(screen, camera_offset, scale)
