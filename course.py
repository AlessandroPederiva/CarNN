import pygame
import numpy as np
import math

class Course:
    def __init__(self, lines, checkpoints, start_position, start_rotation):
        """
        Initialize a course with line obstacles and checkpoints.
        
        Args:
            lines: List of line segments (each line is [start_point, end_point])
            checkpoints: List of checkpoint lines in order
            start_position: Starting position for cars
            start_rotation: Starting rotation for cars
        """
        self.lines = lines
        self.checkpoints = checkpoints
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.cars = []
    
    def add_car(self, car):
        """Add a car to the course."""
        car.course = self  # Set the car's reference to this course
        self.cars.append(car)
    
    def is_point_in_obstacle(self, x, y):
        """
        Check if a point collides with any line obstacle.
        For backward compatibility with the original code.
        """
        # For a point, we check if it's very close to any line
        point = (x, y)
        threshold = 0.5  # Distance threshold to consider collision with a line
        
        for line in self.lines:
            if self.point_to_line_distance(point, line[0], line[1]) < threshold:
                return True
        
        return False
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the shortest distance from a point to a line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the squared length of the line segment
        line_length_squared = (x2 - x1)**2 + (y2 - y1)**2
        
        # If the line segment is actually a point, return the distance to that point
        if line_length_squared == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate the projection of the point onto the line
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_squared))
        
        # Calculate the closest point on the line segment
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Return the distance to the closest point
        return math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
    
    def car_collides_with_lines(self, car):
        """
        Check if a car collides with any line obstacle.
        This is more accurate than just checking the center point.
        """
        # Get car corners in world coordinates
        car_corners = self.get_car_corners(car)
        
        # Check if any car edge intersects with any line
        for i in range(4):
            car_edge_start = car_corners[i]
            car_edge_end = car_corners[(i + 1) % 4]
            
            for line in self.lines:
                if self.line_segments_intersect(car_edge_start, car_edge_end, line[0], line[1]):
                    return True
        
        return False
    
    def get_car_corners(self, car):
        """Get the four corners of the car in world coordinates."""
        # Car dimensions
        half_length = car.config.LENGTH / 2
        half_width = car.config.WIDTH / 2
        
        # Car position and rotation
        pos_x, pos_y = car.position
        rad_rotation = math.radians(car.rotation)
        cos_rot = math.cos(rad_rotation)
        sin_rot = math.sin(rad_rotation)
        
        # Calculate corners (relative to car center)
        corners_rel = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Transform corners to world coordinates
        corners_world = []
        for x_rel, y_rel in corners_rel:
            x_world = pos_x + x_rel * cos_rot - y_rel * sin_rot
            y_world = pos_y + x_rel * sin_rot + y_rel * cos_rot
            corners_world.append((x_world, y_world))
        
        return corners_world
    
    def line_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect."""
        # Convert points to coordinates
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate the direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        # Calculate the determinant
        det = dx1 * dy2 - dy1 * dx2
        
        # If determinant is zero, lines are parallel
        if abs(det) < 1e-8:
            return False
        
        # Calculate parameters for the intersection point
        t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
        t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
        
        # Check if the intersection point is within both line segments
        return 0 <= t1 <= 1 and 0 <= t2 <= 1
    
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
        # Draw wall lines
        for line in self.lines:
            start_screen = (int((line[0][0] - camera_offset[0]) * scale), 
                           int((line[0][1] - camera_offset[1]) * scale))
            end_screen = (int((line[1][0] - camera_offset[0]) * scale), 
                         int((line[1][1] - camera_offset[1]) * scale))
            pygame.draw.line(screen, (100, 100, 100), start_screen, end_screen, 3)
        
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
