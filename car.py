import pygame
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

@dataclass
class CarConfig:
    """Configuration constants for car behavior"""
    SENSOR_COUNT: int = 5
    SENSOR_RANGE: float = 12.0
    SENSOR_SPREAD: float = 90.0
    MAX_SPEED: float = 14.5
    MAX_TURN_RATE: float = 28.0
    WIDTH: float = 1.8
    LENGTH: float = 3.5
    MIN_BASE_VELOCITY: float = 1.7
    ENGINE_BOOST: float = 0.4
    ACCELERATION_FACTOR: float = 6.5
    DRAG_FACTOR: float = 0.94
    HISTORY_SIZE: int = 100
    NOISE_GENERATION_THRESHOLD: int = 10
    NOISE_STD: float = 0.3

class CarPhysics:
    """Handles car physics calculations"""
    def __init__(self, config: CarConfig):
        self.config = config
    def update_velocity(self, current_velocity: float, engine_force: float, 
                       delta_time: float, is_alive: bool) -> float:
        total_force = engine_force + self.config.ENGINE_BOOST
        new_velocity = current_velocity + total_force * delta_time * self.config.ACCELERATION_FACTOR
        new_velocity = max(-self.config.MAX_SPEED, min(new_velocity, self.config.MAX_SPEED))
        if is_alive and abs(new_velocity) < self.config.MIN_BASE_VELOCITY:
            new_velocity = self.config.MIN_BASE_VELOCITY * (1 if new_velocity >= 0 else -1)
        return new_velocity * self.config.DRAG_FACTOR
    def update_rotation(self, current_rotation: float, turning_force: float, 
                       velocity: float, delta_time: float) -> float:
        turn_amount = turning_force * velocity * delta_time * self.config.MAX_TURN_RATE
        return current_rotation + turn_amount
    def update_position(self, current_position: List[float], rotation: float, 
                       velocity: float, delta_time: float) -> List[float]:
        rad_rotation = math.radians(rotation)
        new_x = current_position[0] + math.cos(rad_rotation) * velocity * delta_time
        new_y = current_position[1] + math.sin(rad_rotation) * velocity * delta_time
        return [new_x, new_y]

class CarSensors:
    """Handles car sensor logic"""
    def __init__(self, config: CarConfig):
        self.config = config
        self._cached_angles = self._calculate_sensor_angles()
    def _calculate_sensor_angles(self) -> List[float]:
        angle_step = self.config.SENSOR_SPREAD / (self.config.SENSOR_COUNT - 1)
        return [i * angle_step - self.config.SENSOR_SPREAD/2 
                for i in range(self.config.SENSOR_COUNT)]
    def get_readings(self, position: List[float], rotation: float, course) -> List[float]:
        readings = []
        for relative_angle in self._cached_angles:
            sensor_angle = rotation + relative_angle
            distance = self._cast_ray(position, sensor_angle, course)
            normalized_distance = min(distance / self.config.SENSOR_RANGE, 1.0)
            readings.append(normalized_distance)
        return readings
    def _cast_ray(self, position: List[float], angle: float, course) -> float:
        rad_angle = math.radians(angle)
        ray_dir_x = math.cos(rad_angle)
        ray_dir_y = math.sin(rad_angle)
        step_sizes = [1.0, 0.1]
        current_dist = 0.0
        for step_size in step_sizes:
            while current_dist < self.config.SENSOR_RANGE:
                check_x = position[0] + ray_dir_x * current_dist
                check_y = position[1] + ray_dir_y * current_dist
                if course.is_point_in_obstacle(check_x, check_y):
                    if step_size == 1.0:
                        current_dist = max(0, current_dist - step_size)
                        break
                    else:
                        return current_dist
                current_dist += step_size
        return self.config.SENSOR_RANGE

class FitnessCalculator:
    """Calculates car fitness with modular components"""
    def __init__(self):
        self.weights = {
            'distance': 1.0,
            'completion': 1.0,
            'checkpoints': 100.0,
            'net_distance': 2.0
        }
        self.penalties = {
            'circular_motion_threshold': 0.25,
            'start_proximity_threshold': 5.0,
            'high_turning_threshold': 0.7,
            'penalty_fitness': 0.01
        }
    def calculate(self, car, course) -> float:
        if self._has_circular_motion_penalty(car, course):
            return self.penalties['penalty_fitness']
        if self._has_start_proximity_penalty(car, course):
            return self.penalties['penalty_fitness']
        if self._has_excessive_turning_penalty(car):
            return self.penalties['penalty_fitness']
        base_fitness = car.distance_traveled * self.weights['distance']
        completion_bonus = course.get_completion_percentage(car.position) * self.weights['completion']
        checkpoint_bonus = self._calculate_checkpoint_bonus(car, course)
        net_distance_bonus = self._calculate_net_distance_bonus(car, course)
        return base_fitness * (1.0 + completion_bonus) + checkpoint_bonus + net_distance_bonus
    def _has_circular_motion_penalty(self, car, course) -> bool:
        if car.distance_traveled <= 2.0:
            return False
        net_distance = math.dist(car.position, course.start_position)
        ratio = net_distance / (car.distance_traveled + 1e-6)
        return ratio < self.penalties['circular_motion_threshold']
    def _has_start_proximity_penalty(self, car, course) -> bool:
        net_distance = math.dist(car.position, course.start_position)
        return net_distance < self.penalties['start_proximity_threshold']
    def _has_excessive_turning_penalty(self, car) -> bool:
        if not hasattr(car, 'turning_history') or car.distance_traveled <= 0.5:
            return False
        avg_turn = np.mean([abs(t) for t in car.turning_history])
        return avg_turn > self.penalties['high_turning_threshold']
    def _calculate_checkpoint_bonus(self, car, course) -> float:
        checkpoints_passed = 0
        for i, checkpoint in enumerate(course.checkpoints):
            if course.has_passed_checkpoint(car.position, checkpoint):
                checkpoints_passed = i + 1
        return checkpoints_passed * self.weights['checkpoints']
    def _calculate_net_distance_bonus(self, car, course) -> float:
        net_distance = math.dist(car.position, course.start_position)
        return net_distance * self.weights['net_distance']

class CarRenderer:
    """Handles car rendering"""
    def __init__(self, config: CarConfig):
        self.config = config
    def draw(self, screen, car, camera_offset: Tuple[float, float], 
             scale: float, force_green: bool = False):
        if not car.alive and not force_green:
            return
        self._draw_body(screen, car, camera_offset, scale, force_green)
        self._draw_sensors(screen, car, camera_offset, scale)
    def _draw_body(self, screen, car, camera_offset: Tuple[float, float], 
                   scale: float, force_green: bool):
        screen_pos = self._world_to_screen(car.position, camera_offset, scale)
        car_points = self._get_rotated_car_points(car.rotation, screen_pos, scale)
        color = self._get_car_color(car, force_green)
        pygame.draw.polygon(screen, color, car_points)
    def _draw_sensors(self, screen, car, camera_offset: Tuple[float, float], scale: float):
        screen_pos = self._world_to_screen(car.position, camera_offset, scale)
        for i in range(self.config.SENSOR_COUNT):
            angle_step = self.config.SENSOR_SPREAD / (self.config.SENSOR_COUNT - 1)
            sensor_angle = car.rotation - self.config.SENSOR_SPREAD/2 + i * angle_step
            distance = car.sensors._cast_ray(car.position, sensor_angle, car.course)
            end_pos = self._calculate_sensor_end(screen_pos, sensor_angle, distance, scale)
            pygame.draw.line(screen, (255, 0, 0), screen_pos, end_pos, 1)
            pygame.draw.circle(screen, (255, 255, 255), end_pos, 3)
    def _world_to_screen(self, world_pos: List[float], camera_offset: Tuple[float, float], 
                        scale: float) -> Tuple[int, int]:
        screen_x = int((world_pos[0] - camera_offset[0]) * scale)
        screen_y = int((world_pos[1] - camera_offset[1]) * scale)
        return (screen_x, screen_y)
    def _get_rotated_car_points(self, rotation: float, screen_pos: Tuple[int, int], 
                               scale: float) -> List[Tuple[int, int]]:
        half_length = self.config.LENGTH / 2
        half_width = self.config.WIDTH / 2
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        rad_rotation = math.radians(rotation)
        cos_rot = math.cos(rad_rotation)
        sin_rot = math.sin(rad_rotation)
        rotated_points = []
        for x, y in corners:
            rot_x = x * cos_rot - y * sin_rot
            rot_y = x * sin_rot + y * cos_rot
            screen_x = int(screen_pos[0] + rot_x * scale)
            screen_y = int(screen_pos[1] + rot_y * scale)
            rotated_points.append((screen_x, screen_y))
        return rotated_points
    def _get_car_color(self, car, force_green: bool) -> Tuple[int, int, int]:
        if force_green:
            return (0, 255, 0)
        try:
            max_fitness = max(c.fitness for c in car.course.cars)
            is_best = (car.fitness == max_fitness)
            return (0, 255, 0) if is_best else (200, 200, 100)
        except:
            return (200, 200, 100)
    def _calculate_sensor_end(self, start_pos: Tuple[int, int], angle: float, 
                             distance: float, scale: float) -> Tuple[int, int]:
        rad_angle = math.radians(angle)
        end_x = int(start_pos[0] + math.cos(rad_angle) * distance * scale)
        end_y = int(start_pos[1] + math.sin(rad_angle) * distance * scale)
        return (end_x, end_y)

class Car:
    """Improved Car class with better separation of concerns"""
    def __init__(self, neural_network, start_position: Tuple[float, float], 
                 start_rotation: float, config: Optional[CarConfig] = None):
        self.config = config or CarConfig()
        self.neural_network = neural_network
        self.position = list(start_position)
        self.rotation = start_rotation
        self.velocity = 0.0
        self.alive = True
        self.fitness = 0.0
        self.distance_traveled = 0.0
        self.last_position = list(start_position)
        self.course = None
        self.turning_history = deque(maxlen=self.config.HISTORY_SIZE)
        self.position_history = deque(maxlen=self.config.HISTORY_SIZE)
        self.physics = CarPhysics(self.config)
        self.sensors = CarSensors(self.config)
        self.fitness_calculator = FitnessCalculator()
        self.renderer = CarRenderer(self.config)
        self.time_alive = 0.0  # timer per autodistruzione
        self.max_circle_time = 4.0  # secondi massimi consentiti in giro in tondo
    def update(self, course, delta_time: float):
        if not self.alive:
            return
        try:
            self.time_alive += delta_time
            sensor_readings = self.sensors.get_readings(self.position, self.rotation, course)
            outputs = self._get_neural_network_outputs(sensor_readings, course)
            engine_force, turning_force = self._process_network_outputs(outputs)
            self.turning_history.append(turning_force)
            self._apply_physics(engine_force, turning_force, delta_time)
            if self._check_collision(course):
                self.alive = False
                return
            # Autodistruzione se penalità giro in tondo dopo max_circle_time
            if self.time_alive > self.max_circle_time and self.fitness_calculator._has_circular_motion_penalty(self, course):
                self.alive = False
                return
            self._update_tracking(course)
        except Exception as e:
            print(f"Error updating car: {e}")
            self.alive = False
    def _get_neural_network_outputs(self, sensor_readings: List[float], course) -> List[float]:
        noise_std = 0.0
        if (hasattr(course, 'genetic_algorithm') and 
            getattr(course.genetic_algorithm, 'generation_count', 1) <= self.config.NOISE_GENERATION_THRESHOLD):
            noise_std = self.config.NOISE_STD
        if hasattr(self.neural_network, 'process_inputs'):
            return self.neural_network.process_inputs(sensor_readings, noise_std=noise_std)
        else:
            return self.neural_network.process_inputs(sensor_readings)
    def _process_network_outputs(self, outputs: List[float]) -> Tuple[float, float]:
        # Output continuo: outputs[0] in [-1, 1] (sterzata)
        engine_force = 1.0
        turning_force = float(np.tanh(outputs[0]))
        return engine_force, turning_force
    def _apply_physics(self, engine_force: float, turning_force: float, delta_time: float):
        self.velocity = self.physics.update_velocity(
            self.velocity, engine_force, delta_time, self.alive)
        self.rotation = self.physics.update_rotation(
            self.rotation, turning_force, self.velocity, delta_time)
        self.position = self.physics.update_position(
            self.position, self.rotation, self.velocity, delta_time)
    def _check_collision(self, course) -> bool:
        return course.is_point_in_obstacle(self.position[0], self.position[1])
    def _update_tracking(self, course):
        current_distance = math.dist(self.position, self.last_position)
        self.distance_traveled += current_distance
        self.last_position = list(self.position)
        self.position_history.append(tuple(self.position))
        self.fitness = self.fitness_calculator.calculate(self, course)
    def draw(self, screen, camera_offset: Tuple[float, float], scale: float, 
             force_green: bool = False):
        self.renderer.draw(screen, self, camera_offset, scale, force_green)
