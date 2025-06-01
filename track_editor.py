import pygame
import numpy as np
import json
import os
import math
from course import Course

class TrackEditor:
    """Interactive track editor for creating custom tracks for the car evolution simulation."""
    
    def __init__(self, screen_width=800, screen_height=600):
        """Initialize the track editor."""
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Track Editor - Car Evolution Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Editor state
        self.mode = "wall"  # Modes: "wall", "checkpoint", "start"
        self.drawing = False
        self.current_line_start = None
        self.lines = []  # List of wall lines (each line is a pair of points)
        self.checkpoints = []
        self.start_position = None
        self.start_rotation = 0
        
        # Camera and zoom
        self.camera_offset = [0, 0]
        self.scale = 10.0  # Pixels per unit
        self.dragging = False
        self.drag_start = (0, 0)
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.button_height = 30
        self.button_margin = 10
        self.buttons = [
            {"text": "Wall Mode", "action": self.set_wall_mode, "rect": None},
            {"text": "Checkpoint Mode", "action": self.set_checkpoint_mode, "rect": None},
            {"text": "Start Position", "action": self.set_start_mode, "rect": None},
            {"text": "Eraser", "action": self.set_eraser_mode, "rect": None},  # Nuovo pulsante gomma
            {"text": "Clear All", "action": self.clear_all, "rect": None},
            {"text": "Save Track", "action": self.save_track, "rect": None},
            {"text": "Load Track", "action": self.load_track, "rect": None},
            {"text": "Start Simulation", "action": self.start_simulation, "rect": None}
        ]
        self.update_button_positions()
        
        # Messages
        self.message = ""
        self.message_time = 0
        
        # Track data
        self.track_name = "custom_track"
        self.track_path = os.path.join(os.getcwd(), "tracks")
        if not os.path.exists(self.track_path):
            os.makedirs(self.track_path)
    
    def update_button_positions(self):
        """Update button positions based on screen size."""
        button_width = 150
        x = self.screen.get_width() - button_width - self.button_margin
        y = self.button_margin
        
        for button in self.buttons:
            button["rect"] = pygame.Rect(x, y, button_width, self.button_height)
            y += self.button_height + self.button_margin
    
    def set_wall_mode(self):
        """Set editor to wall drawing mode."""
        self.mode = "wall"
        self.show_message("Wall Mode: Click to start a line, click again to end it")
    
    def set_checkpoint_mode(self):
        """Set editor to checkpoint drawing mode."""
        self.mode = "checkpoint"
        self.show_message("Checkpoint Mode: Click and drag to create checkpoints")
    
    def set_start_mode(self):
        """Set editor to start position mode."""
        self.mode = "start"
        self.show_message("Start Position Mode: Click to set start position, scroll to rotate")
    
    def set_eraser_mode(self):
        """Set editor to eraser mode."""
        self.mode = "eraser"
        self.show_message("Eraser Mode: Click near a line to delete it")
    
    def clear_all(self):
        """Clear all track elements."""
        self.lines = []
        self.checkpoints = []
        self.start_position = None
        self.start_rotation = 0
        self.show_message("Track cleared")
    
    def save_track(self):
        """Save the current track to a file."""
        if not self.lines:
            self.show_message("Error: No walls to save")
            return
        
        if not self.checkpoints:
            self.show_message("Error: No checkpoints to save")
            return
        
        if not self.start_position:
            self.show_message("Error: No start position set")
            return
        
        # Convert track data to serializable format
        track_data = {
            "lines": self.lines,
            "checkpoints": self.checkpoints,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation
        }
        
        # Save to file
        filename = f"{self.track_name}.json"
        filepath = os.path.join(self.track_path, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(track_data, f)
            self.show_message(f"Track saved as {filename}")
        except Exception as e:
            self.show_message(f"Error saving track: {str(e)}")
    
    def load_track(self):
        """Load a track from a file."""
        # Get list of track files
        track_files = [f for f in os.listdir(self.track_path) 
                      if f.endswith('.json')]
        
        if not track_files:
            self.show_message("No track files found")
            return
        
        # For simplicity, just load the first track
        # In a full implementation, you'd want a file selection dialog
        filename = track_files[0]
        filepath = os.path.join(self.track_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                track_data = json.load(f)
            
            # Handle both old and new format
            if "lines" in track_data:
                self.lines = track_data["lines"]
            elif "walls" in track_data:
                # Convert old walls format to lines if needed
                self.lines = []
                for wall in track_data["walls"]:
                    if len(wall) >= 2:
                        for i in range(len(wall) - 1):
                            self.lines.append([wall[i], wall[i+1]])
            
            self.checkpoints = track_data["checkpoints"]
            self.start_position = track_data["start_position"]
            self.start_rotation = track_data["start_rotation"]
            
            self.track_name = os.path.splitext(filename)[0]
            self.show_message(f"Loaded track: {self.track_name}")
        except Exception as e:
            self.show_message(f"Error loading track: {str(e)}")
    
    def start_simulation(self):
        """Start the simulation with the current track."""
        if not self.lines:
            self.show_message("Error: No walls to simulate")
            return
        
        if not self.checkpoints:
            self.show_message("Error: No checkpoints to simulate")
            return
        
        if not self.start_position:
            self.show_message("Error: No start position set")
            return
        
        # Save the track first
        self.save_track()
        
        # Exit the editor - the main script will handle starting the simulation
        self.running = False
    
    def show_message(self, text):
        """Show a temporary message on screen."""
        self.message = text
        self.message_time = pygame.time.get_ticks()
    
    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates."""
        world_x = screen_pos[0] / self.scale + self.camera_offset[0]
        world_y = screen_pos[1] / self.scale + self.camera_offset[1]
        return (world_x, world_y)
    
    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates."""
        screen_x = int((world_pos[0] - self.camera_offset[0]) * self.scale)
        screen_y = int((world_pos[1] - self.camera_offset[1]) * self.scale)
        return (screen_x, screen_y)
    
    def _distance_point_to_segment(self, point, seg_a, seg_b):
        """Calcola la distanza minima tra un punto e un segmento."""
        px, py = point
        x1, y1 = seg_a
        x2, y2 = seg_b
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_1:
                    self.set_wall_mode()
                elif event.key == pygame.K_2:
                    self.set_checkpoint_mode()
                elif event.key == pygame.K_3:
                    self.set_start_mode()
                # --- AGGIUNTA: movimento camera con frecce ---
                elif event.key == pygame.K_LEFT:
                    self.camera_offset[0] -= 10 / self.scale  # muovi a sinistra
                elif event.key == pygame.K_RIGHT:
                    self.camera_offset[0] += 10 / self.scale  # muovi a destra
                elif event.key == pygame.K_UP:
                    self.camera_offset[1] -= 10 / self.scale  # muovi in su
                elif event.key == pygame.K_DOWN:
                    self.camera_offset[1] += 10 / self.scale  # muovi in giù
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicked on UI buttons
                for button in self.buttons:
                    if button["rect"].collidepoint(event.pos):
                        button["action"]()
                        break
                else:  # No button was clicked
                    if event.button == 1:  # Left mouse button
                        if self.mode == "wall":
                            world_pos = self.screen_to_world(event.pos)
                            if self.current_line_start is None:
                                # Start a new line
                                self.current_line_start = world_pos
                                self.drawing = True
                                self.show_message("Click to end the line")
                            else:
                                # End the current line
                                self.lines.append([self.current_line_start, world_pos])
                                self.current_line_start = None
                                self.drawing = False
                                self.show_message(f"Line added, total lines: {len(self.lines)}")
                        elif self.mode == "checkpoint":
                            self.drawing = True
                            world_pos = self.screen_to_world(event.pos)
                            self.current_line_start = world_pos
                        elif self.mode == "start":
                            world_pos = self.screen_to_world(event.pos)
                            self.start_position = world_pos
                            self.show_message(f"Start position set at {world_pos}")
                        elif self.mode == "eraser":
                            # Gomma: cancella la linea più vicina al click se abbastanza vicina
                            mouse_world = self.screen_to_world(event.pos)
                            min_dist = float('inf')
                            min_idx = -1
                            threshold = 10 / self.scale  # 10 pixel in coordinate mondo
                            for idx, line in enumerate(self.lines):
                                dist = self._distance_point_to_segment(mouse_world, line[0], line[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = idx
                            if min_dist < threshold and min_idx != -1:
                                del self.lines[min_idx]
                                self.show_message("Line deleted")
                            else:
                                self.show_message("No line close enough to delete")
                    
                    elif event.button == 3:  # Right mouse button
                        # Cancel current line if drawing
                        if self.mode == "wall" and self.current_line_start is not None:
                            self.current_line_start = None
                            self.drawing = False
                            self.show_message("Line drawing canceled")
                        else:
                            # Start dragging the view
                            self.dragging = True
                            self.drag_start = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    if self.drawing and self.mode == "checkpoint":
                        world_pos = self.screen_to_world(event.pos)
                        self.checkpoints.append((self.current_line_start, world_pos))
                        self.show_message(f"Checkpoint added, total: {len(self.checkpoints)}")
                        self.drawing = False
                
                elif event.button == 3:  # Right mouse button
                    self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing and self.mode == "checkpoint":
                    # Update endpoint of checkpoint preview
                    pass
                
                elif self.dragging:
                    # Move camera
                    dx = (event.pos[0] - self.drag_start[0]) / self.scale
                    dy = (event.pos[1] - self.drag_start[1]) / self.scale
                    self.camera_offset[0] -= dx
                    self.camera_offset[1] -= dy
                    self.drag_start = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out
                zoom_factor = 1.1
                if event.y > 0:
                    # Zoom in
                    self.scale *= zoom_factor
                elif event.y < 0:
                    # Zoom out
                    self.scale /= zoom_factor
                
                # If in start mode, adjust rotation
                if self.mode == "start" and self.start_position:
                    self.start_rotation = (self.start_rotation + event.y * 5) % 360
                    self.show_message(f"Start rotation: {self.start_rotation}°")
    
    def draw(self):
        """Draw the editor state."""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        self.draw_grid()
        
        # Draw wall lines
        for line in self.lines:
            start_screen = self.world_to_screen(line[0])
            end_screen = self.world_to_screen(line[1])
            pygame.draw.line(self.screen, (100, 100, 100), start_screen, end_screen, 3)
        
        # Draw current line being created
        if self.mode == "wall" and self.current_line_start is not None:
            start_screen = self.world_to_screen(self.current_line_start)
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, (200, 0, 0), start_screen, mouse_pos, 3)
        
        # Draw checkpoints
        for checkpoint in self.checkpoints:
            start_screen = self.world_to_screen(checkpoint[0])
            end_screen = self.world_to_screen(checkpoint[1])
            pygame.draw.line(self.screen, (0, 255, 255), start_screen, end_screen, 2)
        
        # Draw checkpoint being created
        if self.mode == "checkpoint" and self.drawing and self.current_line_start is not None:
            start_screen = self.world_to_screen(self.current_line_start)
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, (0, 200, 200), start_screen, mouse_pos, 2)
        
        # Draw start position
        if self.start_position:
            pos_screen = self.world_to_screen(self.start_position)
            
            # Draw a car-like shape
            car_length = 4 * self.scale / 10
            car_width = 2 * self.scale / 10
            
            # Calculate corners based on rotation
            rad_rotation = math.radians(self.start_rotation)
            cos_rot = math.cos(rad_rotation)
            sin_rot = math.sin(rad_rotation)
            
            corners = [
                (-car_length/2, -car_width/2),
                (car_length/2, -car_width/2),
                (car_length/2, car_width/2),
                (-car_length/2, car_width/2)
            ]
            
            rotated_corners = []
            for x, y in corners:
                rot_x = x * cos_rot - y * sin_rot
                rot_y = x * sin_rot + y * cos_rot
                rotated_corners.append((pos_screen[0] + rot_x, pos_screen[1] + rot_y))
            
            pygame.draw.polygon(self.screen, (0, 200, 0), rotated_corners)
            
            # Draw direction indicator
            end_x = pos_screen[0] + car_length * cos_rot
            end_y = pos_screen[1] + car_length * sin_rot
            pygame.draw.line(self.screen, (255, 0, 0), pos_screen, (end_x, end_y), 2)
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw a grid to help with track design."""
        grid_size = 10  # World units
        grid_color = (230, 230, 230)
        
        # Calculate visible grid range
        left = int(self.camera_offset[0] / grid_size) * grid_size
        top = int(self.camera_offset[1] / grid_size) * grid_size
        right = left + int(self.screen.get_width() / self.scale) + grid_size
        bottom = top + int(self.screen.get_height() / self.scale) + grid_size
        
        # Draw vertical lines
        for x in range(left, right + 1, grid_size):
            start = self.world_to_screen((x, top))
            end = self.world_to_screen((x, bottom))
            pygame.draw.line(self.screen, grid_color, start, end)
        
        # Draw horizontal lines
        for y in range(top, bottom + 1, grid_size):
            start = self.world_to_screen((left, y))
            end = self.world_to_screen((right, y))
            pygame.draw.line(self.screen, grid_color, start, end)
    
    def draw_ui(self):
        """Draw UI elements."""
        # Draw buttons
        for button in self.buttons:
            # Highlight active mode button
            if (button["text"] == "Wall Mode" and self.mode == "wall") or \
               (button["text"] == "Checkpoint Mode" and self.mode == "checkpoint") or \
               (button["text"] == "Start Position" and self.mode == "start"):
                color = (200, 200, 255)
            else:
                color = (200, 200, 200)
            
            pygame.draw.rect(self.screen, color, button["rect"])
            pygame.draw.rect(self.screen, (0, 0, 0), button["rect"], 1)
            
            text = self.font.render(button["text"], True, (0, 0, 0))
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)
        
        # Draw current mode
        mode_text = self.font.render(f"Mode: {self.mode.capitalize()}", True, (0, 0, 0))
        self.screen.blit(mode_text, (10, 10))
        
        # Draw stats
        stats = [
            f"Lines: {len(self.lines)}",
            f"Checkpoints: {len(self.checkpoints)}",
            f"Start: {'Set' if self.start_position else 'Not Set'}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (0, 0, 0))
            self.screen.blit(text, (10, 40 + i * 25))
        
        # Draw temporary message
        if self.message and pygame.time.get_ticks() - self.message_time < 3000:
            text = self.font.render(self.message, True, (200, 0, 0))
            self.screen.blit(text, (10, self.screen.get_height() - 30))
        
        # Draw help text
        help_text = [
            "Controls:",
            "Left Click: Draw/Place",
            "Right Click + Drag: Move View",
            "Right Click: Cancel Line",
            "Mouse Wheel: Zoom / Rotate Start",
            "1-3: Switch Modes",
            "ESC: Exit"
        ]
        
        for i, line in enumerate(help_text):
            text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (10, self.screen.get_height() - 150 + i * 20))
    
    def create_course(self):
        """Create a Course object from the current track."""
        if not self.lines or not self.checkpoints or not self.start_position:
            return None
        
        return Course(self.lines, self.checkpoints, self.start_position, self.start_rotation)
    
    def run(self):
        """Run the track editor."""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        
        # Return the created course if complete
        return self.create_course()


if __name__ == "__main__":
    # Run the track editor standalone
    editor = TrackEditor()
    course = editor.run()
    
    if course:
        print("Track created successfully!")
        print(f"Lines: {len(course.lines)}")
        print(f"Checkpoints: {len(course.checkpoints)}")
        print(f"Start position: {course.start_position}")
    else:
        print("No valid track was created.")
