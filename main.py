import pygame
import sys
import os
from simulation import Simulation
from track_editor import TrackEditor

def main():
    """Main entry point with track editor integration."""
    # First run the track editor
    print("Starting Track Editor...")
    editor = TrackEditor(screen_width=800, screen_height=600)
    course = editor.run()
    
    # If a valid course was created, use it in the simulation
    if course:
        print("Track created successfully! Starting simulation...")
        # Create and run simulation with custom course
        sim = Simulation(screen_width=800, screen_height=600, custom_course=course)
        sim.run()
    else:
        print("No valid track was created. Starting simulation with default track...")
        # Create and run simulation with default course
        sim = Simulation(screen_width=800, screen_height=600)
        sim.run()

if __name__ == "__main__":
    main()
