import pygame
import sys
import os
from simulation import Simulation

def main():
    """Main entry point for the car evolution simulation."""
    # Create and run simulation
    sim = Simulation(screen_width=800, screen_height=600)
    sim.run()

if __name__ == "__main__":
    main()
