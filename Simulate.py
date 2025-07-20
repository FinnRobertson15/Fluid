import pygame
import time
import numpy as np
from Fluid import Fluid
import pygame
import numpy as np
import time
import torch

def run_interactive_fluid_simulation(fluid, window_size=512, target_fps=60):
    pygame.init()
    
    # Create display
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption('Interactive Fluid Simulation - Click and drag!')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    running = True
    last_time = time.time()
    mouse_pressed = False
    last_mouse_pos = None
    
    # For adding forces
    force_strength = 5000.0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset simulation
                    fluid.u.zero_()
                    fluid.v.zero_()
                    fluid.p.zero_()
                    fluid.m.fill_(1.0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pressed = True
                    last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_pressed = False
                    last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if mouse_pressed and last_mouse_pos:
                    current_pos = pygame.mouse.get_pos()
                    
                    # Convert screen coordinates to fluid grid coordinates
                    grid_x = int((current_pos[0] / window_size) * fluid.n)
                    grid_y = int((current_pos[1] / window_size) * fluid.n)
                    last_grid_x = int((last_mouse_pos[0] / window_size) * fluid.n)
                    last_grid_y = int((last_mouse_pos[1] / window_size) * fluid.n)
                    
                    # Calculate mouse velocity
                    dx = grid_x - last_grid_x
                    dy = grid_y - last_grid_y
                    
                    # Add force to fluid
                    if 0 <= grid_x < fluid.n and 0 <= grid_y < fluid.n:
                        radius = 3  # Radius of influence
                        for i in range(max(0, grid_x - radius), min(fluid.n, grid_x + radius + 1)):
                            for j in range(max(0, grid_y - radius), min(fluid.n, grid_y + radius + 1)):
                                distance = ((i - grid_x)**2 + (j - grid_y)**2)**0.5
                                if distance <= radius:
                                    strength = (1 - distance / radius) * force_strength
                                    fluid.u[i, j] += dx * strength
                                    fluid.v[i, j] += dy * strength
                    
                    last_mouse_pos = current_pos
        
        # Calculate dt
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Limit dt to prevent instability
        dt = min(dt, 1.0/target_fps)
        
        # Fluid simulation step
        fluid.integrate(dt)
        fluid.solveIncompressibility(dt)
        fluid.advectVel(dt)
        fluid.advectSmoke(dt)
        

        # Render
        rgb = fluid.render()
        rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)
        
        # Convert to pygame surface
        rgb_transposed = np.transpose(rgb_np, (1, 0, 2))
        surface = pygame.surfarray.make_surface(rgb_transposed)
        
        # Scale to window size
        scaled_surface = pygame.transform.scale(surface, (window_size, window_size))
        
        # Blit to screen
        screen.blit(scaled_surface, (0, 0))
        
        # Add instructions
        fps_text = font.render(f'FPS: {clock.get_fps():.1f}', True, (255, 255, 255))
        instruction_text = font.render('Click and drag to add forces, R to reset', True, (255, 255, 255))
        
        screen.blit(fps_text, (10, 10))
        screen.blit(instruction_text, (10, window_size - 30))
        
        pygame.display.flip()
        
        # Maintain target framerate
        clock.tick(target_fps)
    
    pygame.quit()

import argparse
# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fluid simulation.")
    parser.add_argument("--size", type=int, default=128, help="Grid size (e.g., 128)")
    args = parser.parse_args()
    fluid = Fluid(args.size, numIters=25, boundaries=False)
    run_interactive_fluid_simulation(fluid, window_size=512, target_fps=60)