import torch
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from magnetic_entity import MagneticEntity
from magnetic_force_torque import MagneticForceTorque
import numpy as np
import pygame
import threading

class JoystickController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Joystick Controller")
        self.clock = pygame.time.Clock()
        self.center = (200, 200)
        self.joystick_pos = self.center
        self.direction = np.array([0, 0])
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.joystick_pos = event.pos
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        self.joystick_pos = event.pos
                        self.direction = np.array([self.joystick_pos[0] - self.center[0], self.joystick_pos[1] - self.center[1]])
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.joystick_pos = self.center
                    self.direction = np.array([0, 0])

            self.screen.fill((255, 255, 255))
            pygame.draw.circle(self.screen, (0, 0, 0), self.center, 50, 2)
            pygame.draw.circle(self.screen, (0, 0, 255), self.joystick_pos, 20)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def get_direction(self):
        if np.linalg.norm(self.direction) == 0:
            return np.array([0, 0])
        return self.direction / np.linalg.norm(self.direction)

def run_gui(controller):
    controller.run()


def main():

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -0.1, 0.1),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        vis_options=vis_options,
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -9.81),
            dt=4e-4,
            substeps=1,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -1.0, 0.0),
            upper_bound=(0.5, 1.0, 1),
        ),
        show_viewer=True,
    )



    ########################## entities ##########################
    cube_size = torch.tensor([4,4,4], dtype=torch.float64)*1e-3 # 4mm
    cube_size = torch.tensor([1,1,1], dtype=torch.float64)*1e-3 # 1mm
    cube_volume = torch.prod(cube_size)
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, cube_size[2]/2),
            size=cube_size,
            collision = True,
        ),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(

        ),
    )   

    cube.set_friction(0.065)
    plane.set_friction(0.065)

    camera = scene.add_camera(
        res    = (1280, 960),
        pos    = (0, -0.05, 0.05),
        lookat = (0, 0.01, 0.01),
        fov    = 40,
        GUI    = False
    )

    ########################## sover ##########################
    for solver in scene.sim.solvers:
        if not isinstance(solver, RigidSolver):
            continue
        rigid_solver = solver


    ########################## build ##########################
    scene.build()
    magnetic_cube = MagneticEntity(
        volume=cube_volume,
        remanence=1.32 ,
        direction=torch.tensor([0, 0, 1], dtype=torch.float64),
        link_idx=cube._get_ls_idx(),
        rigid_solver=rigid_solver
    )
    magnetic_cube.set_magnets_weight(torch.tensor([0.48])*1e-3) # 0.48g
    rigid_solver.get_links_inertial_mass(cube._get_ls_idx())
    controller = JoystickController()
    gui_thread = threading.Thread(target=run_gui, args=(controller,))
    gui_thread.start()
    field = torch.tensor([0, 0, 0], dtype=torch.float64).reshape(1,3)
    for i in range(5000):
        # Use the direction to control the magnet
        direction = controller.get_direction()
        gradient3 = torch.tensor([direction[0], direction[1], 0], dtype=torch.float64).reshape(1,3)*1e-3    
        currents = magnetic_cube.get_currents_from_field_gradient3(field, gradient3)
        magnetic_cube.apply_force_torque_on_magnet(currents)
        camera.render()

        scene.step()

if __name__ == "__main__":
    main()