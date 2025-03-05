import argparse
import numpy as np
import genesis as gs
import torch

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from scipy.spatial.transform import Rotation as R
from magnetic_entity import MagneticEntity

def main():
    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -9.81),
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.3),
            size=(0.2, 0.2, 0.2),
        ),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    camera = scene.add_camera(
        res    = (1280, 960),
        pos    = (0, -8, 6),
        lookat = (0, 0, 1.0),
        fov    = 20,
        GUI    = False
    )

    force_box = scene.add_entity(
        gs.morphs.Box(
            pos=[-0.5, 0, 0.05],  # Initially at cube's position
            size=(0.05, 0.05, 0.1),  # Small, thin quader for force
        ),
    )


    ########################## build ##########################
    # Apply friction to the cube's rigid geometry
    cube.set_friction(0.065)  # Corrected function call
    plane.set_friction(0.065)

    scene.build()

    for solver in scene.sim.solvers:
        if not isinstance(solver, RigidSolver):
            continue
        rigid_solver = solver


    cube_idx = [
        cube._get_ls_idx(),
    ]
    force_box_idx = [
        force_box._get_ls_idx(),
    ]
    weight_cube = 2
    # print(rigid_solver.mass_mat)
    # rigid_solver.set_links_mass_shift(mass=np.array([weight_cube]), links_idx=cube_idx)
    # rigid_solver.set_links_mass_shift(mass=np.array([0.5]), links_idx=force_box_idx)
    # rigid_solver.set_links_COM_shift(com=np.array([[0, 0, -0.1]]), links_idx=cube_idx)
    rigid_solver.set_links_inertial_mass(invweight=np.array([weight_cube]), links_idx=cube_idx)
    # camera.start_recording()
    initial_mass = rigid_solver.get_links_mass_shift(cube_idx)
    initial_quad = rigid_solver.get_links_quat(cube_idx)
    initial_rot = R.from_quat(initial_quad)
    magnet_dipole = np.array([0, 0, 3])
    for i in range(200):
        cube_pos = rigid_solver.get_links_pos(cube_idx)
        cube_quad = rigid_solver.get_links_quat(cube_idx)

        current_rot_quat = (R.from_quat(cube_quad) * R.from_quat(initial_quad).inv()).as_quat()
        force_box_pos = np.array([-0.5, cube_pos[0][0], 0.05])
        rigid_solver.set_links_pos([force_box_pos], force_box_idx)
        # rigid_solver.set_links_pos([[0, 0, 0.2]], force_box_idx)
        rigid_solver.set_links_quat(current_rot_quat, links_idx=force_box_idx)
        force = [
            [3, 0, 9.81 *0],
        ]
        print("force", force)
        print("cube_idx", cube_idx)
        rigid_solver.apply_links_external_force(
            force=force,
            links_idx=cube_idx,
        )

        torque = [
            [0, 0, 0.2],
        ]
        if i < 50:
            rigid_solver.apply_links_external_torque(
                torque=torque,
                links_idx=cube_idx,
            )
        camera.render()
        scene.step()

    # camera.stop_recording(save_to_filename="Videos/cube_bump.mp4", fps=60)


if __name__ == "__main__":
    main()