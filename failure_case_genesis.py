import torch
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from magnetic_entity import MagneticEntity
from magnetic_force_torque import MagneticForceTorque
import time
def main():
        ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -1, 1),
        camera_lookat=(0.0, 0.0, 0.1),
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
            dt=1e-2,
            substeps=1,
        ),
        show_viewer=True,
    )


    ########################## entities ##########################
    cube_size = torch.tensor([4,4,4], dtype=torch.float64)*1e-3 # 4mm
    cube_volume = torch.prod(cube_size)
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, cube_size[2]/2),
            size=cube_size,
        ),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )   
    cube.set_friction(0.065)
    plane.set_friction(0.065)

    camera = scene.add_camera(
        res    = (1280, 960),
        pos    = (0, -1, 1),
        lookat = (0, 0, 0),
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
        direction=torch.tensor([-1, 1, -5], dtype=torch.float64),
        link_idx=cube._get_ls_idx(),
        rigid_solver=rigid_solver
    )
    magnetic_cube.set_magnets_weight(torch.tensor([0.48])*1e-3) # 0.48g
    rigid_solver.get_links_inertial_mass(cube._get_ls_idx())
    # camera.start_recording()
    for i in range(3000):
        gradient3 = torch.tensor([1e-8, 0, 0], dtype=torch.float64).reshape(1,3)
        field = torch.tensor([1e-3, 0, 0], dtype=torch.float64).reshape(1,3)
        current_dipole_moment = magnetic_cube.get_current_dipole_moment()
        torque = torch.cross(current_dipole_moment, field)
        rigid_solver.apply_links_external_torque(torque, magnetic_cube.link_idx)
        position = magnetic_cube.get_magnets_position()
        if i == 620:
            print('Bump! at', i)
            input('Press Enter to continue...')
            time.sleep(5)
        print('Position of the magnet:', position)
        scene.clear_debug_objects()
        scene.draw_debug_arrow(pos=magnetic_cube.get_magnets_position(),vec=gradient3*1e8, color=(0,1,0,0.5))
        
        camera.render()

        scene.step()

    # camera.stop_recording(save_to_filename="Videos/cube_bump.mp4", fps=60)


if __name__ == "__main__":
    main()