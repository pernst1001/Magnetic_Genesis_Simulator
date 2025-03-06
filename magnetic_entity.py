import torch
from mpem import MPEMHandler
from magnetic_force_torque import MagneticForceTorque
from scipy.spatial.transform import Rotation as R


class MagneticEntity:
    def __init__(self, volume, remanence, direction, link_idx=None, rigid_solver=None):
        self.check_direction(direction)
        self.mu_0 = (4*torch.pi)*1e-7
        self.mangetic_dipole_magnitude = self.calculate_magnetic_dipole_magnitude(volume, remanence)
        direction = direction/torch.linalg.norm(direction)
        self.initial_magnetic_dipole = direction*self.mangetic_dipole_magnitude
        self.link_idx = torch.tensor([link_idx], dtype=torch.int32)
        self.rigid_solver = rigid_solver
        self.mpem_handler = MPEMHandler()
        self.magnetic_force_torque = MagneticForceTorque()
        self.initial_quad = self.rigid_solver.get_links_quat(self.link_idx)

    def calculate_magnetic_dipole_magnitude(self, volume, remanence):
        return (1/self.mu_0)*volume*remanence
    
    def check_direction(self, direction):
        if direction.shape != (3,) and direction.shape != (3,1):
            raise ValueError('Direction should have shape (3,) or (3,1)')
        if type(direction) != torch.Tensor:
            raise ValueError('Direction should be a torch tensor')
        
    def get_magnets_position(self):
        return self.rigid_solver.get_links_pos(self.link_idx)
    
    def get_current_quad(self):
        return self.rigid_solver.get_links_quat(self.link_idx)
    
    def get_current_dipole_moment(self):
        current_quad = self.get_current_quad()
        current_rotation = R.from_quat(current_quad) * R.from_quat(self.initial_quad).inv()
        current_dipole_moment = torch.tensor(current_rotation.apply(self.initial_magnetic_dipole))
        return current_dipole_moment
    
    def set_magnets_weight(self, weight):
        if type(weight) != torch.Tensor:
            raise ValueError('Weight should be a torch tensor')
        if weight.shape != (1,) and weight.shape != (1,1):
            raise ValueError('Weight should have shape (1,) or (1,1)')
        self.rigid_solver.set_links_inertial_mass(invweight=weight, links_idx=self.link_idx)

    def get_force_torque_from_currents(self, currents):
        position = self.get_magnets_position()
        field, gradient = self.mpem_handler.get_field_gradient5(position, currents)
        current_dipole_moment = self.get_current_dipole_moment()
        force = self.magnetic_force_torque.calculate_force(current_dipole_moment, gradient)
        torque = self.magnetic_force_torque.calculate_troque(current_dipole_moment, field)
        return force, torque
    
    def apply_force_torque_on_magnet(self, currents):
        force, torque = self.get_force_torque_from_currents(currents)
        self.rigid_solver.apply_links_external_force(force, self.link_idx)
        self.rigid_solver.apply_links_external_torque(torque, self.link_idx)
    
    def get_currents_from_field_gradient5(self, field, gradient):
        position = self.get_magnets_position()
        currents = self.mpem_handler.get_currents_from_field_grad5(position, field, gradient)
        return currents
        
    def get_currents_from_field_gradient3(self, field, gradient):
        position = self.get_magnets_position()
        dipole = self.get_current_dipole_moment()
        currents = self.mpem_handler.get_currents_from_field_grad3(position, field, dipole, gradient)
        return currents
    
    def get_field_gradient5(self, currents):
        position = self.get_magnets_position()
        field, gradient = self.mpem_handler.get_field_gradient5(position, currents)
        return field, gradient
        

        



        