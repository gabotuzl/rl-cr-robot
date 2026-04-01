from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RodParams:
    n_elements: int = 100                   # Number of elements
    base_length: float = 0.25               # Original length of rod (m)
    base_radius: float = 0.011 / 2          # Original radius of rod (m)
    density: float = 997.7                  # Density of rod (kg/m^3)
    youngs_modulus: float = 16.598637e6     # Elastic Modulus (Pa)
    shear_modulus: float = 7.216880e6       # Shear Modulus (Pa)
    direction: tuple = (1.0, 0.0, 0.0)      # Direction the rod extends
    normal: tuple = (0.0, 0.0, 1.0)         # Normal vector of rod
    start: tuple = (0.0, 0.0, 0.0)          # Starting position of first node in rod

    @property
    def dt_max(self) -> float:
        """Theoretical maximum stable timestep (CFL condition)."""
        return (self.base_length / self.n_elements) * np.sqrt(
            self.density / max(self.youngs_modulus, self.shear_modulus)
        )


@dataclass(frozen=True)
class TendonParams:
    num_tendons: int = 8
    max_tension: float = 20.0
    vertebra_height_long: float = 0.015
    num_vertebrae_long: int = 6
    first_vertebra_node_long: int = 2
    final_vertebra_node_long: int = 98
    vertebra_mass_long: float = 0.002
    vertebra_height_short: float = 0.008
    num_vertebrae_short: int = 6
    first_vertebra_node_short: int = 2
    final_vertebra_node_short: int = 30
    vertebra_mass_short: float = 0.001


@dataclass(frozen=True)
class SimParams:
    time_step: float = 0.8e-5 #RodParams.dt_max() or smaller
    final_time: float = 7.0
    damping_constant: float = 0.2
    rendering_fps: float = 10.0
    enable_gravity: bool = False
    gravity_axis: int = 2          # 0=X, 1=Y, 2=Z
    gravity_magnitude: float = -9.80665

    @property
    def gravity_vector(self) -> np.ndarray:
        vec = np.zeros(3)
        if self.enable_gravity:
            vec[self.gravity_axis] = self.gravity_magnitude
        return vec

    @property
    def step_skip(self) -> int:
        return int(1.0 / (self.rendering_fps * self.time_step))


# Instantiate ready-to-use singletons
ROD_PARAMS = RodParams()
TENDON_PARAMS = TendonParams()
SIM_PARAMS = SimParams()