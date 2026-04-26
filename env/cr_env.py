from sim.sim_params import ROD_PARAMS, TENDON_PARAMS, SIM_PARAMS
from config import CONFIG
from env.reward import GOAL_DIST_THRESHOLD

from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box
import numpy as np 
import random
from datetime import datetime
from common.utils import write_errorfile, get_state
from env.reward import compute_reward
from sim.rod_simulator import RodSimulator, MyCallBack

from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)
from elastica._calculus import _isnan_check
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import GravityForces
from sim.OctoTendonForces import OctoTendonForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.timestepper import extend_stepper_interface
from collections import defaultdict

class cr_env(Env):
    def __init__(self):
        super().__init__()

        # Define: rod_length, max_tension, num_tendons, tendon_orientations
        self.max_tension = TENDON_PARAMS.max_tension 
        self.num_tendons = TENDON_PARAMS.num_tendons 
        self.L = ROD_PARAMS.base_length

        # Setting attributes for later use
        self.X_variation = CONFIG.env.x_variation
        self.Y_variation = CONFIG.env.y_variation
        self.Z_variation = CONFIG.env.z_variation
        self.time_step = ROD_PARAMS.dt
        self.steps_per_learn_update = CONFIG.env.num_timesteps_per_step
        self.max_steps = int((CONFIG.env.episode_time/self.time_step)/self.steps_per_learn_update)  #max steps per episode.
        print(self.max_steps)

        
        # Action space
        # - Tension allowed in the tendons (-1 to 1 to keep it symmetric on 0.0, will be scaled later in step() (0 to self.max_tension)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_tendons,), dtype=np.float64)

        # Observation space
        # - The XYZ position history of the 5 previous tip positions (15 values)
        # - The previous 5 actions (8 components each), intended for smoother control (40 values)
        # - The speeds of 10 nodes on the rod, not counting the tip (10 values)
        # - The positions of the same 10 nodes on the rod, not counting the tip (30 values)
        # - The XYZ position of the target (3 values)
        # - The XYZ self.delta of the tip position of the target and the current state (3 values)
        # - The velocity vector of the tip (3 values)
        # - The speed of the rod tip (1 value)
        n_positions = 3*CONFIG.env.action_history_len
        n_actions = 8*CONFIG.env.action_history_len
        node_speeds_checked = CONFIG.env.nodes_checked 
        node_pos_checked = 3*CONFIG.env.nodes_checked

        obs_size = n_positions + n_actions + node_speeds_checked + node_pos_checked + 3 + 3 + 3 + 1

        # VecNormalize handles obs bounds, which is why they are not constrained
        low  = np.full(obs_size, -np.inf, dtype=np.float64)
        high = np.full(obs_size,  np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float64)

        # Random target 
        self.target_position = np.array([np.random.uniform(self.L - self.X_variation, self.L), 
                                         np.random.uniform(-self.Y_variation, self.Y_variation),
                                         np.random.uniform(-self.Z_variation, self.Z_variation)]) 
        
        # Setting up counters, summers, and state variables
        self.action_history = np.zeros((CONFIG.env.action_history_len,8)) # Previous actions, all with 8 components each
        self.state_history = np.zeros((CONFIG.env.state_history_len,3)) # Previous states, all with 3 components each
        self.NN_tendon_tensions = np.zeros(8) # The tendon tensions (action)
        self.state = np.zeros(3) # Initial position
        self.timestamp_start = datetime.now()
        self.StatefulStepper = PositionVerlet() # Symplectic ime integration scheme

        self.goal_reached_flag = np.array([False])
        self.step_count = 0
        self.score = 0.0
        self.total_step_count = 0
        self.nan_counter = 0
        self.out_of_bounds_counter = 0
        self.max_steps_counter = 0

        # Setting nodes whose speeds and pos will be checked as an obs
        self.n_elements = ROD_PARAMS.n_elements
        nodes_checked = CONFIG.env.nodes_checked
        prev_node = -8
        node_spacer = int(self.n_elements / nodes_checked)

        node_numbers = []
        for i in range(nodes_checked):
            node_numbers.append(node_spacer+prev_node)
            prev_node = node_numbers[-1]
        self.node_numbers = np.array(node_numbers)

        # Creating the rod simulator and saving initial simulator states for later resetting
        self.create_simulator()

        self.initial_position_collection = self.rod_object.position_collection.copy()
        self.initial_director_collection = self.rod_object.director_collection.copy()
        self.initial_lengths = self.rod_object.lengths.copy()
        self.initial_tangents = self.rod_object.tangents.copy()

    def create_simulator(self):

        # Creating the simulator 
        self.rod_simulator = RodSimulator()

        # Creating rod object
        self.rod_object = CosseratRod.straight_rod(
            n_elements=ROD_PARAMS.n_elements,
            start=np.array(ROD_PARAMS.start),
            direction=np.array(ROD_PARAMS.direction),
            normal=np.array(ROD_PARAMS.normal),
            base_length=ROD_PARAMS.base_length,
            base_radius=ROD_PARAMS.base_radius,
            density=ROD_PARAMS.density,
            youngs_modulus=ROD_PARAMS.youngs_modulus,
            shear_modulus=ROD_PARAMS.shear_modulus,
        )

        # Add rod to simulator
        self.rod_simulator.append(self.rod_object)

        # Constrain rod
        self.rod_simulator.constrain(self.rod_object).using(
            OneEndFixedBC,                  # Displacement BC being applied
            constrained_position_idx=(0,),  # Node number to apply BC
            constrained_director_idx=(0,)   # Element number to apply BC
        )

        # Adding tendon forcing to rod
        self.rod_simulator.add_forcing_to(self.rod_object).using(
            OctoTendonForces,
            vertebra_height_long = TENDON_PARAMS.vertebra_height_long,
            num_vertebrae_long = TENDON_PARAMS.num_vertebrae_long,
            first_vertebra_node_long = TENDON_PARAMS.first_vertebra_node_long,
            final_vertebra_node_long = TENDON_PARAMS.final_vertebra_node_long,
            vertebra_mass_long = TENDON_PARAMS.vertebra_mass_long,
            vertebra_height_short = TENDON_PARAMS.vertebra_height_short,
            num_vertebrae_short = TENDON_PARAMS.num_vertebrae_short,
            first_vertebra_node_short = TENDON_PARAMS.first_vertebra_node_short,
            final_vertebra_node_short = TENDON_PARAMS.final_vertebra_node_short,
            vertebra_mass_short = TENDON_PARAMS.vertebra_mass_short,
            tendon_tensions = self.NN_tendon_tensions,
            n_elements = self.n_elements,
        )

        # Adding gravity forces to rod
        self.rod_simulator.add_forcing_to(self.rod_object).using(
            GravityForces,
            acc_gravity = SIM_PARAMS.gravity_vector
        )

        # Adding damping effects to rod
        self.rod_simulator.dampen(self.rod_object).using(
            AnalyticalLinearDamper,
            damping_constant=SIM_PARAMS.damping_constant,
            time_step = self.time_step
        )

        # Create dictionary to hold data from callback function
        self.callback_data_rod_object= defaultdict(list)

        # Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
        self.rod_simulator.collect_diagnostics(self.rod_object).using(
            MyCallBack, step_skip=self.steps_per_learn_update, callback_params=self.callback_data_rod_object)

        self.rod_simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.rod_simulator
        )

    def nan_detected_function(self):
        # Handles the appearance of nan values in the simulation. Handles this, reports it, and finishes episode.

        print('NaN value detected in simulation state, simulation exit.')
        self.nan_counter += 1

        reward = -1.0 # Minimum normalized reward allowed

        # Save information to a text file
        write_errorfile(self, "nan", self.timestamp_start, self.state_history, self.target_position, self.delta, self.tip_velocity, self.action_history, 
                        self.step_count, self.callback_data_rod_object)
            
        self.delta = np.array([-self.L, -self.L, -self.L])
        self.state = self.target_position - self.delta
        current_distance = np.linalg.norm(self.delta)
        self.state_history = np.vstack((self.state_history[1:], self.state)) # Adds new state at -1 and removes oldest at 0


        obs = np.concatenate((self.state_history.flatten(), 
                              self.target_position, 
                              self.delta, 
                              self.tip_velocity, 
                              self.tip_speed, 
                              self.node_speeds,
                              self.node_positions, 
                              self.action_history.flatten()))
        done = True
        info = {'termination_reason': 'nan_detected'}

        return obs, reward, done, done, info 

    def step(self, action):

        self.raw_action = action
        self.scaled_action = (action + 1.0) / 2.0 * self.max_tension # Scaling from [-1, 1] to [0, self.max_tension]

        self.action_history = np.vstack((self.action_history[1:], self.scaled_action)) # Adds new action at -1 and removes oldest at 0

        if self.step_count == 0:
            self.action_history[-2] = self.action_history[-1] # Makes the previous action the same as the current action on the first step only

        # Here is where the NN chooses an action to influence the system, and this action is taken by the simulator
        # The [:] ensures that the underlying array in memory is changed, thus the instantiated Forcing class can use this as well since it checks that 
        # point in memory as well.
        self.NN_tendon_tensions[:] = self.scaled_action


        # Do multiple time step of simulation for (one learning step)
        for _ in range(self.steps_per_learn_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.rod_simulator,
                self.time_tracker,
                self.time_step,
            )

        self.state, self.tip_velocity, self.tip_speed, self.node_speeds, self.node_positions = get_state(np.array(self.rod_object.position_collection[:,-1].tolist()),
                                                                                                                  self.rod_object.velocity_collection, 
                                                                                                                  self.rod_object.position_collection,
                                                                                                                  self.node_numbers, 
                                                                                                                  self.n_elements)
        done = False
        reward = 0

        # Position of the rod cannot be NaN, stops the simulation and resets the episode
        invalid_values_condition = _isnan_check(self.state)

        # Runs nan detection reporting and wraps up episode        
        if invalid_values_condition == True:
            self.nan_detected_function()
        
        # Maximum amount of steps reached
        if self.step_count + 1 == self.max_steps:   
            done = True
            info = {'termination_reason': 'max_steps'}
            self.max_steps_counter += 1

        
        self.delta =  self.target_position - self.state 
        delta_prev = self.target_position - self.state_history[-1]
        current_distance = np.linalg.norm(self.delta)

        # Computing reward
        reward_scalar, reward_components = compute_reward(dist=current_distance,
                                                          tip_speed=self.tip_speed[0],
                                                          action_curr=self.action_history[-1] / self.max_tension,
                                                          action_prev=self.action_history[-2] / self.max_tension,
                                                          node_speeds=self.node_speeds,
                                                          best_dist=self.best_distance,
                                                          num_tendons=self.num_tendons,
                                                          target_position=self.target_position,
                                                          current_position=self.state,
                                                          tip_velocity_vector=self.tip_velocity,
                                                          goal_reached_flag=self.goal_reached_flag
                                                          )
        reward += reward_scalar

        info = {'distance_norm_to_target': current_distance, 'reward_components':reward_components}

        self.state_history = np.vstack((self.state_history[1:], self.state)) # Adds new state at -1 and removes oldest at 0
        self.best_distance = min(current_distance, self.best_distance)

        obs = np.concatenate((self.state_history.flatten(), 
                              self.target_position, 
                              self.delta, 
                              self.tip_velocity, 
                              self.tip_speed, 
                              self.node_speeds,
                              self.node_positions, 
                              self.action_history.flatten())) 
        self.score += reward
        self.step_count += 1
        self.total_step_count += 1

        return obs, reward, done, done, info 


    def render(self):
        pass
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Setting the random seed:
        np.random.seed(seed)

        # Setting a new target position for the NN to be trained generally
        self.target_position = np.array([np.random.uniform(self.L - self.X_variation, self.L), 
                                         np.random.uniform(-self.Y_variation, self.Y_variation),
                                         np.random.uniform(-self.Z_variation, self.Z_variation)]) 


        # Resetting the tendon tensions to zero for all of them, preserving the pointer
        self.NN_tendon_tensions[:] = 0.0

        # Resetting kinematic state
        self.rod_object.position_collection[:] = self.initial_position_collection
        self.rod_object.velocity_collection[:] = 0.0
        self.rod_object.acceleration_collection[:] = 0.0
        self.rod_object.director_collection[:] = self.initial_director_collection
        self.rod_object.omega_collection[:] = 0.0
        self.rod_object.alpha_collection[:] = 0.0

        # Resetting forces/torques
        self.rod_object.internal_forces[:] = 0.0
        self.rod_object.internal_torques[:] = 0.0
        self.rod_object.external_forces[:] = 0.0
        self.rod_object.external_torques[:] = 0.0

        # Resetting derived geometric quantities (depend on position/directors)
        self.rod_object.lengths[:] = self.initial_lengths
        self.rod_object.tangents[:] = self.initial_tangents
        self.rod_object.dilatation[:] = 1.0
        self.rod_object.voronoi_dilatation[:] = 1.0
        self.rod_object.dilatation_rate[:] = 0.0

        # Resetting the episode score to zero
        self.score = 0.0
        self.step_count = 0
        self.time_tracker = np.float64(0.0)
        self.previous_action = None
        self.goal_reached_flag = np.array([False])
        
        self.state, self.tip_velocity, self.tip_speed, self.node_speeds, self.node_positions = get_state(np.array(self.rod_object.position_collection[:,-1].tolist()),
                                                                                                                  self.rod_object.velocity_collection,
                                                                                                                  self.rod_object.position_collection,
                                                                                                                  self.node_numbers,
                                                                                                                  self.n_elements)
        self.delta = self.target_position - self.state
        self.best_distance = np.linalg.norm(self.delta)
        
        self.action_history = np.zeros((CONFIG.env.action_history_len,8))
        self.state_history = np.zeros((CONFIG.env.state_history_len,3))


        obs = np.concatenate((self.state_history.flatten(), 
                              self.target_position, 
                              self.delta, 
                              self.tip_velocity, 
                              self.tip_speed, 
                              self.node_speeds,
                              self.node_positions, 
                              self.action_history.flatten()))
        info = {'Reset the environment'}

        return obs, info