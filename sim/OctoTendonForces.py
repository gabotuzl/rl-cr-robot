from elastica.external_forces import NoForces
import numpy as np
from numba import njit
from sim.sim_params import SIM_PARAMS

class OctoTendonForces(NoForces):
    """
    
    """

    def __init__(self, vertebra_height_long, num_vertebrae_long, first_vertebra_node_long, final_vertebra_node_long, vertebra_mass_long,
    vertebra_height_short, num_vertebrae_short, first_vertebra_node_short, final_vertebra_node_short, vertebra_mass_short, tendon_tensions, n_elements):
        """

        Parameters 
        ----------
        vertebra_height_long: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space. This parameter relates to ALL LONG tendon systems.
        num_vertebrae_long: int
            Amount of vertebrae to be used in the system. This relates to ALL LONG tendon systems.
        first_vertebra_node_long: int
            The first node to have a vertebra, from the base of the rod to the tip. This relates to ALL LONG tendon systems.
        final_vertebra_node_long: int
            The last node to have a vertebra, from the base of the rod to the tip. This relates to ALL LONG tendon systems.
        vertebra_mass_long: float
            Total mass of a single vertebra disk. This relates to ALL LONG tendon systems.
        vertebra_height_short: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space. This parameter relates to ALL SHORT tendon systems.
        num_vertebrae_short: int
            Amount of vertebrae to be used in the system. This relates to ALL SHORT tendon systems.
        first_vertebra_node_short: int
            The first node to have a vertebra, from the base of the rod to the tip. This relates to ALL SHORT tendon systems.
        final_vertebra_node_short: int
            The last node to have a vertebra, from the base of the rod to the tip. This relates to ALL SHORT tendon systems.
        vertebra_mass_short: float
            Total mass of a single vertebra disk. This relates to ALL SHORT tendon systems.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        """
        super(OctoTendonForces, self).__init__()

        # Initializing class attribute to be used in other methods
        self.n_elements = n_elements

        # Calculating the weights vector for the vertebrae. By default, the direction of gravity is in the global -Z direction
        vertebra_weights_vector_long = vertebra_mass_long * SIM_PARAMS.gravity_vector
        vertebra_weights_vector_short = vertebra_mass_short * SIM_PARAMS.gravity_vector

        # Creating vector containing the node numbers with the vertebrae for the long tendon
        vertebra_nodes_long = []
        vertebra_increment_long = (final_vertebra_node_long - first_vertebra_node_long)/(num_vertebrae_long - 1)
        for i in range(num_vertebrae_long):
            vertebra_nodes_long.append(round(i * vertebra_increment_long + first_vertebra_node_long))

        # Creating vector containing the node numbers with the vertebrae for the short tendon
        vertebra_nodes_short = []
        vertebra_increment_short = (final_vertebra_node_short - first_vertebra_node_short)/(num_vertebrae_short - 1)
        for i in range(num_vertebrae_short):
            vertebra_nodes_short.append(round(i * vertebra_increment_short + first_vertebra_node_short))

        # Creating the vector that describe the local vertebra orientation of every vertebra
        dummy_vector = np.array(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]))
        vertebra_heights_long = dummy_vector * vertebra_height_long
        vertebra_heights_short = dummy_vector * vertebra_height_short

        # Turning all into numpy array
        self.vertebra_nodes_long = np.array(vertebra_nodes_long)
        self.vertebra_nodes_short = np.array(vertebra_nodes_short)
        self.vertebra_heights_long = np.array(vertebra_heights_long)
        self.vertebra_heights_short = np.array(vertebra_heights_short)
        self.vertebra_weights_vector_long = np.array(vertebra_weights_vector_long)
        self.vertebra_weights_vector_short = np.array(vertebra_weights_vector_short)

        # Doesn't copy the array, it makes self.tensions_vector point to the same array object in memory as self.NN_tendon_tensions in the simulator env.
        self.tensions_vector = tendon_tensions

        self.forcing_counter = 0

    def apply_forces(self, system, time: np.float64 = 0.0):
        self.forcing_counter += 1
        # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra for the long and short tendons
        unit_norm_vector_array_long = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), self.vertebra_nodes_long, self.vertebra_heights_long)
        unit_norm_vector_array_short = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), self.vertebra_nodes_short, self.vertebra_heights_short)

        # Computes the forces in each vertebra
        self.force_data_long = self.compute_forces(self.tensions_vector[:4], self.vertebra_nodes_long, unit_norm_vector_array_long)
        self.force_data_short = self.compute_forces(self.tensions_vector[4:], self.vertebra_nodes_short, unit_norm_vector_array_short)

        # Creating the force data set to apply to the rod
        apply_force = np.zeros((3,self.n_elements+1))

        # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
        # Apply tendon forces (per tendon)
        for i in range(len(self.vertebra_nodes_long)):
            apply_force[:, self.vertebra_nodes_long[i]] += self.vertebra_weights_vector_long        # Applies weight once per vertebra
            for k in range(4):
                apply_force[:, self.vertebra_nodes_long[i]] += self.force_data_long[k][i]           # Applies force of 4 tendons in each vertebra

        for i in range(len(self.vertebra_nodes_short)):
            apply_force[:, self.vertebra_nodes_short[i]] += self.vertebra_weights_vector_short      # Applies weight once per vertebra
            for k in range(4):
                apply_force[:, self.vertebra_nodes_short[i]] += self.force_data_short[k][i]         # Applies force of 4 tendons in each vertebra

        
        # Applies forces to the rod
        system.external_forces += apply_force


    def apply_torques(self, system, time: np.float64 = 0.0):
        # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
        # Creating the array which will contain the transformed force vectors
        transformed_force_data_long = np.zeros((4, len(self.vertebra_nodes_long), 3), dtype=np.float64)
        transformed_force_data_short = np.zeros((4, len(self.vertebra_nodes_short), 3), dtype=np.float64)

        # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
        # Doing this for all 8 sets of vertebrae
        for k in range(4):
            for i in range(len(self.vertebra_nodes_long)):
                transformed_force_data_long[k][i] = np.ascontiguousarray(system.director_collection[...,(self.vertebra_nodes_long[i]-1)]) @ np.ascontiguousarray(self.force_data_long[k][i])

        for k in range(4):
            for i in range(len(self.vertebra_nodes_short)):
                transformed_force_data_short[k][i] = np.ascontiguousarray(system.director_collection[...,(self.vertebra_nodes_short[i]-1)]) @ np.ascontiguousarray(self.force_data_short[k][i])

        # Calculating torque vectors for vertebrae using both vertical and horizontal tendons, of long and short lengths
        apply_torque_long = self.compute_torques(self.vertebra_heights_long, self.vertebra_nodes_long, transformed_force_data_long, self.n_elements)
        apply_torque_short = self.compute_torques(self.vertebra_heights_short, self.vertebra_nodes_short, transformed_force_data_short, self.n_elements)

        # Applying the torque data set to the rod
        system.external_torques += apply_torque_long + apply_torque_short

    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_heights_vector):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae. This is done for all 8 vertebrae sets

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        n = len(vertebra_nodes)
        unit_norm_vector_array = np.zeros((4, n+1, 3), dtype=np.float64) # n+1 to account for fixed node

        for k in range(4):
            for i in range(n):
                # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
                if i == 0:
                    current_vertebra = 0
                else:
                    current_vertebra = vertebra_nodes[i-1]
                
                next_vertebra = vertebra_nodes[i]

                # Setting up values to be used iteratively
                current_node = position_collection[:, current_vertebra]
                next_node = position_collection[:, next_vertebra]

                current_R = director_collection[:, :, current_vertebra]
                next_R = director_collection[:, :, next_vertebra]

                h = vertebra_heights_vector[k]

                # Calculating relative position vector between vertebrae, considering the vertebra height
                delta_vector = (next_node + np.ascontiguousarray(next_R.T) @ h) - (current_node + np.ascontiguousarray(current_R.T) @ h)
                # Calculating the unit-normed vector based on the differences calculated in the previous step
                norm = np.sqrt(delta_vector[0]**2 + delta_vector[1]**2 + delta_vector[2]**2)
                unit_norm_vector_array[k, i] = delta_vector / norm      

        return unit_norm_vector_array


    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):
        # Returns an array containing the resulting tendon force vectors for each of the 8 sets of vertebrae

        # Creating array to store forces in vertebrae
        force_data = np.zeros((4, len(vertebra_nodes), 3), dtype=np.float64)

        for k in range(4):
            for i in range(len(vertebra_nodes)):
                # This for loop multiplies the unit normed vectors calculated previously, with the tension of each tendon, thus creating the force vector for each vertebra
                # Contiguous array to increase speed in njit decorator
                force_current_prev = unit_norm_vector_array[k][i] * -tension[k]
                force_current_next = unit_norm_vector_array[k][i+1] * tension[k]

                # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
                force_data[k][i] = force_current_prev + force_current_next

        return force_data

    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_heights_vector, vertebra_nodes, transformed_force_data, n_elements):
        # Returns array containing tendon torques applied to respective vertebrae nodes in each of the 8 vertebra sets, in the format PyElastica uses for external forcing 

        apply_torque = np.zeros((3,n_elements))
        
        # Goes through vertebra nodes to calculate torques for them
        for k in range(4):
            for i in range(len(vertebra_nodes)):

                # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
                torque = np.cross(vertebra_heights_vector[k], transformed_force_data[k][i])
            
                node_idx = vertebra_nodes[i] - 1  # converting to 0-indexed
                apply_torque[:, node_idx] += torque

        return apply_torque