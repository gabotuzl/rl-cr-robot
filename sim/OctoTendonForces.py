class OctoTendonForces(NoForces):
    """
    NOTE: THIS CLASS IS EDITED FROM THE ORIGINAL, IT DOES NOT RECEIVE INFO FROM ANY ROS2 NETWORK, IT IS SIMPLY FOR THE OPEN-LOOP CASE.

    This class uses a Quad Tendon Configuration. Its purpose is to allow for closed loop control of the tip of the rod through tendon actuation.
    Quad Tendon Configuration: Tendon configuration which has 4 long tendons (up, down, left, right) and 4 short tendons (up, down, left, right) to allow for the simultaneous 
                               activation of multiple tendons to reach points in the 3D workspace of the continuum robot.
                              
        Attributes
        ----------
        vertebra_height_long: float
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space. This attribute relates to ALL LONG tendon systems.
        vertebra_height_short: numpy.ndarray
            Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space. This attribute relates to ALL SHORT tendon systems.
        num_vertebrae_long: int
            Amount of vertebrae to be used in the system. This attribute relates to ALL LONG vertebrae.
        num_vertebrae_short: int
            Amount of vertebrae to be used in the system. This attribute relates to ALL SHORT vertebrae.
        n_elements: int
            Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.
        vertebra_weight_vector_long: numpy.ndarray
            1D (dim) numpy array. Vector which specifies the orientation and magnitude of the weight of the LONG vertebrae (By default it is in the global -Z direction).
        vertebra_weight_vector_short: numpy.ndarray
            1D (dim) numpy array. Vector which specifies the orientation and magnitude of the weight of the SHORT vertebrae (By default it is in the global -Z direction).
        vertebra_nodes_long: list
            1D (dim) list. Contains the node numbers of every node with LONG vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node_long and final_vertebra_node_long, with an amount equal to num_vertebrae_long.
        vertebra_nodes_short:list
            1D (dim) list. Contains the node numbers of every node with SHORT vertebrae. The vertebrae are assumed to be uniformly spaced through the intervals specified by 
            first_vertebra_node_short and final_vertebra_node_short, with an amount equal to num_vertebrae_short.
        vertebra_height_vector_vertical_long: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the (ACTIVE) VERTICAL LONG vertebrae in the system.
        vertebra_height_vector_horizontal_long: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the (ACTIVE) HORIZONTAL LONG vertebrae in the system.
        vertebra_height_vector_vertical_short: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the (ACTIVE) VERTICAL SHORT vertebrae in the system.
        vertebra_height_vector_horizontal_short: numpy.ndarray
            1D (dim) numpy array. Describes the orientation and height in space of the (ACTIVE) HORIZONTAL SHORT vertebrae in the system.
        tension_vertical_long: float
            Tension applied to the (ACTIVE) VERTICAL LONG tendon in the system.
        tension_horizontal_long: float
            Tension applied to the (ACTIVE) HORIZONTAL LONG tendon in the system.
        tension_vertical_short: float
            Tension applied to the (ACTIVE) VERTICAL SHORT tendon in the system.
        tension_horizontal_short: float
            Tension applied to the (ACTIVE) HORIZONTAL SHORT tendon in the system.
        force_data_vertical_long: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the (ACTIVE) VERTICAL LONG vertebrae.
        force_data_horizontal_long: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the (ACTIVE) HORIZONTAL LONG vertebrae.
        force_data_vertical_short: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the (ACTIVE) VERTICAL SHORT vertebrae.
        force_data_horizontal_short: numpy.ndarray
            2D (dim,3) numpy array. Contains the force vectors caused by tendon forcing for each of the (ACTIVE) HORIZONTAL SHORT vertebrae.
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

        # Initializing class attributes to be used in other methods
        self.vertebra_height_long = vertebra_height_long
        self.vertebra_height_short = vertebra_height_short
        self.num_vertebrae_long = num_vertebrae_long
        self.num_vertebrae_short = num_vertebrae_short
        self.n_elements = n_elements

        # Calculating the weights vector for the vertebrae. By default, the direction of gravity is in the global -Z direction
        vertebra_weight_vector_long = np.array([0.0, 0.0, -vertebra_mass_long * 9.80665])
        vertebra_weight_vector_short = np.array([0.0, 0.0, -vertebra_mass_short * 9.80665])
        self.vertebra_weights_vector = np.concatenate(vertebra_weight_vector_long, vertebra_weight_vector_short)

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

        # Concatenating both vertebra nodes lists
        self.vertebra_nodes = np.concatenate((vertebra_nodes_long, vertebra_nodes_short))

        self.forcing_counter = 0

        # Creating the vector that describe the local vertebra orientation of every vertebra
        dummy_vector = np.array([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
        vertebra_heights_long = dummy_vector * self.vertebra_height_long
        vertebra_heights_short = dummy_vector * self.vertebra_height_short
        self.vertebra_heights_vector = np.concatenate((vertebra_heights_long, vertebra_heights_short))

        # Initializing all the attributes defined in the update_tendon_tension method, so that PyElastica can run even if the controller has not yet sent the parameters
        self.update_tendon_tension(tendon_tensions)


    def update_tendon_tension(self, tendon_tensions, time: np.float64 = 0.0):
        self.tensions_vector = tendon_tensions
        return

    def apply_forces(self, system: SystemType, time: np.float64 = 0.0):
        self.forcing_counter += 1
        # The application of the force data is done outside of the @njit decorated function because self.force_data needs to be referenced in self.compute_torques()

        # Retrieves relative position unit norm vectors between each vertebra for the long and short tendons
        unit_norm_vector_array = self.get_rotations(np.array(system.position_collection), np.array(system.director_collection), np.array(self.vertebra_nodes), self.vertebra_heights_vector)

        # Computes the forces in each vertebra
        self.force_data = self.compute_forces(self.tensions_vector, np.array(self.vertebra_nodes), unit_norm_vector_array)

        # Creating the force data set to apply to the rod
        apply_force = np.zeros((3,self.n_elements+1))

        # PyElastica handles forces in GLOBAL coord. system, so they are applied directly. Also, the vertebra weights are added to each vertebra
        for k in range(8):
            for i in range (len(self.vertebra_nodes)):
                apply_force[:,self.vertebra_nodes[i]] += self.force_data[k][i] + self.vertebra_weights_vector[i]

        # Applies forces to the rod
        system.external_forces += apply_force


    def apply_torques(self, system: SystemType, time: np.float64 = 0.0):
        # The force_data set and vertebra_weight_vector are expressed in the global coordinate frame and must be changed to local reference frames for torque application
        # Creating the array which will contain the transformed force vectors
        transformed_force_data = np.zeros((8, len(self.vertebra_nodes), 3), dtype=np.float64)

        # Transforming the force vectors calculated in the compute_forces method from the global reference frame to the local reference frame
        # Doing this for all 8 sets of vertebrae
        for k in range(8):
            for i in range(len(self.vertebra_nodes)):
                transformed_force_data[k][i] = np.ascontiguousarray(system.director_collection[...,(self.vertebra_nodes[i]-1)]) @ np.ascontiguousarray(self.force_data[k][i])

        # Calculating torque vectors for vertebrae using both vertical and horizontal tendons, of long and short lengths
        apply_torque = self.compute_torques(self.vertebra_heights_vector, np.array(self.vertebra_nodes), transformed_force_data, self.n_elements)

        # Applying the torque data set to the rod
        system.external_torques += apply_torque

    @staticmethod
    @njit(cache=True)
    def get_rotations(position_collection, director_collection, vertebra_nodes, vertebra_heights_vector):
        # Returns an array containing the unit norm vector which describes the orientation of each segment of tendon between vertebrae. This is done for all 8 vertebrae sets

        # Initializing unit_norm_vector_array to store the unit normed vectors that describe the global orientation of the forces in each vertebra
        unit_norm_vector_array = np.zeros((8, len(vertebra_nodes), 3), dtype=np.float64)

        for k in range(8):
            for i in range(len(vertebra_nodes)+1):
                # There is a +1 in the for loop to account for the force between the first vertebra and the fixed node

                # If statement, used for the case when i = 0 and thus there is no vertebra before this one, same for the final vertebra (no vertebra after that one)
                if i==0:
                    current_vertebra = 0
                    next_vertebra = vertebra_nodes[i]
                elif i==len(vertebra_nodes):
                    current_vertebra = vertebra_nodes[i-1]
                    next_vertebra = vertebra_nodes[i-1]
                else:
                    current_vertebra = vertebra_nodes[i-1]
                    next_vertebra = vertebra_nodes[i]

                # Setting up values to be used iteratively
                x_current = position_collection[0, current_vertebra]
                y_current = position_collection[1, current_vertebra]
                z_current = position_collection[2, current_vertebra]

                x_next = position_collection[0, next_vertebra]
                y_next = position_collection[1, next_vertebra]
                z_next = position_collection[2, next_vertebra]

                current_rotation_matrix = director_collection[...,current_vertebra]
                next_rotation_matrix = director_collection[...,next_vertebra]

                current_node = np.array([x_current, y_current, z_current])
                next_node = np.array([x_next, y_next, z_next])

                # Calculating relative position vector between vertebrae, considering the vertebra height
                delta_vector = (next_node + np.ascontiguousarray(next_rotation_matrix.T) @ np.ascontiguousarray(vertebra_heights_vector[k])) - (current_node + np.ascontiguousarray(current_rotation_matrix.T) @ np.ascontiguousarray(vertebra_heights_vector[k]))

                # Calculating the unit-normed vector based on the differences calculated in the previous step
                delta_vector_norm = np.linalg.norm(delta_vector)
                unit_norm_delta_vector = delta_vector / delta_vector_norm

                # This if statement is to stop unit_norm_delta_vector from becoming a 'nan'
                if i==len(vertebra_nodes):
                    unit_norm_delta_vector = np.zeros(3)

                # Storing the unit normed vector to be later used in the compute_forces method
                unit_norm_vector_array[k,i] = unit_norm_delta_vector

        return unit_norm_vector_array


    @staticmethod
    @njit(cache=True)
    def compute_forces(tension, vertebra_nodes, unit_norm_vector_array):
        # Returns an array containing the resulting tendon force vectors for each of the 8 sets of vertebrae

        # Creating array to store forces in vertebrae
        force_data = np.zeros((8, len(vertebra_nodes), 3), dtype=np.float64)

        for k in range(8):
            for i in range(len(vertebra_nodes)):
                # This for loop multiplies the unit normed vectors calculated previously, with the tension of each tendon, thus creating the force vector for each vertebra
                # Contiguous array to increase speed in njit decorator
                force_current_prev = unit_norm_vector_array[k][i] * -tension
                force_current_next = unit_norm_vector_array[k][i+1] * tension

                # Summing the components of both force vectors to get the final force vector, which is then stored for use in the apply_forces and compute_torques methods
                force_data[k][i] = force_current_prev + force_current_next

        return force_data

    @staticmethod
    @njit(cache=True)
    def compute_torques(vertebra_heights_vector, vertebra_nodes, transformed_force_data, n_elements):
        # Returns array containing tendon torques applied to respective vertebrae nodes in each of the 8 vertebra sets, in the format PyElastica uses for external forcing 

        for k in range(8):
            # Goes through vertebra nodes to calculate torques for them
            # Creating torque data set for storage
            torque_data = np.zeros((len(vertebra_nodes), 3),dtype=np.float64)
            for i in range(len(vertebra_nodes)):

                # Cross product between the vertebra height vector and the local force vector due to the tendons, to obtain the tendon torque for that vertebra
                torque_data[i] = np.cross(vertebra_heights_vector[k], transformed_force_data[k][i])
            
            # Appending the computed torque vector to the final torque data set
            apply_torque = np.zeros((3,n_elements))
            dummy_apply_torque = np.zeros((3,n_elements+1))

            m = 0
            for i in range(n_elements):
                if i in vertebra_nodes:
                    dummy_apply_torque[:,i] = torque_data[m]
                    m += 1
            apply_torque += dummy_apply_torque[:,1:]

        return apply_torque