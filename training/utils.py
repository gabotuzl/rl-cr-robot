from env import cr_env

def write_errorfile(self, error_type, timestamp_start, state_history, target_position, delta, tip_velocity, action_history, 
                    step_count, reward_history, callback_data_rod_object):
    with open(f"logs/{error_type}_log.txt", "a") as f:
        timestamp_now = datetime.now()
        time_diff = str(timestamp_now - timestamp_start)
        f.write(f"=== {error_type} DETECTED ===\n")
        f.write(f"TIME: {time_diff}\n")
        f.write(f"State history: {state_history.tolist()}\n")
        f.write(f"Target Position: {target_position.tolist()}\n")
        f.write(f"Delta: {delta.tolist()}\n")
        f.write(f"Tip velocity: {tip_velocity}\n")
        f.write(f"Action History: {action_history.tolist()}\n")
        f.write(f"Step Count: {step_count}\n")
        f.write(f"Reward history: {reward_history}\n")

        position_array = np.array(callback_data_rod_object['position'])
        f.write(f"Position history shape: {position_array.shape}\n")
        f.write(f"Position history: {position_array.tolist()}\n")
        f.write("\n\n")

def make_env(rank=0):
    def _init():
        env = cr_env()  
        env = Monitor(env, filename=f"{CONFIG.paths.log_dir}env_{rank}")
        return env
    return _init

@njit(cache=True)
def get_state(tip_position, velocity_collection, node_numbers, n_elements):

    # Getting the rod tip's XYZ position
    tip_pos = tip_position

    # Getting the velocity of the tip 
    x_vel = velocity_collection[0][-1]
    y_vel = velocity_collection[1][-1]
    z_vel = velocity_collection[2][-1]
    tip_velocity = np.array([x_vel, y_vel, z_vel])

    tip_speed = np.linalg.norm(tip_velocity)

    # Getting the speeds of nodes on the rod
    node_speeds = np.empty(len(node_numbers), dtype=np.float64)
    i = 0
    for node in node_numbers:
        x_vel = (velocity_collection[0][node])
        y_vel = (velocity_collection[1][node])
        z_vel = (velocity_collection[2][node])

        vel = np.array((x_vel, y_vel, z_vel))
        
        node_speeds[i] = np.linalg.norm(vel)
        i += 1

    return tip_pos, tip_velocity, tip_speed, node_speeds