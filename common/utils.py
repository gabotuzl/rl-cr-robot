from numba import njit
import numpy as np

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

@njit(cache=True)
def get_state(tip_position, velocity_collection, node_numbers, n_elements):

    # Getting the velocity of the tip 
    tip_velocity = velocity_collection[:, -1]

    tip_speed = np.sqrt(tip_velocity[0]**2 + tip_velocity[1]**2 + tip_velocity[2]**2)

    # Getting the speeds of nodes on the rod
    node_speeds = np.empty(len(node_numbers), dtype=np.float64)
    for i, node in enumerate(node_numbers):
        vel = velocity_collection[:, node]
        node_speeds[i] = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)  

    return tip_position, tip_velocity, tip_speed, node_speeds