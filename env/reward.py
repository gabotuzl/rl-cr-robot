from numba import njit
import numpy as np

@njit(cache=True)
def antagonist_penalty(action, threshold=0.05, penalty_per_pair=1.5):
    # This function aims to curb the activation of opposite tendons of the same length (forces and moments will cancel out)

    # These pairs are based on how the tendons are defined in the OctoTendonForces class
    antagonist_pairs = [
        (0, 1),  # Long tendon pair
        (2, 3),  # Long tendon pair
        (4, 5),  # Short tendon pair
        (6, 7),  # Short tendon pair
    ]

    penalty = 0.0
    for i, j in antagonist_pairs:
        if action[i] > threshold and action[j] > threshold:
            # Applies penalty if both antagonist tendons are active
            penalty -= penalty_per_pair
    
    return penalty

@njit(cache=True)
def tensions_penalty(action_history_2, action_history_1, num_tendons, threshold=0.3, penalty_per_tendon=0.8):
    # This function aims to curb drastic changes in the tendon tensions
    # The purpose is to try to attenuate oscillations in the trained agent

    delta_tensions = np.abs(action_history_2 - action_history_1)
    penalty = 0.0
    for k in range(num_tendons):
        if delta_tensions[k] >= threshold:
            penalty += -penalty_per_tendon * delta_tensions[k] 
    return penalty

@njit(cache=True)
def node_speeds_penalty(node_speeds, threshold=0.1, penalty_given=3.0):
    # This function aims to curb erratic movements by the rest of the rod
    # By penalizing velocities of other nodes in the rod, it will be incentivized to be still

    penalty = 0.0
    for value in node_speeds:
        if value >= threshold:
            penalty += -penalty_given * value

    return penalty

@njit(cache=True)
def tendon_switching_penalty(prev, current, num_tendons, penalty_per_tendon=0.8):
    # This function aims to penalize the agent for switching active tendons in an effort for it to
    # choose tendons at the beginning and not be changing them over and over
    penalty = 0.0
    marker = True
    for k in range(num_tendons):
        if (prev[k]==0.0 and current[k]!=0) or (current[k]==0.0 and prev[k]!=0):
            # Tendons have been switched ON/OFF which is undersirable
            penalty += -penalty_per_tendon
        if current[k] != 0.0:
            marker = False

    if marker == True:
        # All the tensions are zero, which is not good
        penalty += -10.0

    return penalty

@njit(cache=True)
def tip_speed_penalty(tip_speed, k_factor=0.2):
    return -k_factor * tip_speed ** 2

@njit(cache=True)
def best_distance_bonus(dist, best_distance, k_factor=5.0):
    # Rewards improvements to the best distance achieved
    bonus = 0.0
    if dist < best_distance:
        improvement_ratio = (best_distance - dist) / best_distance
        bonus = k_factor * improvement_ratio
        best_distance = dist
    return bonus

@njit(cache=True)
def correct_direction_bonus(target_position, current_position, tip_velocity_vector, k_factor=1.0):
    # Rewards movements TOWARDS the target position and penalizes movements away from the target position
    target_direction = (target_position - current_position) / np.linalg.norm((target_position - current_position))

    goal_speed = np.dot(tip_velocity_vector, target_direction)
    reward = goal_speed * k_factor

    return reward

def compute_reward(dist, tip_speed, action_curr, action_prev, node_speeds, best_dist, num_tendons):

    if dist < threshold-0.00134: # Reward is piecewise and C1 continuous to allow smooth gradient
        reward = 6000*(dist-threshold)**2 # Quadratic reward when the dist=0.048 (a little lower than threshold)
        reward += min(2.5, 20*max(0, (threshold - tip_speed))) # Reward for stable movement in the desired tip position
    else:
        reward = -16.8*dist + 1.332262 # Linear function that share same slope as quadratic function above at dist=0.07866

    antagonist_penalty_value = antagonist_penalty(action_history[-1]) # Penalty for activating opposite tendons of the same length
    reward += antagonist_penalty_value

    tensions_penalty_value = tensions_penalty(action_history[-2], action_history[-1], num_tendons) # Penalty for having drastic tension changes
    reward += tensions_penalty_value
    
    node_speeds_penalty_value = node_speeds_penalty(node_speeds) # Penalty for having velocities in the nodes which are not the tip (to discourage erratic movements of the rest of the rod)
    reward += node_speeds_penalty_value

    tendon_switching_penalty_value = tendon_switching_penalty(action_history[-2], action_history[-1], num_tendons)
    reward += tendon_switching_penalty_value # Penalty for switching tendons ON/OFF, also penalizing heavily a zero vector for the action

    tip_speed_penalty_value = tip_speed_penalty(tip_speed) # Penalty for having large tip speed (discourage wiggling and quick movements)
    reward += tip_speed_penalty_value

    best_distance_bonus_value = best_distance_bonus(dist, best_distance, k_factor=5.0)
    reward += best_distance_bonus_value

    # correct_direction_bonus_value = correct_direction_bonus(target_position, state, tip_velocity, k_factor=0.2)
    # reward += correct_direction_bonus_value
    # print("correct direction bonus: ", correct_direction_bonus_value, '\n')


    # print("antagonist penalty: ",antagonist_penalty_value)
    # print("tensions penalty: ", tensions_penalty_value)
    # print("node velocities penalty: ", node_speeds_penalty_value)
    # print("tendon switching penalty: ", tendon_switching_penalty_value)
    # print("tip speed penalty: ", tip_speed_penalty_value)
    # print("dist: ",dist)
    # print("total reward: ", reward, "\n")

    return reward