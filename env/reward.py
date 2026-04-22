from numba import njit
import numpy as np
from sim.sim_params import TENDON_PARAMS

# ─── Tuning constants ────────────────────────────────────────────────────────
# Adjust these to shape agent behaviour without touching the logic below.
 
# Distance reward
DIST_THRESHOLD       = 0.08    # Goal radius (m) — tip is "at target" below this
DIST_LINEAR_SLOPE    = -15.0   # Slope of linear region (far from target). Was 16.8.
                                # Higher → stronger pull toward goal from far away.
DIST_QUAD_K          = 2000.0  # Curvature of quadratic region (near target). Keep as-is.
DIST_STABLE_MAX      = 2.5     # Max bonus for holding still inside goal radius

GOAL_DIST_THRESHOLD  = 0.08    # What is considered "reaching" the goal
GOAL_REACH_BONUS     = 50.0    # Bonus for reaching the goal, happens only once per episode
 
# Antagonist penalty
ANTAG_THRESHOLD      = 0.05    # Min tension for both tendons to be considered "active"
ANTAG_PENALTY        = 0.2     # Penalty per co-active antagonist pair
 
# Tensions penalty (replaces tendon switching penalty)
TENSIONS_THRESHOLD   = 0.2     # Min Δtension to trigger penalty (20% of normalized action [0,1])
TENSIONS_PENALTY     = 2.0     # Penalty scale per tendon (× Δtension magnitude)
 
# Node speeds penalty (gated to near-target only)
NODE_SPEED_THRESHOLD = 0.1     # Min node speed to trigger penalty
NODE_SPEED_PENALTY   = 3.0     # Penalty scale per node (× speed)
NODE_SPEED_CAP       = 5.0     # Hard cap on total node speed penalty magnitude
NODE_SPEED_GATE      = 0.20    # Only penalise node speeds when dist < this (m).
                                # Nodes naturally move while tip travels — only
                                # penalise stillness requirement near goal.
 
# Tip speed penalty
TIP_SPEED_K          = 1.0     # Quadratic penalty scale. Was 0.2 — increased to
                                # meaningfully suppress oscillations near goal.
STABLE_SPEED_MAX     = 0.08     # Speed at which the tip can move and considered stable
 
# Best distance bonus
BEST_DIST_K          = 5.0     # Bonus scale for improvements to best distance
 
# Correct direction bonus (gated to far-from-target only)
DIRECTION_K          = 0.8     # Bonus scale for velocity aligned with target direction.
                                # Applied only when dist > DIST_THRESHOLD so it does
                                # not conflict with tip speed penalty near goal.
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True)
def antagonist_penalty(action, penalty_per_pair=ANTAG_PENALTY):
    """
    This function aims to curb the activation of opposite tendons of the same length (forces and moments will cancel out)
    Pairs are defined by the OctoTendonForces layout:
        (0,1), (2,3) — long tendon pairs
        (4,5), (6,7) — short tendon pairs
    """
    antagonist_pairs = [
        (0, 1),  # Long tendon pair
        (2, 3),  # Long tendon pair
        (4, 5),  # Short tendon pair
        (6, 7),  # Short tendon pair
    ]

    penalty = 0.0
    # Penalizes antagonist tendons and the penalization is based on the "wasted force" of the weaker activated tendon
    for i, j in antagonist_pairs:
        penalty -= penalty_per_pair * min(action[i], action[j]) 
        
    return penalty

@njit(cache=True)
def tensions_penalty(action_prev, action_curr, num_tendons,
                     penalty_per_tendon=TENSIONS_PENALTY):


    """
    This function aims to curb drastic changes in the tendon tensions.
    The purpose is to try to attenuate oscillations in the trained agent.
    This subsumes tendon switching: a 0→x switch is always a large Δtension
    and will be caught here, so a separate switching penalty is not needed.
    """
    delta = np.abs(action_curr - action_prev)
    return -penalty_per_tendon * np.sum(delta**2)

@njit(cache=True)
def node_speeds_penalty(node_speeds,
                        threshold=NODE_SPEED_THRESHOLD,
                        penalty_given=NODE_SPEED_PENALTY,
                        cap=NODE_SPEED_CAP):
    """
    Penalises velocities of non-tip rod nodes to discourage erratic rod motion.
    Call this only when dist < NODE_SPEED_GATE (gating is done in compute_reward)
    so the agent is not penalised for natural motion while approaching the target.
    A hard cap prevents this term from drowning out the distance signal.
    """
    penalty = 0.0
    for value in node_speeds:
        if value >= threshold:
            penalty -= penalty_given * value
    return max(-cap, penalty)

@njit(cache=True)
def tip_speed_penalty(tip_speed, k_factor=TIP_SPEED_K):
    """
    Quadratic penalty on tip speed. Scaled up from k=0.2 to suppress oscillations
    meaningfully. Applies everywhere (no gating) since we always want smooth motion.
    """
    return -k_factor * tip_speed ** 2

@njit(cache=True)
def best_distance_bonus(dist, best_distance, k_factor=BEST_DIST_K):
    """
    Rewards improvements to the episode's best-so-far distance.
    Non-stationary within an episode — if value loss climbs during training,
    consider replacing with a fixed potential term: k / (dist + eps).
    """
    bonus = 0.0
    if dist < best_distance:
        improvement_ratio = (best_distance - dist) / best_distance
        bonus = k_factor * improvement_ratio
    return bonus

@njit(cache=True)
def correct_direction_bonus(target_position, current_position,
                            tip_velocity_vector, k_factor=DIRECTION_K):
    """
    Rewards tip velocity aligned with the target direction.
    Gated to far-from-target (dist > DIST_THRESHOLD) in compute_reward so it
    does not conflict with the tip speed penalty near the goal.
    """
    diff = target_position - current_position
    norm = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    if norm < 1e-8:
        return 0.0
    target_dir = diff / norm
    goal_speed = np.dot(np.ascontiguousarray((tip_velocity_vector)), np.ascontiguousarray((target_dir)))
    return goal_speed * k_factor

@njit(cache=True)
def goal_reach_bonus(dist, goal_reached_flag):
    if dist <= GOAL_DIST_THRESHOLD and not goal_reached_flag[0]:
        goal_reached_flag[0] = True
        return GOAL_REACH_BONUS
    else:
        return 0.0


def compute_reward(dist, tip_speed, action_curr, action_prev,
                   node_speeds, best_dist, num_tendons,
                   target_position, current_position, tip_velocity_vector, goal_reached_flag):
    """
    Compute the total reward for one environment step.
 
    Parameters
    ----------
    dist                 : float   — Euclidean distance from tip to target (m)
    tip_speed            : float   — Speed of the tip node (m/s)
    action_curr          : array   — Current tendon tensions [0, 1]^8
    action_prev          : array   — Previous tendon tensions
    node_speeds          : array   — Speeds of non-tip rod nodes (m/s)
    best_dist            : float   — Best (minimum) distance achieved so far this episode
    num_tendons          : int     — Number of tendons (8)
    target_position      : array   — 3D target position (m)
    current_position     : array   — 3D current tip position (m)
    tip_velocity_vector  : array   — 3D tip velocity vector (m/s)
    threshold            : float   — Goal radius (m)
    """
 
    # ── 1. Distance reward (piecewise, C1-continuous) ────────────────────────
    t = DIST_THRESHOLD
    m = DIST_LINEAR_SLOPE
    Q = DIST_QUAD_K


    x_meet = m/(2*Q)+t              # X coordinate where the slope of the quadratic and linear reward functions meet
    y_meet = Q*(x_meet-t)**2        # Y coordinate where the slope of the quadratic and linear reward functions meet
    b = y_meet-m*(x_meet)           # Y-intercept for the linear reward function

    if dist < x_meet:
        # Quadratic bowl — agent is near goal
        distance_reward = DIST_QUAD_K * (dist - t) ** 2
        # Bonus for holding still inside the goal region
        distance_reward += min(DIST_STABLE_MAX, 20.0 * max(0.0, STABLE_SPEED_MAX - tip_speed))
    else:
        # Linear — agent is far from goal.
        # Intercept chosen so the two pieces share the same value and slope at x_meet.
        distance_reward = m * dist + b
 
    reward = (distance_value:= distance_reward)
 
    # ── 2. Antagonist penalty ────────────────────────────────────────────────
    reward += (antagonist_value:= antagonist_penalty(action_curr))
 
    # ── 3. Tensions penalty (replaces tendon switching penalty) ──────────────
    reward += (tensions_value:= tensions_penalty(action_prev, action_curr, num_tendons))
 
    # ── 4. Node speeds penalty (near-target only) ────────────────────────────
    # Silenced for now because tip speed is probably good enough for stabilization. If wiggling is a problem, then reintroduce
    node_speeds_value = 0.0
    correct_direction_value = 0.0
    # if dist < t:
    #     reward += (node_speeds_value:= node_speeds_penalty(node_speeds))
 
    # ── 5. Tip speed penalty (always active) ────────────────────────────────
    reward += (tip_speed_value:= tip_speed_penalty(tip_speed))
 
    # ── 6. Best distance bonus ───────────────────────────────────────────────
    reward += (best_distance_value:= best_distance_bonus(dist, best_dist))
 
    # ── 7. Correct direction bonus  ────────────────────────────────────────────
    reward += (correct_direction_value:= correct_direction_bonus(
        target_position, current_position, tip_velocity_vector
    ))

    # ── 7. Goal reached bonus  ────────────────────────────────────────────────
    reward += (goal_reach_value:= goal_reach_bonus(dist, goal_reached_flag))

    components = {
        'distance': distance_value,
        'antagonist': antagonist_value,
        'tensions': tensions_value,
        'node_speeds': node_speeds_value,
        'tip_speed': tip_speed_value,
        'best_distance': best_distance_value,
        'correct_direction': correct_direction_value,
        'goal_reach': goal_reach_value,
    }

    # print(f"dist\t{dist}\tx_meet={x_meet}\ty_meet={y_meet}\tb={b}\tm={m}")
    # print(f"distance_reward\t{distance_value}")
    # print(f"antagonist_penalty\t{antagonist_value}")
    # print(f"tensions_penalty\t{tensions_value}")
    # print(f"node_speeds_penalty\t{node_speeds_value}")
    # print(f"tip_speed_penalty\t{tip_speed_value}")
    # print(f"best_distance_bonus\t{best_distance_value}")
    # print(f"correct_direction_bonus\t{correct_direction_value}")
    # print(f"goal_reached_value\t{goal_reach_value}")

    return reward, components