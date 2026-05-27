import numpy as np
import time
import json
import redis
import math
from enum import Enum, auto
from dataclasses import dataclass

DEG_TO_RAD = math.pi / 180.0

class State(Enum):
  INIT = auto()
  READY = auto()
  WINDUP = auto()
  THROW = auto()
  FOLLOW = auto()
  RETURN = auto()
  HOLD = auto()



@dataclass
class RedisKeys:
  cartesian_task_goal_position: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position"
  cartesian_task_goal_orientation: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_orientation"
  cartesian_task_current_position: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position"
  cartesian_task_current_orientation: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::current_orientation"
  active_controller: str = "opensai::controllers::Titania::active_controller_name"
  config_file_name: str = "::sai-interfaces-webui::config_file_name"

redis_keys = RedisKeys()

config_file_for_this_example = "basket.xml"
controller_to_use = "cartesian_controller"

def rot_y(deg):
  return np.array([[math.cos(deg * DEG_TO_RAD), 0, -math.sin(deg * DEG_TO_RAD)],
                   [0, 1, 0],
                   [math.sin(deg * DEG_TO_RAD), 0, math.cos(deg * DEG_TO_RAD)]])

def rot_x(deg):
  return np.array([[1, 0, 0],
                   [0, math.cos(deg * DEG_TO_RAD), -math.sin(deg * DEG_TO_RAD)],
                   [0, math.sin(deg * DEG_TO_RAD), math.cos(deg * DEG_TO_RAD)]])

def get_pos_error(curr_pos, goal_pos):
  return np.linalg.norm(goal_pos - curr_pos)

def get_ori_error(curr_ori, goal_ori):
  return np.linalg.norm(goal_ori - curr_ori)

def reached_pos(curr_pos, goal_pos, curr_ori, goal_ori, pos_tol, ori_tol):
    is_pos_reached = get_pos_error(curr_pos, goal_pos) < pos_tol
    is_ori_reached = get_ori_error(curr_ori, goal_ori) < ori_tol
    return is_pos_reached and is_ori_reached


def set_cartesian_goal(redis_client, goal_pos, goal_ori):
    redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(goal_pos.tolist()))
    redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(goal_ori.tolist()))

DEG_TO_RAD = math.pi / 180.0


# VALUE DECLARATIONS
# positions (tuned later when precise setup known)
home_pos = np.array([0.45, 0.0, 0.45])
ready_pos = np.array([0.50, 0.0, 0.48])
windup_pos = np.array([0.30, 0.0, 0.36])
release_pos = np.array([0.72, 0.0, 0.64])
follow_pos = np.array([0.82, 0.0, 0.66])

# orientations
base_ori = np.array([[1.0,0,0],[0,-1.0,0],[0,0,-1.0]])
ready_ori = base_ori @ rot_y(20 * DEG_TO_RAD).T
windup_ori = base_ori @ rot_y(25 * DEG_TO_RAD).T 
release_ori = base_ori @ rot_y(35 * DEG_TO_RAD).T 
follow_ori = release_ori
home_ori = ready_ori

# tolerances for position + orientation
POS_TOL_READY = 0.015
POS_TOL_WINDUP = 0.020
POS_TOL_RELEASE = 0.040
POS_TOL_HOME = 0.020

ORI_TOL_READY = 0.08
ORI_TOL_WINDUP = 0.10
ORI_TOL_RELEASE = 0.15
ORI_TOL_HOME = 0.10

# phase durations
WINDUP_SETTLE_TIME = 0.40
WINDUP_TIMEOUT = 1.50
THROW_TIMEOUT = 0.80
FOLLOW_DURATION = 0.25
RETURN_HOME_TIMEOUT = 5.00



# redis client
redis_client = redis.Redis()

# check that the config file is correct
config_file_name = redis_client.get(redis_keys.config_file_name).decode("utf-8")
if config_file_name != config_file_for_this_example:
    print("This example is meant to be used with the config file: ", config_file_for_this_example)
    exit(0)

# set the correct active controller
while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_to_use:
	redis_client.set(redis_keys.active_controller, controller_to_use)

# set the initial goal position and orientation
redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(init_goal_pos.tolist()))
# redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(init_goal_ori.tolist()))

# loop at 100 Hz
loop_time = 0.0
dt = 0.01
internal_step = 0

# MAIN LOOP STARTS HERE!!!
state = State.INIT

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9
state_start_time = init_time
release_has_triggered = False

try:
  while True:
    loop_time += dt
    time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
    
    # read robot state
    current_position = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_position)))
    current_orientation = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation)))



    # state machine
    if state == State.INIT:
      set_cartesian_goal(redis_client, ready_pos, ready_ori)
      if reached_pos(current_position, ready_pos, current_orientation, ready_ori, POS_TOL_READY, ORI_TOL_READY):
        state = State.READY
        print("[INIT] Arrived at READY state.")

    elif state == State.READY:
        set_cartesian_goal(redis_client, windup_pos, windup_ori)
        if reached_pos(current_position, windup_pos, current_orientation, windup_ori, POS_TOL_WINDUP, ORI_TOL_WINDUP):
            state = State.WINDUP
            state_start_time = time.perf_counter_ns() * 1e-9
            print("[READY] Arrived at WINDUP state.")
    
    elif state == State.WINDUP:
        set_cartesian_goal(redis_client, release_pos, release_ori)
        if reached_pos(current_position, release_pos, current_orientation, release_ori, POS_TOL_RELEASE, ORI_TOL_RELEASE):
            if time.perf_counter_ns() * 1e-9 - state_start_time > WINDUP_SETTLE_TIME:
                state = State.THROW
                state_start_time = time.perf_counter_ns() * 1e-9
                print("[WINDUP] Arrived at THROW state.")
        elif time.perf_counter_ns() * 1e-9 - state_start_time > WINDUP_TIMEOUT:
            print("[WINDUP] Timeout reached. Moving to THROW state.")
            state = State.THROW
            state_start_time = time.perf_counter_ns() * 1e-9

    elif state == State.THROW:
        set_cartesian_goal(redis_client, follow_pos, follow_ori)
        if not release_has_triggered and reached_pos(current_position, follow_pos, current_orientation, follow_ori, POS_TOL_FOLLOW, ORI_TOL_FOLLOW):
            print("[THROW] Release triggered.")
            release_has_triggered = True
            state_start_time = time.perf_counter_ns() * 1e-9
        elif release_has_triggered and time.perf_counter_ns() * 1e-9 - state_start_time > FOLLOW_DURATION:
            state = State.RETURN
            state_start_time = time.perf_counter_ns() * 1e-9
            print("[THROW] Moving to RETURN state.")
        elif time.perf_counter_ns() * 1e-9 - state_start_time > THROW_TIMEOUT:
            print("[THROW] Timeout reached. Moving to RETURN state.")
            state = State.RETURN
            state_start_time = time.perf_counter_ns() * 1e-9

    elif state == State.RETURN:
        set_cartesian_goal(redis_client, home_pos, home_ori)
        if reached_pos(current_position, home_pos, current_orientation, home_ori, POS_TOL_HOME, ORI_TOL_HOME):
            state = State.HOLD
            print("[RETURN] Arrived at HOLD state.")
        elif time.perf_counter_ns() * 1e-9 - state_start_time > RETURN_HOME_TIMEOUT:
            print("[RETURN] Timeout reached. Moving to HOLD state.")
            state = State.HOLD
except KeyboardInterrupt:
  print("Keyboard interrupt")
  pass
except Exception as e:
  print(e)
  pass