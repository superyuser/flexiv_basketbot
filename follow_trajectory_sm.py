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
  theta = deg * DEG_TO_RAD
  return np.array([[math.cos(theta), 0, -math.sin(theta)],
                   [0, 1, 0],
                   [math.sin(theta), 0, math.cos(theta)]])


def rot_x(deg):
  theta = deg * DEG_TO_RAD
  return np.array([[1, 0, 0],
                   [0, math.cos(theta), -math.sin(theta)],
                   [0, math.sin(theta), math.cos(theta)]])


def get_pos_error(curr_pos, goal_pos):
  return np.linalg.norm(goal_pos - curr_pos)


def get_ori_error(curr_ori, goal_ori):
  return np.linalg.norm(goal_ori - curr_ori)


def reached_pose(curr_pos, goal_pos, curr_ori, goal_ori, pos_tol, ori_tol):
  is_pos_reached = get_pos_error(curr_pos, goal_pos) < pos_tol
  is_ori_reached = get_ori_error(curr_ori, goal_ori) < ori_tol
  return is_pos_reached and is_ori_reached


def set_cartesian_goal(redis_client, goal_pos, goal_ori):
  redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(goal_pos.tolist()))
  redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(goal_ori.tolist()))


def read_cartesian_state(redis_client):
  pos_raw = redis_client.get(redis_keys.cartesian_task_current_position)
  ori_raw = redis_client.get(redis_keys.cartesian_task_current_orientation)

  if pos_raw is None or ori_raw is None:
    raise RuntimeError("Could not read current Cartesian state from Redis.")

  current_position = np.array(json.loads(pos_raw))
  current_orientation = np.array(json.loads(ori_raw))

  return current_position, current_orientation


# VALUE DECLARATIONS
home_pos = np.array([0.45, 0.00, 0.45])
ready_pos = np.array([0.50, 0.00, 0.48])
windup_pos = np.array([0.30, 0.00, 0.36])
release_pos = np.array([0.72, 0.00, 0.64])
follow_pos = np.array([0.82, 0.00, 0.66])

base_ori = np.array([[1.0, 0, 0],
                     [0, -1.0, 0],
                     [0, 0, -1.0]])

ready_ori = base_ori @ rot_y(20.0).T
windup_ori = base_ori @ rot_y(25.0).T
release_ori = base_ori @ rot_y(35.0).T
follow_ori = release_ori
home_ori = ready_ori

POS_TOL_READY = 0.015
POS_TOL_WINDUP = 0.020
POS_TOL_RELEASE = 0.040
POS_TOL_FOLLOW = 0.050
POS_TOL_HOME = 0.020

ORI_TOL_READY = 0.08
ORI_TOL_WINDUP = 0.10
ORI_TOL_RELEASE = 0.15
ORI_TOL_FOLLOW = 0.15
ORI_TOL_HOME = 0.10

INIT_TIMEOUT = 4.00
READY_SETTLE_TIME = 0.50
WINDUP_SETTLE_TIME = 0.40
WINDUP_TIMEOUT = 1.50
THROW_TIMEOUT = 0.80
FOLLOW_DURATION = 0.25
RETURN_HOME_TIMEOUT = 5.00


# REDIS SETUP
redis_client = redis.Redis()

config_raw = redis_client.get(redis_keys.config_file_name)
if config_raw is None:
  raise RuntimeError("Could not read config file name from Redis.")

config_file_name = config_raw.decode("utf-8")

if config_file_name != config_file_for_this_example:
  print("This example is meant to be used with the config file:", config_file_for_this_example)
  print("Current config file is:", config_file_name)
  exit(0)

while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_to_use:
  redis_client.set(redis_keys.active_controller, controller_to_use)

print("Using controller:", controller_to_use)

set_cartesian_goal(redis_client, ready_pos, ready_ori)


# MAIN LOOP STARTS HERE!!!
loop_time = 0.0
dt = 0.01

state = State.INIT
print("Starting in INIT state.")

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9
state_start_time = init_time

release_has_triggered = False

set_cartesian_goal(redis_client, ready_pos, ready_ori)

try:
  while True:
    loop_time += dt

    now = time.perf_counter_ns() * 1e-9
    elapsed = now - init_time
    time.sleep(max(0.0, loop_time - elapsed))

    now = time.perf_counter_ns() * 1e-9
    state_elapsed = now - state_start_time

    current_position, current_orientation = read_cartesian_state(redis_client)

    if state == State.INIT:
        pass   

    #   if reached_pose(current_position, ready_pos, current_orientation, ready_ori, POS_TOL_READY, ORI_TOL_READY):
    #     state = State.READY
    #     state_start_time = now
    #     print("[INIT] Arrived at READY state.")

    #   elif state_elapsed > INIT_TIMEOUT:
    #     pos_close = get_pos_error(current_position, ready_pos) < 0.035

    #     if pos_close:
    #       state = State.READY
    #       state_start_time = now
    #       print("[INIT] Timeout fallback. Moving to READY state.")
    #     else:
    #       print("[INIT] Timeout reached. Holding READY goal.")
    #       state_start_time = now

    elif state == State.READY:
      set_cartesian_goal(redis_client, ready_pos, ready_ori)

      if state_elapsed > READY_SETTLE_TIME:
        state = State.WINDUP
        state_start_time = now
        print("[READY] Moving to WINDUP state.")

    elif state == State.WINDUP:
      set_cartesian_goal(redis_client, windup_pos, windup_ori)

      if reached_pose(current_position, windup_pos, current_orientation, windup_ori, POS_TOL_WINDUP, ORI_TOL_WINDUP):
        if state_elapsed > WINDUP_SETTLE_TIME:
          state = State.THROW
          state_start_time = now
          release_has_triggered = False
          print("[WINDUP] Moving to THROW state.")

      elif state_elapsed > WINDUP_TIMEOUT:
        state = State.THROW
        state_start_time = now
        release_has_triggered = False
        print("[WINDUP] Timeout reached. Moving to THROW state.")

    elif state == State.THROW:
      set_cartesian_goal(redis_client, release_pos, release_ori)

      release_error = get_pos_error(current_position, release_pos)

      if not release_has_triggered and (release_error < POS_TOL_RELEASE or state_elapsed > THROW_TIMEOUT):
        print("[THROW] Release triggered.")
        release_has_triggered = True
        state = State.FOLLOW
        state_start_time = now
        print("[THROW] Moving to FOLLOW state.")

    elif state == State.FOLLOW:
      set_cartesian_goal(redis_client, follow_pos, follow_ori)

      if state_elapsed > FOLLOW_DURATION:
        state = State.RETURN
        state_start_time = now
        print("[FOLLOW] Moving to RETURN state.")

    elif state == State.RETURN:
      set_cartesian_goal(redis_client, home_pos, home_ori)

      if reached_pose(current_position, home_pos, current_orientation, home_ori, POS_TOL_HOME, ORI_TOL_HOME):
        state = State.HOLD
        state_start_time = now
        print("[RETURN] Arrived at HOLD state.")

      elif state_elapsed > RETURN_HOME_TIMEOUT:
        state = State.HOLD
        state_start_time = now
        print("[RETURN] Timeout reached. Moving to HOLD state.")

    elif state == State.HOLD:
      set_cartesian_goal(redis_client, home_pos, home_ori)

except KeyboardInterrupt:
  print("Keyboard interrupt")
  pass

except Exception as e:
  print(e)
  pass