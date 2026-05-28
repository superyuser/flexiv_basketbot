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

rot_y_5_deg = rot_y(5.0)
rot_y_neg_5_deg = rot_y(-5.0)
rot_x_5_deg = rot_x(5.0)
rot_x_neg_5_deg = rot_x(-5.0)


# init_goal_pos = np.array([0.55, 0.0, 0.50])
# init_goal_ori = np.dot(np.array([[1.0,0,0],[0,-1.0,0],[0,0,-1.0]]),rot_y_15_deg.T)

# left_goal_pos = init_goal_pos - np.array([0, 0.2, 0])
# right_goal_pos = init_goal_pos + np.array([0, 0.2, 0])
# left_goal_ori = np.dot(init_goal_ori, rot_x_30_deg.T)
# right_goal_ori = np.dot(init_goal_ori, rot_x_30_deg)

# left_goal_ori = left_goal_ori @ np.array([[0,-1,0],[1,0,0],[0,0,1]]).T
# right_goal_ori = right_goal_ori @ np.array([[0,-1,0],[1,0,0],[0,0,1]]).T

# INIT = auto()
# READY = auto()
# WINDUP = auto()
# THROW = auto()
# FOLLOW = auto()
# RETURN = auto()
# HOLD = auto()

# INIT -> READY -> WINDUP -> THROW -> FOLLOW -> RETURN -> HOLD
# where the INIT "homed" position should be! (since robot's current pos and ori aren't "homed")
# robot is to first reach this pos + ori whenever started, before following trajectory
init_goal_pos = np.array([0.55, 0.0, 0.50]) # this is like the resting position we want to start the state machine at every time; once reached, transition to READY
init_goal_ori = np.dot(np.array([[1.0,0,0],[0,-1.0,0],[0,0,-1.0]]), rot_y_5_deg.T)

ready_goal_pos = init_goal_pos - np.array([0, 0.2, 0])
windup_goal_pos = init_goal_pos + np.array([0, 0.2, 0])
throw_goal_pos = init_goal_pos - np.array([0, 0.2, 0])
follow_goal_pos = init_goal_pos + np.array([0, 0.2, 0])
return_goal_pos = init_goal_pos - np.array([0, 0.2, 0])
hold_goal_pos = return_goal_pos

ready_goal_ori = np.dot(init_goal_ori, rot_x_5_deg.T)
windup_goal_ori = np.dot(init_goal_ori, rot_x_5_deg)
throw_goal_ori = np.dot(init_goal_ori, rot_x_5_deg.T)
follow_goal_ori = np.dot(init_goal_ori, rot_x_5_deg)
return_goal_ori = np.dot(init_goal_ori, rot_x_5_deg.T)
hold_goal_ori = return_goal_ori


goal_pos = init_goal_pos
goal_ori = init_goal_ori

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
state = State.INIT

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

try:
  while True:
    loop_time += dt
    time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
    
    # read robot state
    current_position = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_position)))
    current_orientation = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation)))

    # state machine
    if state == State.INIT: # next up: READY
      # monitor error
      pos_error = np.linalg.norm(init_goal_pos - current_position)
      ori_error = np.linalg.norm(init_goal_ori - current_orientation)
      if pos_error < 5e-2:
        redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(ready_goal_pos.tolist()))
        # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(ready_goal_ori.tolist()))
        state = State.READY
        print("Going to READY")

    elif state == State.READY:  # next up: WINDUP
      # monitor error
      pos_error = np.linalg.norm(ready_goal_pos - current_position)
      ori_error = np.linalg.norm(ready_goal_ori - current_orientation)
      if pos_error < 5e-2:
        redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(windup_goal_pos.tolist()))
        # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(windup_goal_ori.tolist()))
        state = State.WINDUP
        print("Going to WINDUP")

    elif state == State.WINDUP:  # next up: THROW
      # monitor error
      pos_error = np.linalg.norm(windup_goal_pos - current_position)
      ori_error = np.linalg.norm(windup_goal_ori - current_orientation)
      if pos_error < 5e-2:
        redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(throw_goal_pos.tolist()))
        # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(windup_goal_ori.tolist()))
        state = State.THROW
        print("Going to THROW")

    elif state == State.THROW:  # next up: FOLLOW
      # monitor error
      pos_error = np.linalg.norm(throw_goal_pos - current_position)
      ori_error = np.linalg.norm(throw_goal_ori - current_orientation)
      if pos_error < 5e-2:
        redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(follow_goal_pos.tolist()))
        # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(throw_goal_ori.tolist()))
        state = State.FOLLOW
        print("Going to FOLLOW")

    elif state == State.FOLLOW:  # next up: RETURN
      # monitor error
      pos_error = np.linalg.norm(follow_goal_pos - current_position)
      ori_error = np.linalg.norm(follow_goal_ori - current_orientation)
      if pos_error < 5e-2:
        redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(return_goal_pos.tolist()))
        # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(throw_goal_ori.tolist()))
        state = State.RETURN
        print("Going to RETURN")
        
    elif state == State.RETURN:  # next up: HOLD    
        # monitor error
        pos_error = np.linalg.norm(return_goal_pos - current_position)
        ori_error = np.linalg.norm(return_goal_ori - current_orientation)
        if pos_error < 5e-2:
            redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(hold_goal_pos.tolist()))
            # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(throw_goal_ori.tolist()))
            state = State.HOLD
            print("Going to HOLD")
    
    elif state == State.HOLD:  # next up: READY
        time.sleep(15.0)


    # elif state == State.GOING_LEFT:
    #   # monitor error
    #   pos_error = np.linalg.norm(left_goal_pos - current_position)
    #   ori_error = np.linalg.norm(left_goal_ori - current_orientation)
    #   if pos_error < 5e-2:
    #     redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(right_goal_pos.tolist()))
    #     # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(right_goal_ori.tolist()))
    #     state = State.GOING_RIGHT
    #     print("Going Right")

    # elif state == State.GOING_RIGHT:
    #   # monitor error
    #   pos_error = np.linalg.norm(right_goal_pos - current_position)
    #   ori_error = np.linalg.norm(right_goal_ori - current_orientation)
    #   if pos_error < 5e-2:
    #     redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(left_goal_pos.tolist()))
    #     # redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(left_goal_ori.tolist()))
    #     state = State.GOING_LEFT
    #     print("Going Left")

except KeyboardInterrupt:
  print("Keyboard interrupt")
  pass
except Exception as e:
  print(e)
  pass