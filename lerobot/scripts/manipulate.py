from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

import torch 
import os
import platform

record_time_s = 30
fps = 60

states = []
actions = []

leader_port = "/dev/ttyACM0"
follower_port = "/dev/ttyACM1"

leader_config = DynamixelMotorsBusConfig(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_config = DynamixelMotorsBusConfig(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)
# leader_arm = DynamixelMotorsBus(leader_config)
# follower_arm = DynamixelMotorsBus(follower_config)

# robot = ManipulatorRobot(
#     leader_arms={"main": leader_arm},
#     follower_arms={"main": follower_arm},
#     calibration_dir=".cache/calibration/koch",
#     cameras={
#         "laptop": OpenCVCameraConfig(2, fps=30, width=640, height=480),
#         "phone": OpenCVCameraConfig(4, fps=30, width=640, height=480),
#     },
# )
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
robot_config = KochRobotConfig(
    leader_arms={"main": leader_config},
    follower_arms={"main": follower_config},
    calibration_dir=".cache/calibration/koch",
    cameras={        
        "laptop": OpenCVCameraConfig(4, fps=30, width=640, height=480),
        "phone": OpenCVCameraConfig(2, fps=30, width=640, height=480),},
)
robot = ManipulatorRobot(robot_config)
robot.connect()

# def say(text, blocking=False):
#     # Check if mac, linux, or windows.
#     if platform.system() == "Darwin":
#         cmd = f'say "{text}"'
#     elif platform.system() == "Linux":
#         cmd = f'spd-say "{text}"'
#     elif platform.system() == "Windows":
#         cmd = (
#             'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
#             f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
#         )

#     if not blocking and platform.system() in ["Darwin", "Linux"]:
#         # TODO(rcadene): Make it work for Windows
#         # Use the ampersand to run command in the background
#         cmd += " &"

#     os.system(cmd)


inference_time_s = 120
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"
ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
# ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
# ckpt_path = "/Users/helper2424/Documents/lerobot/outputs/koch_move_obj_static_cameras/model.safetensors"
#policy = ACTPolicy.from_pretrained("helper2424/koch_move_obj_static_cameras", force_download=True)
policy = ACTPolicy.from_pretrained(ckpt_path,local_files_only=True)
policy.to(device)

# say("I am going to collect objects")
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

# say("I have finished collecting objects")
