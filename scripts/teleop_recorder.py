import argparse
import os
import csv
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim import SimulationApp

# =================================================================================
# Simulation Application Setup
# =================================================================================
CONFIG = {
    "headless": False, 
    "enable_cameras": True,
}
simulation_app = SimulationApp(CONFIG)

# Isaac Sim / Omniverse core imports must happen after SimulationApp is initialized
import carb
import carb.settings
import omni
import omni.kit.viewport.utility 
from pxr import UsdGeom, Gf, Usd, Sdf
import omni.usd

from omni.isaac.core.utils.stage import open_stage, create_new_stage
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction

# Robot & Controller
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka import Franka

# Sensors
import omni.replicator.core as rep

# =================================================================================
# Configuration Constants
# =================================================================================
CAMERA_CONFIG = {
    "wrist": "/World/Franka/panda_hand/Camera_Wrist",
    "perspective": "/World/Camera_Perspective",
    "drone": "/World/quadcopter/chassis/Camera",
    "side": "/World/Camera_Side"
}

# Standard Gripper Parameters
FRANKA_OPEN_POS = 0.04 
FRANKA_CLOSE_POS = 0.00 

# =================================================================================
# Viewport Setup
# =================================================================================
def setup_custom_viewports():
    """Configures multiple viewports for different camera perspectives."""
    print("[Setup] Configuring multi-viewports...")
    SMALL_WIDTH = 400
    SMALL_HEIGHT = 300
    
    try:
        vp_quad = omni.kit.viewport.utility.create_viewport_window(
            "Quadcopter View", width=SMALL_WIDTH, height=SMALL_HEIGHT
        )
        if vp_quad: 
            vp_quad.viewport_api.camera_path = "/World/Quadcopter/chassis/Camera"
        
        vp_wrist = omni.kit.viewport.utility.create_viewport_window(
            "Wrist View", width=SMALL_WIDTH, height=SMALL_HEIGHT
        )
        if vp_wrist: 
            vp_wrist.viewport_api.camera_path = "/World/Franka/panda_hand/Camera_Wrist"
            
    except Exception as e:
        print(f"[Warning] Viewport creation failed: {e}")

    settings = carb.settings.get_settings()
    settings.set_bool("/app/viewport/displayOptions/guides", True)
    settings.set_bool("/persistent/app/viewport/displayOptions/guides", True)

# =================================================================================
# Data Recording Module
# =================================================================================
class DataRecorder:
    """Handles recording of robot states, actions, and camera feeds to disk."""
    def __init__(self, root_dir, franka_prim, base_prim, ee_prim):
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        self.franka = franka_prim
        self.base_prim = base_prim
        self.ee_prim = ee_prim
        
        self.current_episode_idx = self._get_next_episode_idx()
        
        self.rep_data = {}
        self.render_products = []
        self.state_keys = ["base_pos", "base_rot", "joint_pos", "ee_pos", "ee_rot"]
        self.reset_buffers()

    def _get_next_episode_idx(self):
        existing = [d for d in os.listdir(self.root_dir) if d.startswith("episode_")]
        if not existing: return 0
        indices = []
        for d in existing:
            try: indices.append(int(d.split("_")[1]))
            except ValueError: pass
        return max(indices) + 1 if indices else 0

    def setup_sensors(self):
        self.render_products = []
        self.rep_data = {}
        for name, path in CAMERA_CONFIG.items():
            try:
                rp = rep.create.render_product(path, resolution=(640, 480))
                self.render_products.append(rp)
                rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb.attach(rp)
                self.rep_data[f"{name}_rgb"] = rgb
            except Exception:
                pass
        self.reset_buffers()

    def reset_buffers(self):
        self.buffer_actions = []
        self.buffer_sensors = {key: [] for key in self.rep_data.keys()}
        self.prev_state = {k: None for k in self.state_keys}

    def record_step(self, gripper_closed, dt=1.0/60.0):
        try:
            b_pos, b_quat = self.base_prim.get_world_poses()
            b_pos = b_pos[0]
            b_euler = R.from_quat([b_quat[0][1], b_quat[0][2], b_quat[0][3], b_quat[0][0]]).as_euler('xyz')
            j_pos = self.franka.get_joint_positions()
            j_vel = self.franka.get_joint_velocities()
            ee_pos, ee_quat = self.ee_prim.get_world_poses()
            ee_pos = ee_pos[0]
            ee_euler = R.from_quat([ee_quat[0][1], ee_quat[0][2], ee_quat[0][3], ee_quat[0][0]]).as_euler('xyz')
        except Exception:
            return 

        if self.prev_state["base_pos"] is None:
            delta_base = np.zeros(4)
            delta_joints = np.zeros_like(j_pos)
            delta_ee_pos = np.zeros(3)
            delta_ee_rot = np.zeros(3)
        else:
            delta_base = np.concatenate([
                b_pos - self.prev_state["base_pos"],
                [b_euler[2] - self.prev_state["base_rot"][2]]
            ])
            delta_joints = j_pos - self.prev_state["joint_pos"]
            delta_ee_pos = ee_pos - self.prev_state["ee_pos"]
            delta_ee_rot = ee_euler - self.prev_state["ee_rot"]

        self.prev_state["base_pos"] = b_pos
        self.prev_state["base_rot"] = b_euler
        self.prev_state["joint_pos"] = j_pos
        self.prev_state["ee_pos"] = ee_pos
        self.prev_state["ee_rot"] = ee_euler

        base_lin_vel = delta_base[:3] / dt
        base_ang_vel = delta_base[3] / dt
        ee_lin_vel = delta_ee_pos / dt
        ee_ang_vel = delta_ee_rot / dt

        action_row = {}
        action_row.update({f"drone_d{k}": v for k, v in zip(['x','y','z','w'], delta_base)})
        action_row.update({f"drone_v{k}": v for k, v in zip(['x','y','z','w'], [*base_lin_vel, base_ang_vel])})
        for i in range(len(j_pos)):
            action_row[f"joint_{i}_pos"] = j_pos[i]
            action_row[f"joint_{i}_vel"] = j_vel[i]
            action_row[f"joint_{i}_delta"] = delta_joints[i]
        action_row.update({f"ee_d{k}": v for k, v in zip(['x','y','z'], delta_ee_pos)})
        action_row.update({f"ee_drot_{k}": v for k, v in zip(['x','y','z'], delta_ee_rot)})
        action_row.update({f"ee_v{k}": v for k, v in zip(['x','y','z'], ee_lin_vel)})
        action_row.update({f"ee_vrot_{k}": v for k, v in zip(['x','y','z'], ee_ang_vel)})
        action_row["gripper_closed"] = 1 if gripper_closed else 0
        
        self.buffer_actions.append(action_row)

        for name, anno in self.rep_data.items():
            try:
                data = anno.get_data()
                if data is not None:
                    if "rgb" in name:
                        self.buffer_sensors[name].append(data[:, :, :3].copy())
            except Exception: pass

    def flush_to_disk(self):
        if not self.buffer_actions:
            print("[Warning] No data to save.")
            return

        ep_dir = os.path.join(self.root_dir, f"episode_{self.current_episode_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        print(f"[Storage] Saving Episode {self.current_episode_idx} to {ep_dir} ...")

        csv_path = os.path.join(ep_dir, "actions.csv")
        keys = self.buffer_actions[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.buffer_actions)

        for name, data_list in self.buffer_sensors.items():
            if len(data_list) == 0: continue
            sensor_dir = os.path.join(ep_dir, name)
            os.makedirs(sensor_dir, exist_ok=True)
            if "rgb" in name:
                video_path = os.path.join(sensor_dir, f"{name}.mp4")
                if len(data_list) > 0:
                    h, w, _ = data_list[0].shape
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (w, h))
                    for frame in data_list: out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
        
        print(f"[Storage] Episode {self.current_episode_idx} saved successfully.")
        self.current_episode_idx += 1
        self.reset_buffers()

# =================================================================================
# Keyboard Teleoperation Controller
# =================================================================================
class KeyboardController:
    """Handles keyboard inputs for teleoperation."""
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_callback)
        self.keys = {k: False for k in ["W","S","A","D","Q","E","R","F","UP","DOWN","LEFT","RIGHT","M","N","NUMPAD_4","NUMPAD_1","Z","X","SPACE","ENTER","P","H","J","K","U","T","G"]}
        self.gripper_closed = False     
        self.last_space_state = False   

    def _sub_callback(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._update_key(event.input, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._update_key(event.input, False)

    def _update_key(self, input_item, is_pressed):
        key_name = input_item.name
        if key_name in self.keys: self.keys[key_name] = is_pressed
        map_dict = {"UP":"UP", "DOWN":"DOWN", "LEFT":"LEFT", "RIGHT":"RIGHT", "SPACE":"SPACE", "NUMPAD_4":"NUMPAD_4", "NUMPAD_1":"NUMPAD_1", "ENTER":"ENTER", "P":"P", "T":"T", "G":"G"}
        if key_name in map_dict: self.keys[map_dict[key_name]] = is_pressed

    def reset(self):
        self.gripper_closed = False
        self.last_space_state = False
        print("[Controller] KeyboardController reset.")

    def get_deltas(self, speed_base, speed_ee, speed_rot):
        d_base_local = np.array([0.0, 0.0, 0.0])
        if self.keys["UP"]: d_base_local[0] += speed_base     
        if self.keys["DOWN"]: d_base_local[0] -= speed_base   
        if self.keys["LEFT"]: d_base_local[1] -= speed_base   
        if self.keys["RIGHT"]: d_base_local[1] += speed_base  
        
        if self.keys["NUMPAD_4"]: d_base_local[2] += speed_base 
        if self.keys["NUMPAD_1"]: d_base_local[2] -= speed_base 

        d_base_rot = 0.0
        if self.keys["Z"]: d_base_rot -= speed_rot 
        if self.keys["X"]: d_base_rot += speed_rot 

        d_ee = np.array([0.0, 0.0, 0.0])
        if self.keys["W"]: d_ee[0] += speed_ee
        if self.keys["S"]: d_ee[0] -= speed_ee
        if self.keys["A"]: d_ee[1] -= speed_ee 
        if self.keys["D"]: d_ee[1] += speed_ee 
        if self.keys["R"]: d_ee[2] += speed_ee 
        if self.keys["F"]: d_ee[2] -= speed_ee 

        d_ee_yaw = 0.0
        if self.keys["Q"]: d_ee_yaw -= speed_rot 
        if self.keys["E"]: d_ee_yaw += speed_rot 

        orientation_cmd = None
        if self.keys["J"]: orientation_cmd = "DOWN" 
        if self.keys["U"]: orientation_cmd = "FORWARD"
        if self.keys["H"]: orientation_cmd = "RIGHT"
        if self.keys["K"]: orientation_cmd = "LEFT" 
        
        is_tilting_up = self.keys["T"]
        is_tilting_down = self.keys["G"]

        if self.keys["SPACE"] and not self.last_space_state:
            self.gripper_closed = not self.gripper_closed
        self.last_space_state = self.keys["SPACE"]

        return d_base_local, d_base_rot, d_ee, d_ee_yaw, self.gripper_closed, self.keys["ENTER"], self.keys["P"], orientation_cmd, is_tilting_up, is_tilting_down

def rotate_vector_2d(vector, angle_rad):
    x, y = vector[0], vector[1]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([x*c - y*s, x*s + y*c, vector[2]])

def transform_local_to_world(local_delta, current_yaw):
    c, s = np.cos(current_yaw), np.sin(current_yaw)
    return np.array([local_delta[0]*c - local_delta[1]*s, local_delta[0]*s + local_delta[1]*c, local_delta[2]])

# =================================================================================
# Main Execution
# =================================================================================
if __name__ == "__main__":
    # Setup Argument Parser for Open-Source flexibility
    parser = argparse.ArgumentParser(description="Isaac Sim Dataset Recording Script")
    parser.add_argument("--usd_path", type=str, default="environments/manipulation.usd", help="Path to the USD scene file")
    parser.add_argument("--dataset_root", type=str, default="data/dataset_raw/object/task_name", help="Root directory to save the recorded dataset")
    args = parser.parse_args()

    # Verify or create necessary paths
    if not os.path.exists(args.usd_path) and not args.usd_path.startswith("omniverse://"):
        print(f"[Warning] USD path '{args.usd_path}' does not exist. Please check your configuration.")
    
    os.makedirs(args.dataset_root, exist_ok=True)

    print(f"[Init] Loading Stage from: {args.usd_path}")
    create_new_stage()      
    open_stage(args.usd_path)    
    world = World()         

    # -----------------------------------------------------------------------------
    # Robot Initialization
    # -----------------------------------------------------------------------------
    franka_prim_path = "/World/Franka"
    franka = world.scene.add(Franka(prim_path=franka_prim_path, name="franka"))
    franka_parent = XFormPrim(franka_prim_path)
    ee_prim = XFormPrim("/World/Franka/panda_hand")

    world.reset()
    setup_custom_viewports()

    recorder = DataRecorder(args.dataset_root, franka, franka_parent, ee_prim)
    recorder.setup_sensors()

    starting_joint_positions = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.04, 0.04])
    all_dof_names = franka.dof_names
    target_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
        "panda_joint5", "panda_joint6", "panda_joint7", 
        "panda_finger_joint1", "panda_finger_joint2"
    ]
    sorted_indices = []
    for target in target_joint_names:
        for i, name in enumerate(all_dof_names):
            if target == name or target in name: 
                sorted_indices.append(i)
                break
    SORTED_INDICES_ARR = np.array(sorted_indices, dtype=np.int32)

    if len(SORTED_INDICES_ARR) == 9:
        franka.set_joint_positions(starting_joint_positions, joint_indices=SORTED_INDICES_ARR)

    for i in range(50): world.step()

    kb_controller = KeyboardController()
    rmp_controller = RMPFlowController(name="rmp_controller", robot_articulation=franka, physics_dt=1.0/60.0)

    initial_base_pos, initial_base_quat = franka_parent.get_world_poses()
    initial_base_pos = np.array(initial_base_pos[0], dtype=np.float64)
    initial_base_quat = np.array(initial_base_quat[0], dtype=np.float64)
    initial_r_base = R.from_quat([initial_base_quat[1], initial_base_quat[2], initial_base_quat[3], initial_base_quat[0]])
    initial_base_euler = initial_r_base.as_euler('xyz')

    current_base_pos = initial_base_pos.copy()
    current_base_quat = initial_base_quat.copy()
    current_base_euler = initial_base_euler.copy()

    ee_pos_start, _ = ee_prim.get_world_poses()
    target_ee_pos = np.array(ee_pos_start[0])

    current_ee_euler_base = np.array([0.0, np.pi, 0.0]) 
    current_ee_yaw_local = 0.0
    current_ee_pitch_local = 0.0

    def full_reset_scene():
        """Fully resets the physics scene, robot poses, and buffers."""
        global current_base_pos, current_base_quat, current_base_euler, target_ee_pos
        global current_ee_euler_base, current_ee_yaw_local, current_ee_pitch_local
        
        world.reset()
        recorder.reset_buffers()
        franka.initialize() 
        if len(SORTED_INDICES_ARR) == 9:
            franka.set_joint_positions(starting_joint_positions, joint_indices=SORTED_INDICES_ARR)
        
        current_base_pos = initial_base_pos.copy()
        current_base_euler = initial_base_euler.copy()
        r_new = R.from_euler('xyz', current_base_euler)
        q_xyzw = r_new.as_quat()
        current_base_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        
        franka_parent.set_world_poses(positions=np.array([current_base_pos]), orientations=np.array([current_base_quat]))
        rmp_controller._motion_policy.set_robot_base_pose(robot_position=current_base_pos, robot_orientation=current_base_quat)
        
        kb_controller.reset()
        world.step(render=False) 
        
        ee_pos_reset, _ = ee_prim.get_world_poses()
        target_ee_pos = np.array(ee_pos_reset[0])
        
        current_ee_euler_base = np.array([0.0, np.pi, 0.0]) 
        current_ee_yaw_local = 0.0
        current_ee_pitch_local = 0.0

        print("[System] Scene has been completely reset.")
        for _ in range(10): world.step()

    print("============================================")
    print("🎮 Teleoperation Recording Mode started")
    print(" [J] Down  | [U] Forward")
    print(" [H] Right | [K] Left")
    print(" [Q/E] Rotate | [T/G] Tilt Up/Down")
    print(" [SPACE] Toggle Gripper Open/Close")
    print(" [ENTER] Save Episode | [P] Discard Episode")
    print("============================================")

    # -----------------------------------------------------------------------------
    # Main Simulation Loop
    # -----------------------------------------------------------------------------
    while simulation_app.is_running():
        
        d_base_local, d_base_rot, d_ee, d_ee_yaw, is_gripper_closed, save_trigger, discard_trigger, orientation_cmd, is_tilting_up, is_tilting_down = kb_controller.get_deltas(
            speed_base=0.002, speed_ee=0.002, speed_rot=0.005   
        )

        if save_trigger:
            recorder.flush_to_disk()
            full_reset_scene()
            continue

        if discard_trigger:
            print("[Action] Discarding current episode...")
            full_reset_scene()
            continue

        if orientation_cmd == "DOWN":
            current_ee_euler_base = np.array([0.0, np.pi, 0.0]) 
            current_ee_yaw_local = 0.0
            current_ee_pitch_local = 0.0
        elif orientation_cmd == "FORWARD":
            current_ee_euler_base = np.array([0.0, np.pi/2, 0.0])
            current_ee_yaw_local = 0.0
            current_ee_pitch_local = 0.0
        elif orientation_cmd == "RIGHT":
            current_ee_euler_base = np.array([np.pi/2, np.pi, 0.0])
            current_ee_yaw_local = 0.0
            current_ee_pitch_local = 0.0
        elif orientation_cmd == "LEFT":
            current_ee_euler_base = np.array([-np.pi/2, np.pi, 0.0])
            current_ee_yaw_local = 0.0
            current_ee_pitch_local = 0.0

        if is_tilting_up: current_ee_pitch_local -= 0.02 
        if is_tilting_down: current_ee_pitch_local += 0.02 

        base_changed = False 
        if abs(d_base_rot) > 0:
            current_base_euler[2] += d_base_rot
            current_ee_yaw_local += d_base_rot 
            base_changed = True

        if np.linalg.norm(d_base_local) > 0:
            current_yaw = current_base_euler[2]
            d_base_world = transform_local_to_world(d_base_local, current_yaw)
            current_base_pos += d_base_world
            target_ee_pos += d_base_world
            base_changed = True

        if abs(d_base_rot) > 0 and np.linalg.norm(d_base_local) == 0:
            relative_vector = target_ee_pos - current_base_pos
            rotated_vector = rotate_vector_2d(relative_vector, d_base_rot)
            target_ee_pos = current_base_pos + rotated_vector

        if base_changed:
            r_new = R.from_euler('xyz', current_base_euler)
            q_xyzw = r_new.as_quat()
            current_base_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            franka_parent.set_world_poses(positions=np.array([current_base_pos]), orientations=np.array([current_base_quat]))
            rmp_controller._motion_policy.set_robot_base_pose(robot_position=current_base_pos, robot_orientation=current_base_quat)

        target_ee_pos += d_ee
        current_ee_yaw_local += d_ee_yaw
        
        r_base_preset = R.from_euler('xyz', current_ee_euler_base)
        r_local_yaw = R.from_euler('z', current_ee_yaw_local)
        r_local_pitch = R.from_euler('y', current_ee_pitch_local)
        r_final = r_base_preset * r_local_yaw * r_local_pitch
        q_xyzw = r_final.as_quat()
        target_ee_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # Execute RMPFlow controller
        actions = rmp_controller.forward(target_end_effector_position=target_ee_pos, target_end_effector_orientation=target_ee_quat)
        franka.apply_action(actions)
        
        # Standard physics-based Gripper Control
        if is_gripper_closed:
            franka.gripper.apply_action(ArticulationAction(joint_positions=np.array([FRANKA_CLOSE_POS, FRANKA_CLOSE_POS]))) 
        else:
            franka.gripper.apply_action(ArticulationAction(joint_positions=np.array([FRANKA_OPEN_POS, FRANKA_OPEN_POS]))) 

        world.step(render=True)
        recorder.record_step(is_gripper_closed)

    simulation_app.close()
