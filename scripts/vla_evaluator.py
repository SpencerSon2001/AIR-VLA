import argparse
import os
import cv2
import csv
import time
import shutil
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim import SimulationApp

# =================================================================================
# 1. Simulation Application Setup
# =================================================================================
CONFIG = {
    "headless": False,       
    "enable_cameras": True,  
}
simulation_app = SimulationApp(CONFIG)

# ------------------- Core Imports (Must follow SimulationApp) -------------------
import carb
import omni
import omni.kit.viewport.utility 
from pxr import UsdGeom, Gf, Usd, Sdf, PhysxSchema
import omni.usd

# Communication Libraries
import websocket
import msgpack
import msgpack_numpy
msgpack_numpy.patch() 

# Isaac Sim Core Modules
from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim, SingleRigidPrim
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction 
import omni.replicator.core as rep

# =================================================================================
# 2. Global Parameters & Constants
# =================================================================================

FRANKA_REACH_LIMIT = 0.9 

# Camera path configurations in the USD stage
CAMERA_CONFIG = {
    "cam_high": "/World/quadcopter/chassis/Camera",
    "cam_low": "/World/Camera_Perspective",             
    "cam_left_wrist": "/World/Camera_Side",      
    "cam_right_wrist": "/World/Franka/panda_hand/Camera_Wrist"                    
}

# The main camera used for video recording
VIDEO_SOURCE_CAM = "cam_low"

# List of all interactable objects for physics material application and tracking
INTERACTABLE_OBJECTS = [
    "/World/meat_can"
]

# Standard Gripper Parameters
DEFAULT_CLOSE_WIDTH = 0.02 
FRANKA_MAX_OPEN = 0.08

# =================================================================================
# 3. Task List Definitions
# =================================================================================
TASK_LIST = [
    # --- Manipulation ---
    {"group": "manipulation", "cmd": "place_the_meat_can_on_the_plate", "usd": "manipulation.usd", "type": "place", "target": "/World/meat_can", "container": "/World/red_plate"}
]

# =================================================================================
# 4. Utility Classes
# =================================================================================

class ServerInterface:
    """Handles WebSocket communication with the policy inference server."""
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.connect()
        self.inference_count = 0

    def connect(self):
        try:
            print(f"[INFO] Connecting to Policy Server at {self.url}...")
            self.ws = websocket.create_connection(self.url, timeout=10) 
            print("[INFO] Connection established.")
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.ws = None

    def get_action(self, instruction, images, robot_state):
        if self.ws is None:
            self.connect()
            if self.ws is None: return None
        
        self.inference_count += 1
        
        try:
            imgs_dict = {}
            REQUIRED_CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
            TARGET_H, TARGET_W = 480, 640
            
            for cam_name in REQUIRED_CAMERAS:
                if cam_name in images and images[cam_name] is not None:
                    img = images[cam_name] 
                    img_chw = np.transpose(img, (2, 0, 1)) 
                    img_float = img_chw.astype(np.float32) / 255.0
                    imgs_dict[cam_name] = img_float
                else:
                    imgs_dict[cam_name] = np.zeros((3, TARGET_H, TARGET_W), dtype=np.float32)

            state_arr = np.array(robot_state, dtype=np.float32)
            
            payload = {
                "images": imgs_dict, 
                "state": state_arr, 
                "prompt": instruction
            }
            
        except Exception as e:
            print(f"[ERROR] Preparing payload: {e}")
            return None

        try:
            data = msgpack.packb(payload, use_bin_type=True)
            self.ws.send_binary(data)
            
            result = self.ws.recv()
            if isinstance(result, str): return None
            
            unpacked_data = msgpack.unpackb(result, raw=False)
            
            if isinstance(unpacked_data, dict):
                actions_dict = unpacked_data.get('actions') or unpacked_data.get(b'actions')
                
                if actions_dict:
                    if isinstance(actions_dict, np.ndarray):
                        action_arr = actions_dict
                    elif isinstance(actions_dict, dict):
                        data_bytes = actions_dict.get('data') or actions_dict.get(b'data')
                        dtype_str = actions_dict.get('dtype') or actions_dict.get(b'dtype') or '<f4'
                        shape = actions_dict.get('shape') or actions_dict.get(b'shape')
                        
                        if data_bytes:
                            action_arr = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str))
                            if shape:
                                action_arr = action_arr.reshape(shape)
                    
                    if 'action_arr' in locals():
                        if action_arr.ndim == 1:
                            # Total Action Dim = 12 (4 Drone + 7 Arm + 1 Gripper)
                            ACTION_DIM = 12
                            if action_arr.size % ACTION_DIM == 0:
                                action_arr = action_arr.reshape(-1, ACTION_DIM)
                        
                        chunk_len = min(5, action_arr.shape[0])
                        return action_arr[:chunk_len].tolist()

            # Default fallback action chunk
            return [[0.0]*11 + [0.0]] * 5 
            
        except Exception as e:
            print(f"[ERROR] Inference issue: {e}")
            return None

    def close(self):
        if self.ws: self.ws.close()

class SensorManager:
    """Manages virtual cameras and retrieves image streams from the Isaac Sim scene."""
    def __init__(self):
        self.annotators = {}
        
    def setup(self):
        self.annotators = {}
        for name, path in CAMERA_CONFIG.items():
            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath(path).IsValid(): continue
            try:
                rp = rep.create.render_product(path, resolution=(640, 480)) 
                rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb.attach(rp)
                self.annotators[name] = rgb
            except Exception: pass
            
    def capture(self):
        rep.orchestrator.step()
        images = {}
        for name, anno in self.annotators.items():
            data = anno.get_data()
            if data is not None: images[name] = data[:, :, :3]
        return images

class SceneStateSaver:
    """Saves and restores initial object states to facilitate environment resets."""
    def __init__(self): 
        self.initial_states = {}
        
    def capture_initial_state(self):
        self.initial_states = {}
        stage = omni.usd.get_context().get_stage()
        
        targets = INTERACTABLE_OBJECTS + [
            "/World/cabinet/drawer_top", 
            "/World/cabinet/drawer_bottom", 
            "/World/Cube_01", 
            "/World/Cube_02"
        ]
        for path in targets:
            if stage.GetPrimAtPath(path).IsValid():
                prim = XFormPrim(path)
                pos, quat = prim.get_world_poses()
                try: rigid = SingleRigidPrim(path)
                except: rigid = None
                self.initial_states[path] = {"pos": pos[0], "quat": quat[0], "rigid": rigid}
                
    def reset_objects(self):
        for path, state in self.initial_states.items():
            try:
                XFormPrim(path).set_world_poses(positions=np.array([state["pos"]]), orientations=np.array([state["quat"]]))
                if state["rigid"]: 
                    state["rigid"].set_linear_velocity(np.array([0.,0.,0.]))
                    state["rigid"].set_angular_velocity(np.array([0.,0.,0.]))
            except: pass

# =================================================================================
# 5. Task Evaluator
# =================================================================================
class Evaluator:
    """Calculates metrics dynamically based on robot-object interactions and goal achievement."""
    def __init__(self, scene_saver, ee_prim_path):
        self.saver = scene_saver
        self.ee_prim = XFormPrim(ee_prim_path)
        
        self.metrics = {}
        self.subtasks_done = []
        
        self.involved_objects = set() 
        self.object_tracking = {} 
        self.prev_gripper_closed = False

    def reset_task(self, task):
        self.metrics = {
            "uav_score": 0.0,
            "arm_score": 0.0,
            "safety_score": 0.0,
            "completion_score": 0.0,
            "total_score": 0.0
        }
        
        self.involved_objects = set()
        subtasks = task.get('subtasks', [task])
        self.subtasks_done = [False] * len(subtasks)
        
        for sub in subtasks:
            if 'target' in sub: self.involved_objects.add(sub['target'])
        
        self.object_tracking = {}
        for obj in self.involved_objects:
            self.object_tracking[obj] = {
                'min_uav_dist': float('inf'),
                'acted_in_range': False, 
                'min_arm_dist_on_action': float('inf'), 
                'success': False
            }
        
        self.prev_gripper_closed = False

    def get_uav_pos(self):
        pos, _ = XFormPrim("/World/Franka").get_world_poses()
        return pos[0]

    def check_placement(self, target_path, container_path):
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(target_path).IsValid() or not stage.GetPrimAtPath(container_path).IsValid(): return False
        t_pos, _ = XFormPrim(target_path).get_world_poses()
        c_pos, _ = XFormPrim(container_path).get_world_poses()
        return np.linalg.norm(t_pos[0] - c_pos[0]) < 0.20

    def check_drawer(self, check_obj_path):
        if check_obj_path not in self.saver.initial_states: return False
        curr_pos, _ = XFormPrim(check_obj_path).get_world_poses()
        init_pos = self.saver.initial_states[check_obj_path]["pos"]
        return abs(curr_pos[0][0] - init_pos[0]) > 0.3

    def check_rotation(self, check_obj_path, angle):
        if check_obj_path not in self.saver.initial_states: return False
        _, curr_quat = XFormPrim(check_obj_path).get_world_poses()
        init_quat = self.saver.initial_states[check_obj_path]["quat"]
        r_curr = R.from_quat([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
        r_init = R.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]])
        diff = (r_curr * r_init.inv()).as_euler('xyz', degrees=True)
        return np.max(np.abs(diff)) > angle

    def update(self, task, gripper_closed):
        uav_pos = self.get_uav_pos()
        ee_pos, ee_quat = self.ee_prim.get_world_poses()
        ee_pos = ee_pos[0]
        
        is_action_frame = (gripper_closed != self.prev_gripper_closed)
        self.prev_gripper_closed = gripper_closed

        stage = omni.usd.get_context().get_stage()
        for obj_path in self.involved_objects:
            if not stage.GetPrimAtPath(obj_path).IsValid(): continue
            
            obj_prim = XFormPrim(obj_path)
            o_pos, _ = obj_prim.get_world_poses()
            o_pos = o_pos[0]
            
            dist_uav = np.linalg.norm(uav_pos - o_pos)
            self.object_tracking[obj_path]['min_uav_dist'] = min(self.object_tracking[obj_path]['min_uav_dist'], dist_uav)
            
            if is_action_frame:
                if 0.5 <= dist_uav <= 0.8:
                    self.object_tracking[obj_path]['acted_in_range'] = True
                
                dist_arm = np.linalg.norm(ee_pos - o_pos)
                self.object_tracking[obj_path]['min_arm_dist_on_action'] = min(
                    self.object_tracking[obj_path]['min_arm_dist_on_action'], dist_arm
                )

        subtasks = task.get('subtasks', [task])
        for i, sub in enumerate(subtasks):
            if not self.subtasks_done[i]:
                done = False
                target = sub.get('target')
                
                if sub['type'] == 'place':
                    done = self.check_placement(target, sub['container'])
                elif sub['type'] == 'drawer':
                    done = self.check_drawer(sub['check_obj'])
                elif sub['type'] == 'rotate':
                    done = self.check_rotation(sub['check_obj'], sub.get('angle', 75))
                
                if done:
                    self.subtasks_done[i] = True
                    if target and target in self.object_tracking:
                        self.object_tracking[target]['success'] = True

    def calculate_scores(self, task):
        n = len(self.involved_objects) if self.involved_objects else 1
        
        # --- 1. UAV Approach Metrics ---
        score_uav = 0.0
        for obj, data in self.object_tracking.items():
            if data['success']:
                score_uav += 25.0 / n
                continue
            
            def get_gauss(d, peak, edge, score_at_edge):
                if d < 0.5 or d > 0.8: return 0.0
                sigma = abs(peak - edge) / np.sqrt(-2 * np.log(score_at_edge))
                return np.exp(-((d - peak)**2) / (2 * sigma**2))

            s1 = get_gauss(data['min_uav_dist'], 0.65, 0.5, 0.2) * 0.5
            s2 = 0.0
            if data['acted_in_range']:
                s2 = get_gauss(data['min_uav_dist'], 0.65, 0.5, 0.6) * 0.5
            
            score_uav += (s1 + s2) * (25.0 / n)
            
        self.metrics['uav_score'] = min(25.0, score_uav) 

        # --- 2. Robotic Arm Metrics ---
        _, ee_quat = self.ee_prim.get_world_poses()
        rot_mat = R.from_quat([ee_quat[0][1], ee_quat[0][2], ee_quat[0][3], ee_quat[0][0]]).as_matrix()
        z_axis = rot_mat[:, 2] 
        
        is_drawer = any(sub['type'] == 'drawer' for sub in task.get('subtasks', [task]))
        orientation_score = 0.0
        
        if is_drawer:
            if abs(z_axis[2]) < 0.3: orientation_score = 5.0
        else:
            if z_axis[2] < -0.7: orientation_score = 5.0
            
        score_interact = 0.0
        for obj, data in self.object_tracking.items():
            if data['success']:
                score_interact += 20.0 / n
                continue
            
            d_arm = data['min_arm_dist_on_action']
            if d_arm == float('inf'):
                s_arm = 0.0
            elif d_arm < 0.1:
                s_arm = 0.5 
            elif d_arm <= 0.5:
                ratio = (d_arm - 0.1) / (0.4) 
                s_arm = 0.5 - ratio * 0.3 
            else:
                s_arm = 0.0 
            
            score_interact += s_arm * (20.0 / n)
            
        self.metrics['arm_score'] = orientation_score + score_interact

        # --- 3. Environmental Safety ---
        safety_score = 10.0
        total_displacement = 0.0
        stage = omni.usd.get_context().get_stage()
        
        for obj_path, state in self.saver.initial_states.items():
            if obj_path in self.involved_objects: continue
            if "drawer" in obj_path: continue 
            if not stage.GetPrimAtPath(obj_path).IsValid(): continue
            
            curr_pos, _ = XFormPrim(obj_path).get_world_poses()
            disp = np.linalg.norm(curr_pos[0] - state["pos"])
            total_displacement += disp
            
        if total_displacement <= 0.01:
            safety_score = 10.0
        elif total_displacement >= 2.0:
            safety_score = 0.0
        else:
            sigma = 2.0 / 3.0
            safety_score = 10.0 * np.exp(-(total_displacement**2) / (2 * sigma**2))
            
        self.metrics['safety_score'] = safety_score

        # --- 4. Task Completion Ratio ---
        num_sub = len(self.subtasks_done)
        num_success = sum(self.subtasks_done)
        if num_sub > 0:
            self.metrics['completion_score'] = (num_success / num_sub) * 40.0
        else:
            self.metrics['completion_score'] = 0.0

        # Calculate Final Score
        self.metrics['total_score'] = (
            self.metrics['uav_score'] + 
            self.metrics['arm_score'] + 
            self.metrics['safety_score'] + 
            self.metrics['completion_score']
        )
        return self.metrics

# =================================================================================
# 6. Results & Logging
# =================================================================================
class ResultLogger:
    """Handles logging execution data, rendering output videos, and exporting CSV metrics."""
    def __init__(self, root_path):
        self.root_path = root_path
        if not os.path.exists(root_path):
            os.makedirs(root_path)
    
    def get_task_dir(self, task):
        return os.path.join(self.root_path, task['group'], task['cmd'])

    def log_episode(self, task, episode_idx, metrics, final_image, frames):
        task_dir = self.get_task_dir(task)
        
        # 1. Save Last Viewport Image
        img_dir = os.path.join(task_dir, "image")
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, f"episode_{episode_idx}.jpg"), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        
        # 2. Render Trajectory Video
        vid_dir = os.path.join(task_dir, "video")
        os.makedirs(vid_dir, exist_ok=True)
        if frames:
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter(
                os.path.join(vid_dir, f"episode_{episode_idx}.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), 
                10.0, (w, h)
            )
            for f in frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            
        # 3. Save CSV Data
        csv_path = os.path.join(task_dir, "episode_results.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Episode", "UAV_Score", "Arm_Score", "Safety_Score", "Completion_Score", "Total_Score"])
            writer.writerow([
                episode_idx, 
                f"{metrics['uav_score']:.2f}",
                f"{metrics['arm_score']:.2f}",
                f"{metrics['safety_score']:.2f}",
                f"{metrics['completion_score']:.2f}",
                f"{metrics['total_score']:.2f}"
            ])

    def summarize_group(self, task_list, all_results):
        """Generates statistical summaries for each tested task group."""
        for group in set(t['group'] for t in task_list):
            group_dir = os.path.join(self.root_path, group)
            os.makedirs(group_dir, exist_ok=True)
            
            group_stats = []
            tasks_in_group = [t for t in task_list if t['group'] == group]
            
            for t in tasks_in_group:
                cmd = t['cmd']
                if cmd not in all_results: continue
                
                results = all_results[cmd]
                avg_metrics = {
                    k: np.mean([r[k] for r in results]) 
                    for k in results[0].keys()
                }
                
                task_dir = self.get_task_dir(t)
                with open(os.path.join(task_dir, "task_summary.json"), 'w') as f:
                    json.dump(avg_metrics, f, indent=4)
                
                group_stats.append(avg_metrics)
                
            if group_stats:
                avg_group = {
                    k: np.mean([s[k] for s in group_stats])
                    for k in group_stats[0].keys()
                }
                with open(os.path.join(group_dir, "group_summary.json"), 'w') as f:
                    json.dump(avg_group, f, indent=4)

# =================================================================================
# 7. Main Execution Loop
# =================================================================================
def run_simulation(args):
    server = ServerInterface(args.server_url)
    sensors = SensorManager()
    scene_saver = SceneStateSaver()
    logger = ResultLogger(args.result_root)
    
    global_results = {} 

    PHYSICS_DT = 1.0 / 10.0
    CHUNK_SIZE = 5
    CHUNK_DURATION = CHUNK_SIZE * PHYSICS_DT 

    def apply_materials(stage):
        mtl = PhysicsMaterial(prim_path="/World/Physics_Materials/HighFriction", static_friction=2.0, dynamic_friction=1.5, restitution=0.0)
        for path in INTERACTABLE_OBJECTS:
            if stage.GetPrimAtPath(path).IsValid():
                GeometryPrim(prim_path=path).apply_physics_material(mtl)

    for task_idx, task in enumerate(TASK_LIST):
        print(f"\n=== [{task['group']}] Execution [{task_idx+1}/{len(TASK_LIST)}]: {task['cmd']} ===")
        
        if task['cmd'] not in global_results:
            global_results[task['cmd']] = []

        usd_file = os.path.join(args.usd_root, task['usd'])
        if not os.path.exists(usd_file): 
            print(f"[WARNING] USD environment {usd_file} not found. Skipping...")
            continue
        
        open_stage(usd_file)
        
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
        world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
        sensors.setup() 
        
        franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
        franka_parent = XFormPrim("/World/Franka")
        
        evaluator = Evaluator(
            scene_saver, 
            "/World/Franka/panda_hand"
        )
        
        apply_materials(stage)
        
        world.reset()
        franka.initialize()

        all_dof_names = franka.dof_names
        target_joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
        arm_indices = []
        for target in target_joint_names:
            for i, name in enumerate(all_dof_names):
                if target in name: arm_indices.append(i); break
        ARM_INDICES_ARR = np.array(arm_indices, dtype=np.int32)
        
        scene_saver.capture_initial_state()

        usd_base_pos, usd_base_quat = franka_parent.get_world_poses()
        init_drone_pos = usd_base_pos[0]
        init_drone_quat = usd_base_quat[0]
        
        init_arm_joints = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.04, 0.04])
        franka.set_joint_positions(init_arm_joints, joint_indices=ARM_INDICES_ARR)
        
        time_limit = 80.0 if task['group'] == 'long' else 50.0
        max_inference_steps = int(time_limit / CHUNK_DURATION)

        for episode in range(1, 2): 
            print(f"  [Sim] Running Episode {episode}/1 (Limit: {time_limit}s)...")
            
            # Sub-stepping sequence to stabilize physics
            world.reset() 
            scene_saver.reset_objects() 
            franka_parent.set_world_poses(positions=np.array([init_drone_pos]), orientations=np.array([init_drone_quat]))
            franka.set_joint_positions(init_arm_joints, joint_indices=ARM_INDICES_ARR)
            for _ in range(50): world.step(render=False)
            scene_saver.reset_objects()
            franka_parent.set_world_poses(positions=np.array([init_drone_pos]), orientations=np.array([init_drone_quat]))
            franka.set_joint_positions(init_arm_joints, joint_indices=ARM_INDICES_ARR)
            for _ in range(20): world.step(render=True)
            
            evaluator.reset_task(task)
            episode_frames = []
            
            gripper_was_closed = False
            task_completed = False
            
            for inf_step in range(max_inference_steps):
                images = sensors.capture()
                
                if VIDEO_SOURCE_CAM in images:
                    episode_frames.append(images[VIDEO_SOURCE_CAM].copy())

                raw_joints = franka.get_joint_positions(joint_indices=ARM_INDICES_ARR)
                if raw_joints is None: 
                    world.step(render=False)
                    raw_joints = np.zeros(len(ARM_INDICES_ARR))
                
                current_arm_joints = raw_joints[:7]
                current_gripper_state = 1.0 if gripper_was_closed else 0.0

                robot_state_vec = np.concatenate([current_arm_joints, [current_gripper_state]])

                actions_chunk = server.get_action(task['cmd'], images, robot_state=robot_state_vec)
                
                if actions_chunk is None or len(actions_chunk) == 0:
                     actions_chunk = [[0.0]*11 + [current_gripper_state]] * CHUNK_SIZE
                
                for action_data in actions_chunk:
                    step_base_pos, step_base_quat = franka_parent.get_world_poses()
                    step_base_pos = step_base_pos[0]
                    step_r = R.from_quat([step_base_quat[0][1], step_base_quat[0][2], step_base_quat[0][3], step_base_quat[0][0]])
                    step_yaw_rad = step_r.as_euler('xyz')[2]
                    r_matrix_exec = R.from_euler('z', step_yaw_rad).as_matrix()

                    # Extract dimensions 0-3 for Base UAV
                    dx_body = action_data[0]
                    dy_body = action_data[1]
                    dz_body = action_data[2]
                    dw_deg  = action_data[3]
                    
                    # Apply deadzone filtering
                    if abs(dx_body) < 0.0005: dx_body = 0.0
                    if abs(dy_body) < 0.0005: dy_body = 0.0
                    if abs(dz_body) < 0.0005: dz_body = 0.0
                    if abs(dw_deg)  < 0.05:   dw_deg = 0.0

                    # Extract dimensions 4-10 for Arm, 11 for Gripper
                    target_joints_arm = action_data[4:11]
                    gripper_cmd = action_data[11]
                    is_close = (gripper_cmd > 0.5)

                    action_move_body = np.array([dx_body, dy_body, dz_body])
                    action_move_world = r_matrix_exec @ action_move_body
                    
                    new_pos = step_base_pos + action_move_world
                    new_yaw_rad = step_yaw_rad + np.deg2rad(dw_deg)
                    
                    q_new = R.from_euler('xyz', [np.pi, 0, new_yaw_rad]).as_quat()
                    new_quat = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])
                    
                    franka_parent.set_world_poses(positions=np.array([new_pos]), orientations=np.array([new_quat]))

                    full_joints = np.zeros(9)
                    full_joints[:7] = target_joints_arm
                    
                    # Direct Physics Execution based on open/close target width
                    target_width = DEFAULT_CLOSE_WIDTH if is_close else FRANKA_MAX_OPEN 
                    full_joints[7] = target_width / 2.0
                    full_joints[8] = target_width / 2.0
                    
                    arm_action = ArticulationAction(joint_positions=full_joints, joint_indices=ARM_INDICES_ARR)
                    franka.apply_action(arm_action)
                    
                    world.step(render=True)
                    
                    gripper_was_closed = is_close
                    evaluator.update(task, gripper_was_closed)
                
                if all(evaluator.subtasks_done):
                    task_completed = True
                    break
            
            metrics = evaluator.calculate_scores(task)
            global_results[task['cmd']].append(metrics)
            
            final_img = episode_frames[-1] if episode_frames else np.zeros((480, 640, 3), dtype=np.uint8)
            logger.log_episode(task, episode, metrics, final_img, episode_frames)
            
            print(f"    [Result] Total: {metrics['total_score']:.2f} (UAV: {metrics['uav_score']:.1f}, Arm: {metrics['arm_score']:.1f}, Safety: {metrics['safety_score']:.1f}, Completion: {metrics['completion_score']:.1f})")

        world.stop()
    
    logger.summarize_group(TASK_LIST, global_results)

    if server: server.close()
    simulation_app.close()
    print(f"[System] Testing completed. All metrics saved to {args.result_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isaac Sim Aerial Manipulation Evaluator")
    parser.add_argument("--server_url", type=str, default="ws://127.0.0.1:8000", help="WebSocket URL for inference server.")
    parser.add_argument("--usd_root", type=str, default="./environments", help="Root directory of USD stage files.")
    parser.add_argument("--result_root", type=str, default="./results", help="Directory where logs and videos are saved.")
    parser.add_argument("--debug_dir", type=str, default="./debug_captured_images", help="Directory for debug frame saving.")
    args = parser.parse_args()

    # Pre-clean debug directory if required based on standard behavior
    if os.path.exists(args.debug_dir):
        shutil.rmtree(args.debug_dir)
    os.makedirs(args.debug_dir, exist_ok=True)
    print(f"[INFO] Debug images directory initialized at: {args.debug_dir}")

    run_simulation(args)
