import os
import pandas as pd
import h5py
import cv2
import numpy as np
import tqdm

# ================= 配置区域 =================
# 指定 dataset_raw 根目录
DATASET_RAW_ROOT = "/media/ubuntu/data/dataset_raw"
# 指定输出 HDF5 的根目录
DATASET_HDF5_ROOT = "/media/ubuntu/data/dataset_hdf5"

# ================= 基础配置 =================
# 根据实际 actions.csv 表头，夹爪列名为 joint_16_pos
basic_columns = ['drone_dx', 'drone_dy', 'drone_dz', 'drone_dw', 'joint_16_pos']

# 定义需要使用「原有自由度映射」的任务路径（精确匹配前缀）
ORIGINAL_JOINT_TASKS = [
    "long",
    "object",
    "semantic_understanding",
    "manipulation"
]
# 原有关节列（第一种映射）
ORIGINAL_JOINT_COLUMNS = ['joint_0_pos', 'joint_1_pos', 'joint_6_pos', 'joint_11_pos', 'joint_12_pos', 'joint_13_pos', 'joint_14_pos']
# 替换后的关节列（第二种映射）
REPLACE_JOINT_COLUMNS = ['joint_0_pos', 'joint_1_pos', 'joint_2_pos', 'joint_3_pos', 'joint_4_pos', 'joint_5_pos', 'joint_6_pos']

def get_video_frame_count(video_path):
    """获取视频的帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def extract_video_frames(video_path):
    """提取视频的所有帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def is_use_original_joint_mapping(task_relative_path):
    """判断当前任务是否使用原有自由度映射"""
    for target_path in ORIGINAL_JOINT_TASKS:
        if task_relative_path.startswith(target_path):
            return True
    return False

def process_episode(episode_folder, episode_input_path, output_folder, task_name, task_relative_path, episode_index):
    """处理单个episode"""
    # 1. 读取 actions.csv
    actions_path = os.path.join(episode_input_path, 'actions.csv')
    if not os.path.exists(actions_path):
        print(f"{episode_folder} - actions.csv not found!")
        return False
    
    try:
        df = pd.read_csv(actions_path, header=0)
    except Exception as e:
        print(f"{episode_folder} - Failed to read actions.csv: {e}")
        return False
    
    # 2. 判断关节映射
    use_original_joint = is_use_original_joint_mapping(task_relative_path)
    if use_original_joint:
        joint_columns = ORIGINAL_JOINT_COLUMNS
    else:
        joint_columns = REPLACE_JOINT_COLUMNS
    
    # 补充缺失列
    required_columns = basic_columns + joint_columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0

    actions_count = len(df)
    
    # 3. 检查视频帧数
    video_folders = ['drone_rgb', 'perspective_rgb', 'side_rgb', 'wrist_rgb']
    video_frame_counts = {}
    
    for folder in video_folders:
        video_path = os.path.join(episode_input_path, folder, f'{folder}.mp4')
        if os.path.exists(video_path):
            frame_count = get_video_frame_count(video_path)
            video_frame_counts[folder] = frame_count
        else:
            print(f"{episode_folder} - {folder}: Video file not found")
            return False
    
    # 检查一致性
    if len(set(video_frame_counts.values())) != 1:
        print(f"{episode_folder} - Video frame counts mismatch: {video_frame_counts}")
        return False
    
    video_frame_count = list(video_frame_counts.values())[0]
    
    # 校验 CSV 行数与视频帧数
    if actions_count != video_frame_count and actions_count != video_frame_count + 1:
        print(f"{episode_folder} - Actions count ({actions_count}) doesn't match video frame count ({video_frame_count})")
        return False
    
    if actions_count == video_frame_count + 1:
        df = df.iloc[:video_frame_count]
        actions_count = video_frame_count
    
    # 4. 抽帧处理 (60Hz -> 20Hz)
    downsampled_count = (actions_count + 2) // 3
    window_size = 3
    
    # --- 提取无人机数据 (Action 需要) ---
    drone_dx = df['drone_dx'].values
    drone_dy = df['drone_dy'].values
    drone_dz = df['drone_dz'].values
    drone_dw = np.rad2deg(df['drone_dw'].values)
    
    # 累积和计算窗口增量
    cumsum_dx = np.insert(np.cumsum(drone_dx), 0, 0)
    cumsum_dy = np.insert(np.cumsum(drone_dy), 0, 0)
    cumsum_dz = np.insert(np.cumsum(drone_dz), 0, 0)
    cumsum_dw = np.insert(np.cumsum(drone_dw), 0, 0)
    
    window_ends = np.arange(window_size, actions_count + window_size, window_size)[:downsampled_count]
    window_ends = np.minimum(window_ends, actions_count)
    window_starts = window_ends - window_size
    
    window_dxs = cumsum_dx[window_ends] - cumsum_dx[window_starts]
    window_dys = cumsum_dy[window_ends] - cumsum_dy[window_starts]
    window_dzs = cumsum_dz[window_ends] - cumsum_dz[window_starts]
    window_dws = cumsum_dw[window_ends] - cumsum_dw[window_starts]

    # --- 提取关节数据 (最后一帧) ---
    last_frame_indices = [min((i + 1) * window_size - 1, actions_count - 1) for i in range(downsampled_count)]
    joint_data_list = []
    for col in joint_columns:
        joint_data_list.append(df.iloc[last_frame_indices][col].values)
    
    joint_0_pos, joint_1_pos, joint_2_or_6_pos, joint_3_or_11_pos, joint_4_or_12_pos, joint_5_or_13_pos, joint_6_or_14_pos = joint_data_list
    
    # 使用实际的夹爪列 joint_16_pos 代替原本硬编码的 gripper_closed
    gripper_closed = df.iloc[last_frame_indices]['joint_16_pos'].values
    
    # --- 构建 State (8维) 和 Action (12维) ---
    obs_dim = 8   # 7 joints + 1 gripper
    act_dim = 12  # 4 drone + 7 joints + 1 gripper
    
    downsampled_states = np.zeros((downsampled_count, obs_dim), dtype=np.float32)
    downsampled_actions = np.zeros((downsampled_count, act_dim), dtype=np.float32)
    
    for i in range(downsampled_count):
        # Observation/State: 仅包含机械臂和夹爪 (8维)
        downsampled_states[i] = np.array([
            joint_0_pos[i], joint_1_pos[i], joint_2_or_6_pos[i],
            joint_3_or_11_pos[i], joint_4_or_12_pos[i], joint_5_or_13_pos[i],
            joint_6_or_14_pos[i], gripper_closed[i]
        ], dtype=np.float32)
        
        # Action: 包含无人机位姿变化 + 机械臂 + 夹爪 (12维)
        if (i + 1) < downsampled_count:
            downsampled_actions[i] = np.array([
                window_dxs[i + 1], window_dys[i + 1], window_dzs[i + 1], window_dws[i + 1], # Drone
                joint_0_pos[i + 1], joint_1_pos[i + 1], joint_2_or_6_pos[i + 1],
                joint_3_or_11_pos[i + 1], joint_4_or_12_pos[i + 1], joint_5_or_13_pos[i + 1],
                joint_6_or_14_pos[i + 1], gripper_closed[i + 1]
            ], dtype=np.float32)
        else:
            # 最后一个动作，复制当前状态并补全 Drone 部分 (设为0或窗口值)
            downsampled_actions[i] = np.array([
                window_dxs[i], window_dys[i], window_dzs[i], window_dws[i],
                joint_0_pos[i], joint_1_pos[i], joint_2_or_6_pos[i],
                joint_3_or_11_pos[i], joint_4_or_12_pos[i], joint_5_or_13_pos[i],
                joint_6_or_14_pos[i], gripper_closed[i]
            ], dtype=np.float32)
            
    # --- 处理视频 ---
    downsampled_video_frames = {}
    for folder in video_folders:
        video_path = os.path.join(episode_input_path, folder, f'{folder}.mp4')
        all_frames = extract_video_frames(video_path)
        downsampled_frames = []
        for i in range(downsampled_count):
            frame_idx = i * 3
            if frame_idx < len(all_frames):
                downsampled_frames.append(all_frames[frame_idx])
            else:
                downsampled_frames.append(all_frames[-1])
        downsampled_video_frames[folder] = downsampled_frames
        
    timestamps = np.arange(0, downsampled_count) / 20.0
    
    # 写入 HDF5
    episode_output_name = f'episode_{episode_index:03d}.hdf5'
    episode_output_path = os.path.join(output_folder, episode_output_name)
    
    try:
        with h5py.File(episode_output_path, 'w') as h5f:
            h5f.create_dataset('/observations/state', data=downsampled_states, dtype=np.float32) # (N, 8)
            h5f.create_dataset('/action', data=downsampled_actions, dtype=np.float32)            # (N, 12)
            
            for folder, frames in downsampled_video_frames.items():
                image_shape = frames[0].shape
                images_dataset = h5f.create_dataset(f'/observations/images/{folder}', 
                                                  shape=(len(frames),) + image_shape, 
                                                  dtype=np.uint8)
                for i, frame in enumerate(frames):
                    images_dataset[i] = frame
            
            h5f.create_dataset('/timestamps', data=timestamps, dtype=np.float32)
            h5f.create_dataset('/task', data=task_name.encode('utf-8'))
    except Exception as e:
        print(f"{episode_folder} - Failed to create hdf5 file: {e}")
        return False
    
    return True

def process_all_datasets():
    """遍历处理 dataset_raw 下的所有大类和任务文件夹"""
    
    if not os.path.exists(DATASET_RAW_ROOT):
        print(f"Error: Raw dataset root does not exist: {DATASET_RAW_ROOT}")
        return

    # 获取所有大类 (例如: long, manipulation, object, semantic_understanding)
    categories = [d for d in os.listdir(DATASET_RAW_ROOT) if os.path.isdir(os.path.join(DATASET_RAW_ROOT, d))]
    
    total_success = 0
    total_fail = 0

    for category in categories:
        category_path = os.path.join(DATASET_RAW_ROOT, category)
        # 获取大类下的所有子任务 (例如: open_the_top_drawer_and_put_the_banana_inside)
        tasks = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        for task in tasks:
            task_input_path = os.path.join(category_path, task)
            task_output_path = os.path.join(DATASET_HDF5_ROOT, category, task)
            
            # 创建对应的输出目录
            os.makedirs(task_output_path, exist_ok=True)
            
            # 生成任务名称和相对路径供写入和映射判断
            task_name_str = task.replace('_', ' ')
            task_relative_path = f"{category}/{task}"
            
            # 获取当前任务下的所有 episode 文件夹
            episode_folders = [
                f for f in os.listdir(task_input_path)
                if f.startswith('episode_') and os.path.isdir(os.path.join(task_input_path, f))
            ]
            
            if not episode_folders:
                continue
                
            # 按数字排序 episode
            episode_folders.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x)
            
            print(f"\n==============================================")
            print(f"Processing Task: {task_relative_path}")
            print(f"Found {len(episode_folders)} episodes.")
            print(f"Output to: {task_output_path}")
            
            task_success = 0
            task_fail = 0
            
            # 遍历并处理所有 episode
            for idx, episode_folder in enumerate(tqdm.tqdm(episode_folders, desc="Episodes")):
                episode_input_path = os.path.join(task_input_path, episode_folder)
                
                if process_episode(episode_folder, episode_input_path, task_output_path, task_name_str, task_relative_path, idx):
                    task_success += 1
                    total_success += 1
                else:
                    task_fail += 1
                    total_fail += 1
                    
            print(f"Task '{task}' Finish - Success: {task_success}, Failed: {task_fail}")

    print(f"\n================ All Processing Completed ================")
    print(f"Total Success: {total_success}")
    print(f"Total Failed:  {total_fail}")
    print(f"HDF5 Datasets saved in: {DATASET_HDF5_ROOT}")

if __name__ == "__main__":
    process_all_datasets()
