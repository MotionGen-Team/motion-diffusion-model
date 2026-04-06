# batch_visualize_humanml3d_final.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# -------------------------------
# 配置
# -------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
input_dir = os.path.join(base_dir, 'generate_npy', 'results.npy')  # 263 数据路径
video_dir = os.path.abspath('../videos')
os.makedirs(video_dir, exist_ok=True)

max_samples = 1   # None=全部,可以改成你想生成的样本数
scale_factor = 5     # 放大骨架
fps = 20             # 帧率

# -------------------------------
# HumanML3D 22 joints 英文编号
# -------------------------------
pelvis_idx      = 0
spine_low_idx   = 3
spine_mid_idx   = 6
chest_idx       = 9
neck_idx        = 12
head_idx        = 15

left_thigh_idx  = 2
left_shin_idx   = 5
left_foot_idx   = 8
left_toe_idx    = 11  # optional

right_thigh_idx = 1
right_shin_idx  = 4
right_foot_idx  = 7
right_toe_idx   = 10  # optional

left_shoulder_idx  = 14
left_upper_arm_idx = 17
left_forearm_idx   = 19
left_hand_idx      = 21

right_shoulder_idx  = 13
right_upper_arm_idx = 16
right_forearm_idx   = 18
right_hand_idx      = 20

kinematic_tree = [
    # spine
    [pelvis_idx, spine_low_idx],
    [spine_low_idx, spine_mid_idx],
    [spine_mid_idx, chest_idx],
    [chest_idx, neck_idx],
    [neck_idx, head_idx],

    # left leg
    [pelvis_idx, left_thigh_idx],
    [left_thigh_idx, left_shin_idx],
    [left_shin_idx, left_foot_idx],
    [left_foot_idx, left_toe_idx],

    # right leg
    [pelvis_idx, right_thigh_idx],
    [right_thigh_idx, right_shin_idx],
    [right_shin_idx, right_foot_idx],
    [right_foot_idx, right_toe_idx],

    # left arm
    [chest_idx, left_shoulder_idx],
    [left_shoulder_idx, left_upper_arm_idx],
    [left_upper_arm_idx, left_forearm_idx],
    [left_forearm_idx, left_hand_idx],

    # right arm
    [chest_idx, right_shoulder_idx],
    [right_shoulder_idx, right_upper_arm_idx],
    [right_upper_arm_idx, right_forearm_idx],
    [right_forearm_idx, right_hand_idx],
]

# -------------------------------
# 批量处理
# -------------------------------
files = [input_dir]

for i, npy_path in enumerate(files):
    file = os.path.basename(npy_path)

    motion = np.load(npy_path)  # (22,3,T)
    motion = np.transpose(motion, (2, 0, 1))  # (T,22,3)

    print(f"[{i + 1}/{len(files)}] Processing {file}, shape: {motion.shape}")

    # 转 tensor
    data_263_tensor = torch.from_numpy(motion).float()

    # recover joints
    #joints_rec = recover_from_ric(data_263_tensor, 22)
    joints_rec = motion  # 直接用
    #joints_rec_np = joints_rec.cpu().numpy().astype(np.float32)
    joints_rec_np = joints_rec.astype(np.float32)

    # 居中 + 放大骨架
    joints_rec_np = joints_rec_np - joints_rec_np.mean(axis=(0,1), keepdims=True)
    joints_rec_np = joints_rec_np * scale_factor

    n_frames = joints_rec_np.shape[0]
    video_path = os.path.join(video_dir, file.replace('.npy', '.mp4'))

    # -------------------------------
    # 定义生成每一帧
    # -------------------------------
    def make_frame(t):
        idx = int(t * fps)
        idx = min(idx, n_frames - 1)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        x = joints_rec_np[idx,:,0]
        y = joints_rec_np[idx,:,1]
        z = joints_rec_np[idx,:,2]

        ax.scatter(x, y, z, c='red', s=50)

        # 绘制骨架
        for a,b in kinematic_tree:
            ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]], c='black', linewidth=2)

        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        ax.view_init(elev=120, azim=-90)
        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    # 生成视频
    clip = VideoClip(make_frame, duration=n_frames/fps)
    clip.write_videofile(video_path, fps=fps)

    print(f"✅ Saved video: {video_path}")