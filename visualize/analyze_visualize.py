# visualize/analyze_and_visualize_all.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import VideoClip
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# -------------------------------
# 自动定位根目录（无论脚本在哪）
# -------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# -------------------------------
# 配置路径
# -------------------------------
original_dir = os.path.join(base_dir, 'dataset', 'OriginalModel_npy')
my_model_dir = os.path.join(base_dir, 'dataset', 'MyModel_npy')
train_loss_path = os.path.join(base_dir, 'logs', 'train_loss.npy')
val_loss_path = os.path.join(base_dir, 'logs', 'val_loss.npy')

output_frame_dir = os.path.join(base_dir, 'analysis_frames')
output_video_dir = os.path.join(base_dir, 'analysis_videos')
os.makedirs(output_frame_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

num_random_videos = 3
fps = 20
frame_scale = 5

# -------------------------------
# 英文编号 kinematic_tree
# -------------------------------
kinematic_tree = [
    [0,3],[3,6],[6,9],[9,12],[12,15],
    [0,2],[2,5],[5,8],[8,11],
    [0,1],[1,4],[4,7],[7,10],
    [9,14],[14,17],[17,19],[19,21],
    [9,13],[13,16],[16,18],[18,20]
]

# -------------------------------
# 函数：计算长度、位移、关节幅度
# -------------------------------
def get_lengths(input_dir):
    lengths = []
    for f in os.listdir(input_dir):
        if f.endswith('.npy'):
            motion = np.load(os.path.join(input_dir,f))
            lengths.append(motion.shape[0])
    return lengths

def get_displacements(input_dir):
    displacements = []
    for f in os.listdir(input_dir):
        if f.endswith('.npy'):
            motion = np.load(os.path.join(input_dir,f))
            joints = recover_from_ric(torch.from_numpy(motion).float(), 22).cpu().numpy()
            disp = np.linalg.norm(joints.max(axis=0) - joints.min(axis=0))
            displacements.append(disp)
    return displacements

def get_joint_ranges(input_dir):
    joint_ranges = np.zeros((22,3))
    count = 0
    for f in os.listdir(input_dir):
        if f.endswith('.npy'):
            motion = np.load(os.path.join(input_dir,f))
            joints = recover_from_ric(torch.from_numpy(motion).float(),22).cpu().numpy()
            joint_ranges += joints.max(axis=0) - joints.min(axis=0)
            count += 1
    return joint_ranges / count

# -------------------------------
# 数据统计
# -------------------------------
original_lengths = get_lengths(original_dir)
my_lengths = get_lengths(my_model_dir)
original_disp = get_displacements(original_dir)
my_disp = get_displacements(my_model_dir)
original_joint_range = get_joint_ranges(original_dir)
my_joint_range = get_joint_ranges(my_model_dir)

# -------------------------------
# 绘图：动作长度
# -------------------------------
plt.figure()
plt.hist(original_lengths, bins=20, alpha=0.5, label='Original Model')
plt.hist(my_lengths, bins=20, alpha=0.5, label='My Model')
plt.xlabel('Frames per motion')
plt.ylabel('Count')
plt.title('Motion length distribution')
plt.legend()
plt.show()

# -------------------------------
# 绘图：空间位移
# -------------------------------
plt.figure()
plt.hist(original_disp, bins=20, alpha=0.5, label='Original Model')
plt.hist(my_disp, bins=20, alpha=0.5, label='My Model')
plt.xlabel('Spatial displacement')
plt.ylabel('Count')
plt.title('Motion spatial extent')
plt.legend()
plt.show()

# -------------------------------
# 绘图：关节幅度热力图
# -------------------------------
plt.figure(figsize=(10,4))
sns.heatmap(original_joint_range, annot=True, fmt=".2f", cmap='Blues')
plt.title('Original Model joint movement range (x,y,z)')
plt.show()

plt.figure(figsize=(10,4))
sns.heatmap(my_joint_range, annot=True, fmt=".2f", cmap='Reds')
plt.title('My Model joint movement range (x,y,z)')
plt.show()

# -------------------------------
# 绘图：Loss vs Epoch
# -------------------------------
train_loss = np.load(train_loss_path)
val_loss = np.load(val_loss_path)
epochs = range(1, len(train_loss)+1)

plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

# -------------------------------
# 随机动作视频对比
# -------------------------------
original_files = [f for f in os.listdir(original_dir) if f.endswith('.npy')]
sample_files = np.random.choice(original_files, size=min(num_random_videos,len(original_files)), replace=False)

for f in sample_files:
    orig_motion = np.load(os.path.join(original_dir,f))
    my_motion = np.load(os.path.join(my_model_dir,f))
    orig_joints = recover_from_ric(torch.from_numpy(orig_motion).float(),22).cpu().numpy()
    my_joints = recover_from_ric(torch.from_numpy(my_motion).float(),22).cpu().numpy()
    orig_joints = (orig_joints - orig_joints.mean(axis=(0,1),keepdims=True))*frame_scale
    my_joints = (my_joints - my_joints.mean(axis=(0,1),keepdims=True))*frame_scale
    n_frames = min(orig_joints.shape[0], my_joints.shape[0])

    def make_frame(t):
        idx = min(int(t*fps), n_frames-1)
        fig, axes = plt.subplots(1,2,subplot_kw={'projection':'3d'},figsize=(10,5))
        for ax,joints,label in zip(axes,[orig_joints,my_joints],['Original','My Model']):
            x,y,z = joints[idx,:,0], joints[idx,:,1], joints[idx,:,2]
            ax.scatter(x,y,z,c='red',s=50)
            for a,b in kinematic_tree:
                ax.plot([x[a],x[b]],[y[a],y[b]],[z[a],z[b]],c='black',linewidth=2)
            ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_zlim(-10,10)
            ax.view_init(elev=120,azim=-90)
            ax.set_title(label)
            ax.axis('off')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        plt.close(fig)
        return frame

    video_path = os.path.join(output_video_dir,f.replace('.npy','_compare.mp4'))
    clip = VideoClip(make_frame, duration=n_frames/fps)
    clip.write_videofile(video_path, fps=fps)
    print(f"✅ Saved comparison video: {video_path}")