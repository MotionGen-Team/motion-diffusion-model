import numpy as np
import torch
import torch.nn as nn
import os

# ===== CNN模型 =====
class MotionRefiner(nn.Module):
    def __init__(self, joints=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(joints*3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, joints*3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, T, J*3)
        x = x.permute(0, 2, 1)  # → (B, J*3, T)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


# ===== 主逻辑 =====
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    npy_path=os.path.join(base_dir, 'generate_npy', 'results.npy')  # 改成你的路径

    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()

    motion = data['motion']  # (1, 22, 3, 120)

    #  转成 (B, T, J*3)
    motion = motion[0]  # (22, 3, T)
    motion = np.transpose(motion, (2, 0, 1))  # (T, 22, 3)
    T = motion.shape[0]

    motion = motion.reshape(T, -1)  # (T, 66)
    motion = torch.from_numpy(motion).float().unsqueeze(0)  # (1, T, 66)

    # ===== 跑CNN =====
    model = MotionRefiner(joints=22)
    model.eval()

    with torch.no_grad():
        refined = model(motion)

    refined = refined.squeeze(0).numpy()  # (T, 66)

    #  还原回 (T, 22, 3)
    refined = refined.reshape(T, 22, 3)

    #  再转回你可视化用的格式 (22,3,T)
    refined = np.transpose(refined, (1, 2, 0))

    # 保存
    save_dir = os.path.join(base_dir, 'generate_npy')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "refined.npy")
    np.save(save_path, refined)

    print(f"✅ 已保存到 {save_path}")


if __name__ == "__main__":
    main()