import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time


class MNISTVideoGenerator:
    def __init__(self, fps=24, duration=3, output_size=(256, 256), noise_std=8):
        """
        :param noise_std: 噪声强度 (0-255范围)，建议 5-15 之间
        """
        self.fps = fps
        self.duration = duration
        self.output_size = output_size
        self.total_frames = int(fps * duration)
        self.noise_std = noise_std  # 噪声标准差

    def _tensor_to_pil(self, tensor):
        img = tensor.clone().detach().cpu()
        if img.min() < 0: img = (img + 1) / 2
        return transforms.ToPILImage()(img)

    def _add_grain_noise(self, frame):
        """为 OpenCV 图像帧添加随机高斯噪声"""
        # 生成均值为0，标准差为 self.noise_std 的正态分布噪声
        noise = np.random.normal(0, self.noise_size_level(), frame.shape).astype(np.float32)
        # 将噪声叠加到原图，并限制在 0-255 之间
        noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_frame

    def noise_size_level(self):
        """让噪声强度也随时间发生微小波动，看起来更真实"""
        return self.noise_std * np.random.uniform(0.8, 1.2)

    def generate_video(self, image_tensor, label, instruction, output_path="output.mp4"):
        # 初始放大 (LANCZOS 依然只用一次)
        pil_img = self._tensor_to_pil(image_tensor)
        base_img = pil_img.resize(self.output_size, Image.Resampling.LANCZOS)

        inst = instruction.lower()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.output_size)

        print(f"Generating noisy animation: {instruction}...")

        try:
            for i in range(self.total_frames):
                progress = i / self.total_frames

                # --- 1. 参数计算 (加入微小随机抖动 jitter) ---
                angle = 0
                scale = 1.0
                jitter_x = np.random.uniform(-1, 1)  # 每帧 1 像素以内的随机位移
                jitter_y = np.random.uniform(-1, 1)

                if "rotate" in inst:
                    max_angle = float(inst.split()[-1])
                    # 基础角度 + 0.5度以内的随机抖动
                    angle = max_angle * progress + np.random.uniform(-0.5, 0.5)

                if "zoom" in inst:
                    target_scale = float(inst.split()[-1])
                    # 基础缩放 + 微小抖动
                    scale = 1.0 + (target_scale - 1.0) * progress + np.random.uniform(-0.01, 0.01)

                # --- 2. 图像变换 ---
                # 先平移(抖动)再旋转
                frame_img = base_img.rotate(angle, resample=Image.Resampling.BICUBIC, translate=(jitter_x, jitter_y))

                if scale > 1.0:
                    w, h = self.output_size
                    new_w, new_h = int(w / scale), int(h / scale)
                    left = (w - new_w) // 2
                    top = (h - new_h) // 2
                    frame_img = frame_img.crop((left, top, left + new_w, top + new_h))
                    frame_img = frame_img.resize(self.output_size, Image.Resampling.BILINEAR)

                if "flip_h" in inst and progress > 0.5:
                    frame_img = frame_img.transpose(Image.Resampling.FLIP_LEFT_RIGHT)
                if "flip_v" in inst and progress > 0.5:
                    frame_img = frame_img.transpose(Image.Resampling.FLIP_TOP_BOTTOM)

                # --- 3. 核心：添加动态噪声 ---
                # 转为 Numpy 数组
                numpy_frame = np.array(frame_img)
                # 添加像素噪点
                noisy_numpy_frame = self._add_grain_noise(numpy_frame)

                # 转为 BGR 用于 OpenCV 写入
                final_frame = cv2.cvtColor(noisy_numpy_frame, cv2.COLOR_GRAY2BGR)

                # --- 4. 渲染 UI (在噪声之后渲染，保证 UI 清晰) ---
                cv2.putText(final_frame, f"Label: {label}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(final_frame, f"INST: {instruction}", (20, self.output_size[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                video_writer.write(final_frame)

        finally:
            video_writer.release()
            print(f"Finish! Video saved to {output_path}")


# --- 运行测试 ---
if __name__ == "__main__":

    # 模拟 MNIST 数字 7
    mock_img = torch.zeros((1, 28, 28))
    mock_img[0, 5:10, 5:22] = 1.0  # 横
    mock_img[0, 10:25, 18:22] = 1.0  # 竖

    # 设置噪声强度为 12 (较明显)
    gen = MNISTVideoGenerator(fps=24, duration=3, noise_std=12)

    # 生成一个带缩放和噪声的动画
    gen.generate_video(mock_img, label=7, instruction="zoom 2.0", output_path="noisy_zoom.mp4")
    # 生成一个带旋转和噪声的动画
    gen.generate_video(mock_img, label=7, instruction="rotate 180", output_path="noisy_rotate.mp4")