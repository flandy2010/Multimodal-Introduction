import os
import re
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


class VideoTransformEngine:
    def __init__(self, output_size=(256, 256), fps=24, noise_std=5):

        self.output_size = output_size
        self.fps = fps
        self.noise_std = noise_std
        try:
            # 这里的路径根据系统修改
            self.font = ImageFont.truetype("Arial Unicode.ttf", 20)
        except:
            self.font = ImageFont.load_default()

    def transform_to_tensor(self, image, inst_text, num_frames=16):
        """单图转视频 Tensor: 增加了平滑翻转动画"""
        frames = []
        params = self._parse_instruction(inst_text)
        inst_type, val = params['type'], params['val']

        for i in range(num_frames):
            progress = i / (num_frames - 1)
            frame = image.clone()

            if inst_type == "旋转":
                frame = TF.rotate(frame, val * progress)
            elif inst_type == "放大":
                scale = 1.0 + (val - 1.0) * progress
                frame = self._center_scale(frame, scale)
            elif inst_type == "缩小":
                scale = 1.0 - (1.0 - 1.0 / val) * progress
                frame = self._center_scale(frame, scale)

            # --- 核心改进：平滑水平翻转 ---
            elif inst_type == "水平翻转":
                h_scale = abs(1 - 2 * progress)
                c, h, w = frame.shape
                # 关键修正：确保 new_w 最小为 1
                new_w = max(int(w * h_scale), 1)

                if progress > 0.5:
                    frame = TF.hflip(frame)

                # 现在这里不会报错了，因为 new_w 至少是 1
                frame = TF.resize(frame, [h, new_w], antialias=True)

                pad_left = (w - new_w) // 2
                pad_right = w - new_w - pad_left
                frame = F.pad(frame, (pad_left, pad_right, 0, 0), value=-1.0)

            # --- 修正后的垂直翻转 ---
            elif inst_type == "垂直翻转":
                v_scale = abs(1 - 2 * progress)
                c, h, w = frame.shape
                # 关键修正：确保 new_h 最小为 1
                new_h = max(int(h * v_scale), 1)

                if progress > 0.5:
                    frame = TF.vflip(frame)

                frame = TF.resize(frame, [new_h, w], antialias=True)

                pad_top = (h - new_h) // 2
                pad_bottom = h - new_h - pad_top
                frame = F.pad(frame, (0, 0, pad_top, pad_bottom), value=-1.0)

            frames.append(frame)
        return torch.stack(frames)

    def save_to_mp4(self, video_tensor, label, inst_text, save_dir="./examples"):
        """保存为 mp4 视频"""
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"数字{label}_{inst_text}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, self.output_size)

        for i in range(len(video_tensor)):
            img_pil = TF.to_pil_image((video_tensor[i] + 1) / 2).convert("RGB")
            img_pil = img_pil.resize(self.output_size, Image.LANCZOS)

            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), f"标签: {label}", font=self.font, fill=(0, 255, 0))
            draw.text((10, self.output_size[1] - 30), f"指令: {inst_text}", font=self.font, fill=(255, 255, 255))

            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        return save_path

    def save_to_grid_image(self, video_tensor, label, inst_text, save_dir="./examples", cell_size=(64, 64)):
        """
        核心新增：将视频帧拼成 10 帧一行的网格大图
        cell_size: 每一帧在网格中的预览大小
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_frames = video_tensor.shape[0]
        cols = 10
        rows = math.ceil(num_frames / cols)

        # 创建大画布 (背景黑色)
        grid_w = cols * cell_size[0]
        grid_h = rows * cell_size[1]
        grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

        for i in range(num_frames):
            # 1. 转换并缩放单帧
            frame_pil = TF.to_pil_image((video_tensor[i] + 1) / 2).convert("RGB")
            frame_pil = frame_pil.resize(cell_size, Image.NEAREST)  # 使用 NEAREST 保留像素感

            # 2. 计算位置
            r = i // cols
            c = i % cols
            grid_img.paste(frame_pil, (c * cell_size[0], r * cell_size[1]))

        # 3. 在图片顶部或底部写上信息
        # 如果需要可以在画布预留空间写字，这里直接保存
        save_path = os.path.join(save_dir, f"数字{label}_{inst_text}_预览.png")
        grid_img.save(save_path)
        print(f"Grid image saved to {save_path}")
        return save_path

    # --- 内部工具方法 ---
    def _parse_instruction(self, text):
        res = {'type': None, 'val': 0}
        nums = re.findall(r'\d+', text)
        val = int(nums[0]) if nums else 0
        if "旋转" in text:
            res.update({'type': "旋转", 'val': val})
        elif "放大" in text:
            res.update({'type': "放大", 'val': val})
        elif "缩小" in text:
            res.update({'type': "缩小", 'val': val})
        elif "水平" in text:
            res.update({'type': "水平翻转"})
        elif "垂直" in text:
            res.update({'type': "垂直翻转"})
        return res

    def _center_scale(self, tensor, scale):
        c, h, w = tensor.shape
        nh, nw = int(h * scale), int(w * scale)
        x = TF.resize(tensor, [nh, nw], antialias=True)
        if scale >= 1.0:
            return TF.center_crop(x, [h, w])
        else:
            ph, pw = (h - nh) // 2, (w - nw) // 2
            return F.pad(x, (pw, w - nw - pw, ph, h - nh - ph), value=-1.0)


def create_synthetic_five(background=-1.0, foreground=1.0):
    """
    伪造一个 1x28x28 的数字 5
    background: 背景数值 (通常为 -1.0)
    foreground: 笔画数值 (通常为 1.0)
    """

    img = torch.full((1, 28, 28), background)
    img[0, 6:8, 8:21] = foreground
    img[0, 8:15, 8:10] = foreground
    for r in range(28):
        for c in range(28):
            dist = ((r - 18) ** 2 + (c - 14) ** 2) ** 0.5
            if 5 <= dist <= 8:
                if not (r < 18 and c < 14):
                    img[0, r, c] = foreground
    img[0, 14, 8:15] = foreground
    return img


if __name__ == '__main__':

    engine = VideoTransformEngine()
    image = create_synthetic_five()

    for instruction in ["放大3倍", "缩小2倍", "水平翻转", "垂直翻转", "旋转60度"]:
        video_tensor = engine.transform_to_tensor(image, instruction, num_frames=30)
        # engine.save_to_mp4(video_tensor, 5, instruction)
        engine.save_to_grid_image(video_tensor, 5, instruction)