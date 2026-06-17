import os
import re
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


class VideoTransformEngine:
    def __init__(self, output_size=(256, 256), fps=24):
        self.output_size = output_size
        self.fps = fps
        # MNIST 统计量
        self.MEAN = 0.1307
        self.STD = 0.3081
        # 背景（黑色像素 0）在标准正态分布下的值
        self.BG_VAL = (0.0 - self.MEAN) / self.STD  # 约为 -0.4242

        try:
            self.font = ImageFont.truetype("Arial Unicode.ttf", 20)
        except:
            self.font = ImageFont.load_default()

    def _denormalize(self, tensor):
        """将 N(0,1) 的数据还原回 [0, 1] 像素范围用于显示"""
        return (tensor * self.STD + self.MEAN).clamp(0, 1)

    def transform_to_tensor(self, image, inst_text, num_frames=16):
        frames = []
        params = self._parse_instruction(inst_text)
        inst_type, val = params['type'], params['val']

        for i in range(num_frames):
            progress = i / (num_frames - 1)
            frame = image.clone()

            if inst_type == "旋转":
                # 修改 fill 值为背景值
                frame = TF.rotate(frame, val * progress, fill=self.BG_VAL)
            elif inst_type == "放大":
                scale = 1.0 + (val - 1.0) * progress
                frame = self._center_scale(frame, scale)
            elif inst_type == "缩小":
                scale = 1.0 - (1.0 - 1.0 / val) * progress
                frame = self._center_scale(frame, scale)

            elif inst_type == "水平翻转":
                h_scale = abs(1 - 2 * progress)
                c, h, w = frame.shape
                new_w = max(int(w * h_scale), 1)
                if progress > 0.5:
                    frame = TF.hflip(frame)
                frame = TF.resize(frame, [h, new_w], antialias=True)
                pad_left = (w - new_w) // 2
                pad_right = w - new_w - pad_left
                # 修改填充值
                frame = F.pad(frame, (pad_left, pad_right, 0, 0), value=self.BG_VAL)

            elif inst_type == "垂直翻转":
                v_scale = abs(1 - 2 * progress)
                c, h, w = frame.shape
                new_h = max(int(h * v_scale), 1)
                if progress > 0.5:
                    frame = TF.vflip(frame)
                frame = TF.resize(frame, [new_h, w], antialias=True)
                pad_top = (h - new_h) // 2
                pad_bottom = h - new_h - pad_top
                # 修改填充值
                frame = F.pad(frame, (0, 0, pad_top, pad_bottom), value=self.BG_VAL)

            frames.append(frame)
        return torch.stack(frames)

    def save_to_mp4(self, video_tensor, inst_text, save_dir="./examples"):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{inst_text}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, self.output_size)

        for i in range(len(video_tensor)):
            # 修改可视化转换逻辑
            img_normalized = self._denormalize(video_tensor[i])
            img_pil = TF.to_pil_image(img_normalized).convert("RGB")
            img_pil = img_pil.resize(self.output_size, Image.LANCZOS)

            draw = ImageDraw.Draw(img_pil)
            draw.text((10, self.output_size[1] - 30), f"指令: {inst_text}", font=self.font, fill=(255, 255, 255))
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        return save_path

    def save_to_gif(self, video_tensor, inst_text, save_dir="./examples"):
        """将视频张量保存为 GIF 动图"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{inst_text}.gif")

        frames_pil = []
        for i in range(len(video_tensor)):
            # 1. 反归一化并转为 PIL 图像
            img_normalized = self._denormalize(video_tensor[i])
            img_pil = TF.to_pil_image(img_normalized).convert("RGB")

            # 2. 调整尺寸
            img_pil = img_pil.resize(self.output_size, Image.LANCZOS)

            # 3. 叠加指令文字 (与 MP4 逻辑一致)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, self.output_size[1] - 30), f"指令: {inst_text}", font=self.font, fill=(255, 255, 255))

            frames_pil.append(img_pil)

        # 4. 计算每帧时长 (GIF duration 以毫秒为单位)
        # 例如 24fps 对应 duration 约为 41.6ms
        frame_duration = int(1000 / self.fps)

        # 5. 保存 GIF
        # save_all=True: 保存序列帧
        # append_images: 后续帧列表
        # duration: 每帧停留时间
        # loop=0: 无限循环播放
        frames_pil[0].save(
            save_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )

        print(f"GIF 已保存至: {save_path}")
        return save_path

    def save_to_grid_image(self, video_tensor, inst_text, save_dir="./examples", cell_size=(64, 64)):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        num_frames = video_tensor.shape[0]
        cols = 10
        rows = math.ceil(num_frames / cols)
        grid_w, grid_h = cols * cell_size[0], rows * cell_size[1]
        grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

        for i in range(num_frames):
            # 修改可视化转换逻辑
            img_normalized = self._denormalize(video_tensor[i])
            frame_pil = TF.to_pil_image(img_normalized).convert("RGB")
            frame_pil = frame_pil.resize(cell_size, Image.NEAREST)

            r, c = i // cols, i % cols
            grid_img.paste(frame_pil, (c * cell_size[0], r * cell_size[1]))

        save_path = os.path.join(save_dir, f"{inst_text}_预览.png")
        grid_img.save(save_path)
        print(f"Grid image saved to {save_path}")
        return save_path

    def _center_scale(self, tensor, scale):
        c, h, w = tensor.shape
        nh, nw = int(h * scale), int(w * scale)
        x = TF.resize(tensor, [nh, nw], antialias=True)
        if scale >= 1.0:
            return TF.center_crop(x, [h, w])
        else:
            ph, pw = (h - nh) // 2, (w - nw) // 2
            # 修改填充值
            return F.pad(x, (pw, w - nw - pw, ph, h - nh - ph), value=self.BG_VAL)

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


def create_synthetic_five_n01():
    """创建一个符合 N(0,1) 分布的数字 5"""
    mean, std = 0.1307, 0.3081
    bg_val = (0.0 - mean) / std
    fg_val = (1.0 - mean) / std

    img = torch.full((1, 28, 28), bg_val)
    img[0, 6:8, 8:21] = fg_val
    img[0, 8:15, 8:10] = fg_val
    for r in range(28):
        for c in range(28):
            dist = ((r - 18) ** 2 + (c - 14) ** 2) ** 0.5
            if 5 <= dist <= 8:
                if not (r < 18 and c < 14):
                    img[0, r, c] = fg_val
    img[0, 14, 8:15] = fg_val
    return img


if __name__ == '__main__':
    engine = VideoTransformEngine()
    # 使用新生成的符合 N(0,1) 的图像
    image = create_synthetic_five_n01()

    for instruction in ["放大3倍", "缩小2倍", "水平翻转", "垂直翻转", "旋转60度"]:
        video_tensor = engine.transform_to_tensor(image, instruction, num_frames=30)
        engine.save_to_gif(video_tensor, f"将5{instruction}")