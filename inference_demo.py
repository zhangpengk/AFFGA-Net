# -*- coding: utf-8 -*-
"""
简易推理 Demo - 抓取检测可视化
此 demo 用于演示如何使用 AFFGA 模型进行抓取检测推理
"""

import cv2
import os
import torch
import math
import numpy as np
from skimage.feature import peak_local_max


def calcAngle2(angle):
    """
    根据给定的 angle 计算与之反向的 angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


def drawGrasps(img, grasps, mode='arrow'):
    """
    绘制 grasp
    :param img: 图像
    :param grasps: list() 元素是 [row, col, angle, width]
    :param mode: arrow / region
    :return: 绘制后的图像
    """
    assert mode in ['arrow', 'region']
    
    num = len(grasps)
    if num == 0:
        return img
    
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp
        
        if mode == 'arrow':
            width = min(width / 2, 50)  # 限制宽度显示
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            
            if abs(k) < 1e-6:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx
            
            # 颜色渐变
            color_b = int(255 / max(num, 1) * i)
            color_r = 0
            color_g = int(-255 / max(num, 1) * i + 255)
            
            if angle < math.pi:
                cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), 
                              (0, 0, 255), 2, 8, 0, 0.5)
            else:
                cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), 
                              (0, 0, 255), 2, 8, 0, 0.5)
            
            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), 
                        (0, 0, 255), 2)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), 
                        (0, 0, 255), 2)
            
            cv2.circle(img, (col, row), 3, (color_b, color_g, color_r), -1)
        
        else:
            color_b = int(255 / max(num, 1) * i)
            color_r = 0
            color_g = int(-255 / max(num, 1) * i + 255)
            if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
                img[row, col] = [color_b, color_g, color_r]
    
    return img


def drawRect(img, rect, color=(0, 255, 0), thickness=2):
    """
    绘制矩形
    :param img: 图像
    :param rect: [x1, y1, x2, y2]
    :param color: BGR 颜色
    :param thickness: 线宽
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)
    return img


class SimpleGraspDetector:
    """
    简易抓取检测器（模拟推理）
    用于演示，实际使用时请替换为真实模型
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.input_size = 320
        self.angle_cls = 120
        print(f'>> 初始化抓取检测器，设备：{device}')
    
    def preprocess(self, img):
        """
        预处理：裁剪中间 320x320 区域并归一化
        """
        h, w = img.shape[:2]
        assert h >= self.input_size and w >= self.input_size, \
            f'输入图像必须大于等于 ({self.input_size}, {self.input_size})'
        
        crop_x1 = int((w - self.input_size) / 2)
        crop_y1 = int((h - self.input_size) / 2)
        crop_x2 = crop_x1 + self.input_size
        crop_y2 = crop_y1 + self.input_size
        
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        
        # 归一化
        rgb = crop_img.astype(np.float32) / 255.0
        rgb -= rgb.mean()
        
        # 转换为 tensor (1, 3, 320, 320)
        rgb = rgb.transpose((2, 0, 1))
        rgb_tensor = torch.from_numpy(np.expand_dims(rgb, 0).astype(np.float32))
        
        return rgb_tensor, crop_x1, crop_y1
    
    def predict(self, img, thresh=0.5, peak_dist=10):
        """
        预测抓取点
        :param img: 输入图像 np.array (h, w, 3)
        :param thresh: 置信度阈值
        :param peak_dist: 峰值最小距离
        :return: pred_grasps, crop_x1, crop_y1
        """
        # 预处理
        input_tensor, crop_x1, crop_y1 = self.preprocess(img)
        input_tensor = input_tensor.to(self.device)
        
        # === 模拟推理结果 ===
        # 实际使用时，这里应该加载真实模型并进行前向传播
        # able_out, angle_out, width_out = model(input_tensor)
        
        # 这里生成一些模拟的抓取点用于演示
        np.random.seed(42)
        able_pred = np.random.rand(self.input_size, self.input_size) * 0.3
        able_pred[100:150, 100:150] = 0.7 + np.random.rand(50, 50) * 0.3
        able_pred[200:250, 200:250] = 0.6 + np.random.rand(50, 50) * 0.4
        
        angle_pred = np.random.rand(self.input_size, self.input_size) * self.angle_cls
        width_pred = np.random.rand(self.input_size, self.input_size) * 50 + 20
        
        # 后处理 - 提取峰值点
        pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
        
        # 构建抓取列表
        pred_grasps = []
        for idx in range(pred_pts.shape[0]):
            row, col = pred_pts[idx]
            angle = angle_pred[row, col] / self.angle_cls * 2 * np.pi
            width = width_pred[row, col]
            
            # 转换到原图坐标
            row += crop_y1
            col += crop_x1
            
            pred_grasps.append([row, col, angle, width])
        
        # 按置信度排序
        pred_grasps.sort(key=lambda x: able_pred[int(x[0]-crop_y1), int(x[1]-crop_x1)], reverse=True)
        
        # 只保留前 5 个
        pred_grasps = pred_grasps[:5]
        
        return pred_grasps, crop_x1, crop_y1


def run_demo(input_path, output_path, use_real_model=False, model_path=None):
    """
    运行推理 demo
    :param input_path: 输入图像文件夹
    :param output_path: 输出结果文件夹
    :param use_real_model: 是否使用真实模型
    :param model_path: 模型路径（如果使用真实模型）
    """
    # 检查设备
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f'>> 使用设备：{device_name}')
    
    # 初始化检测器
    if use_real_model and model_path:
        # TODO: 加载真实模型
        from utils.affga import AFFGA
        detector = AFFGA(model_path, device=device_name)
        predict_func = lambda img: detector.predict(img, device, mode='peak', thresh=0.5, peak_dist=2)
        print(f'>> 已加载真实模型：{model_path}')
    else:
        detector = SimpleGraspDetector(device=device_name)
        predict_func = lambda img: detector.predict(img, thresh=0.5, peak_dist=10)
        print('>> 使用模拟检测器（演示模式）')
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 处理所有图像
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f'警告：在 {input_path} 中未找到图像文件')
        return
    
    print(f'\n找到 {len(image_files)} 张图像，开始处理...\n')
    
    for file in image_files:
        print(f'processing {file}...')
        
        img_file = os.path.join(input_path, file)
        img = cv2.imread(img_file)
        
        if img is None:
            print(f'  无法读取图像：{img_file}')
            continue
        
        # 预测
        grasps, x1, y1 = predict_func(img)
        
        # 绘制结果
        im_rest = drawGrasps(img.copy(), grasps, mode='arrow')
        
        # 绘制裁剪区域
        rect = [x1, y1, x1 + 320, y1 + 320]
        im_rest = drawRect(im_rest, rect, color=(0, 255, 0), thickness=2)
        
        # 添加文字信息
        info_text = f'Grasps: {len(grasps)}'
        cv2.putText(im_rest, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 保存结果
        save_file = os.path.join(output_path, file)
        cv2.imwrite(save_file, im_rest)
        print(f'  检测到 {len(grasps)} 个抓取点 -> 保存到 {save_file}')
    
    print('\n完成！')
    
    # 如果有 fps 方法，打印 FPS
    if hasattr(detector, 'fps'):
        print(f'FPS: {detector.fps():.2f}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='抓取检测推理 Demo')
    parser.add_argument('--input', type=str, default='demo/input', 
                       help='输入图像文件夹')
    parser.add_argument('--output', type=str, default='demo/output', 
                       help='输出结果文件夹')
    parser.add_argument('--model', type=str, default=None, 
                       help='预训练模型路径（可选，不提供则使用模拟检测器）')
    parser.add_argument('--use_real', action='store_true', 
                       help='使用真实模型进行推理')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('抓取检测推理 Demo')
    print('=' * 60)
    
    run_demo(
        input_path=args.input,
        output_path=args.output,
        use_real_model=args.use_real,
        model_path=args.model
    )
