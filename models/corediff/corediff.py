import os.path as osp
from torch.nn import functional as F
import torch
import torchvision
import argparse
import tqdm
import copy
import pydicom
from pydicom.dataset import Dataset, FileDataset
from utils.measure import *
from utils.measure_ct import compute_measure
from utils.loss_function import PerceptualLoss
from utils.ema import EMA
import os
import torchvision
import wandb
from models.basic_template import TrainTask
import csv
# corediff
from .corediff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion
import torch.nn as nn
# RED_CNN
# from .model.RED_CNN import RED_CNN
import torch
from torch.utils.data import random_split, DataLoader
import torch
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, ProfilerActivity
from collections import OrderedDict
import cv2
from collections import defaultdict
from utils.dataset_mayo import Mayo2016Dataset, Mayo2020Dataset
# 更改导入方式，使用新的工厂函数
from utils.dataset import get_mix_test_dataset
# from utils.dataset import CTDataset, Bern_Dataset, UI_Dataset
import clip
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import concurrent.futures  # 导入并发处理模块
import sys
sys.path.append(
    "/data/wangweitao/i_am_yibo/mnt/sdb/i_am_yibo/project/欧核/主模型/OTF_nii/utils"
)

# 将相对导入改为绝对导入
from common import dataIO, transformData
from evaluation_metric import compute_measure_pet
from dataset_UI_Bern import Test_Data  # 保留这个导入，因为后面可能需要用
io = dataIO()
transform = transformData()
import matplotlib.pyplot as plt


def transfer_calculate_window(
    img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
    img = img * (MAX_B - MIN_B) + MIN_B
    img[img < cut_min] = cut_min
    img[img > cut_max] = cut_max
    img = 255 * (img - cut_min) / (cut_max - cut_min)
    return img

def calculate_rmse(img1, img2):
    """计算均方根误差 RMSE
    
    Args:
        img1: 参考图像
        img2: 测试图像
    
    Returns:
        RMSE值，值越小越好
    """
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def compute_and_print_model_metrics(model, inputs, y=None, n_iter=0):
    """
    计算并打印模型的 FLOPS、参数量和 GPU 内存消耗。

    Args:
        model (nn.Module): 需要评估的模型
        inputs (torch.Tensor): 输入张量
        y (torch.Tensor): 额外的输入，默认为 None
        n_iter (int): 当前迭代次数，默认为 0
    """
    model.eval()  # 切换到评估模式

    # 包装模型以适应 FlopCountAnalysis
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, y, n_iter):
            super().__init__()
            self.model = model
            self.y = y
            self.n_iter = n_iter

        def forward(self, x):
            # 将封装参数传递到原始模型
            return self.model(x, self.y, self.n_iter)

    wrapped_model = ModelWrapper(model, y, n_iter)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 计算 FLOPS
    flops = FlopCountAnalysis(wrapped_model, inputs)
    print(f"FLOPS: {flops.total() / 1e9:.2f} GMac")

    # 前向传播以激活 GPU 内存
    with torch.no_grad():
        _ = wrapped_model(inputs)

    # 统计 GPU 内存消耗
    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Current GPU Memory Usage: {current_memory:.2f} MB")
    print(f"Max GPU Memory Usage: {max_memory:.2f} MB")


class corediff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser(
            'Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)

        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)
        
        # 添加数据集类型参数
        parser.add_argument('--dataset_types', type=str, default="Bern_0,Bern_1,Bern_2", 
                           help='要加载的数据集类型，用逗号分隔，例如"Bern_0,Bern_1,Bern_2"')

        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context

        denoise_fn = Network(in_channels=opt.in_channels, context=opt.context)

        model = Diffusion(
            denoise_fn=denoise_fn,
            image_size=512,
            timesteps=opt.T,
            context=opt.context
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()

        self.reset_parameters()

        # 添加CLIP模型初始化
        # self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        # self.clip_model = self.clip_model.eval()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, inputs, n_iter):
        opt = self.opt  # 确保在函数开始处定义 opt 变量
        
        # 初始化数据集和加载器（若未初始化）
        if not hasattr(self, 'test_loaders'):
            print("正在初始化测试数据集和加载器...")
            self.test_loaders = {}
            
            # 获取dataset_types参数
            dataset_types = opt.dataset_types.split(',') if hasattr(opt, 'dataset_types') and opt.dataset_types else None
            
            # 映射关系：数据集类型 -> 数据加载器键名
            dataset_loaders_map = {
                "UI_0": "UI_plane_0",
                "UI_1": "UI_plane_1", 
                "UI_2": "UI_plane_2",
                "Bern_0": "Bern_plane_0", 
                "Bern_1": "Bern_plane_1", 
                "Bern_2": "Bern_plane_2",
                "Mayo2016": "Mayo2016",
                "Mayo2020_chest": "Mayo2020_chest",
                "Mayo2020_abdomen": "Mayo2020_abdomen"
            }
            
            # 过滤要加载的数据集类型
            if dataset_types is None:
                # 如果未指定，使用默认的Bern数据集
                dataset_types = ["Bern_0", "Bern_1", "Bern_2"]
            
            print(f"将加载以下测试数据集: {dataset_types}")
            
            # 只加载指定的数据集
            for dataset_type in dataset_types:
                if dataset_type == "UI_0":
                    ui_test_dataset = get_mix_test_dataset(dataset_types=["UI_0"], test_samples="0:3")
                    self.test_loaders['UI_plane_0'] = DataLoader(
                        ui_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载UI_plane_0测试数据集，大小: {len(ui_test_dataset)}")
                    
                elif dataset_type == "UI_1":
                    ui_test_dataset = get_mix_test_dataset(dataset_types=["UI_1"], test_samples="0:3")
                    self.test_loaders['UI_plane_1'] = DataLoader(
                        ui_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载UI_plane_1测试数据集，大小: {len(ui_test_dataset)}")
                    
                elif dataset_type == "UI_2":
                    ui_test_dataset = get_mix_test_dataset(dataset_types=["UI_2"], test_samples="0:3")
                    self.test_loaders['UI_plane_2'] = DataLoader(
                        ui_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载UI_plane_2测试数据集，大小: {len(ui_test_dataset)}")
                    
                elif dataset_type == "Bern_0":
                    bern_test_dataset = get_mix_test_dataset(dataset_types=["Bern_0"], test_samples="0:3")
                    self.test_loaders['Bern_plane_0'] = DataLoader(
                        bern_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Bern_plane_0测试数据集，大小: {len(bern_test_dataset)}")
                    
                elif dataset_type == "Bern_1":
                    bern_test_dataset = get_mix_test_dataset(dataset_types=["Bern_1"], test_samples="0:3")
                    self.test_loaders['Bern_plane_1'] = DataLoader(
                        bern_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Bern_plane_1测试数据集，大小: {len(bern_test_dataset)}")
                    
                elif dataset_type == "Bern_2":
                    bern_test_dataset = get_mix_test_dataset(dataset_types=["Bern_2"], test_samples="0:3")
                    self.test_loaders['Bern_plane_2'] = DataLoader(
                        bern_test_dataset, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Bern_plane_2测试数据集，大小: {len(bern_test_dataset)}")
                    
                elif dataset_type == "Mayo2016":
                    Mayo2016_dataset_test = Mayo2016Dataset(mode="val", test_id=9, dose=25, body_part="abdomen",
                                                          context=False, patch_size=None, patch_n=None, debug=False)
                    self.test_loaders['Mayo2016'] = DataLoader(
                        Mayo2016_dataset_test, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Mayo2016测试数据集，大小: {len(Mayo2016_dataset_test)}")
                    
                elif dataset_type == "Mayo2020_chest":
                    Mayo2020_dataset_chest_test = Mayo2020Dataset(mode="test", test_id=None, dose=10, body_part="chest",
                                                                context=False, patch_size=None, patch_n=None, debug=False)
                    self.test_loaders['Mayo2020_chest'] = DataLoader(
                        Mayo2020_dataset_chest_test, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Mayo2020_chest测试数据集，大小: {len(Mayo2020_dataset_chest_test)}")
                    
                elif dataset_type == "Mayo2020_abdomen":
                    Mayo2020_dataset_abdomen_test = Mayo2020Dataset(mode="test", test_id=None, dose=10, body_part="abdomen",
                                                                  context=False, patch_size=None, patch_n=None, debug=False)
                    self.test_loaders['Mayo2020_abdomen'] = DataLoader(
                        Mayo2020_dataset_abdomen_test, batch_size=opt.test_batch_size, shuffle=False, 
                        pin_memory=True, num_workers=5)
                    print(f"已加载Mayo2020_abdomen测试数据集，大小: {len(Mayo2020_dataset_abdomen_test)}")
            
            print(f"测试数据集和加载器初始化完成，共加载 {len(self.test_loaders)} 个数据集")
        
        self.model.train()
        self.ema_model.train()
        # low_dose, full_dose, _, _ = inputs
        low_dose = inputs['input']
        full_dose = inputs['target']
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
        text_prompt = inputs['semantic_info']
            
        # 训练过程
        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1 = self.model(
            low_dose, full_dose, n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter,
        )
        
        loss = 0.5 * self.lossfn(gen_full_dose,
                                 full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1,
                                                                     full_dose)

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        self.logger.msg([loss, lr], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

        # 完整评估 - 只对指定数据集进行测试
        if n_iter % opt.eval_freq == 0:
            print(f"\n[{n_iter}] 执行模型评估...")
            base_output = opt.eval_output_dir
            
            # 获取之前初始化的测试加载器
            if hasattr(self, 'test_loaders'):
                # 对每个已加载的测试数据集进行评估
                for loader_name, loader in self.test_loaders.items():
                    # 根据加载器名称确定数据集类型和平面
                    if loader_name.startswith('UI_plane_'):
                        plane = loader_name.replace('UI_plane_', 'plane_')
                        dataset_source = 'UI'
                        print(f"评估 UI 数据集 {plane}...")
                        self.test_PET(
                            loader,
                            n_iter=n_iter,
                            base_output=base_output,
                            plane=plane,
                            my_dataset_source=dataset_source
                        )
                    elif loader_name.startswith('Bern_plane_'):
                        plane = loader_name.replace('Bern_plane_', 'plane_')
                        dataset_source = 'Bern'
                        print(f"评估 Bern 数据集 {plane}...")
                        self.test_PET(
                            loader,
                            n_iter=n_iter,
                            base_output=base_output,
                            plane=plane,
                            my_dataset_source=dataset_source
                        )
                    elif loader_name == 'Mayo2016' or loader_name.startswith('Mayo2020_'):
                        print(f"评估 {loader_name} 数据集...")
                        self.test_CT(
                            loader,
                            n_iter=n_iter,
                            base_output=base_output,
                            plane='plane_1'  # CT数据集使用plane_1
                        )
            else:
                print("警告：未找到任何测试数据加载器，跳过评估")

    @torch.no_grad()
    def resize_back(self, img, original_size):
        """
        将图像调整回原始大小
        Args:
            img: 输入图像张量
            original_size: 原始图像大小的元组 (height, width)
        Returns:
            调整大小后的图像张量
        """
        # 确保original_size是一个元组或列表
        if isinstance(original_size, torch.Tensor):
            original_size = (original_size[0].item(), original_size[1].item())
        elif not isinstance(original_size, (tuple, list)):
            raise ValueError(f"original_size必须是元组或张量，而不是{type(original_size)}")

        try:
            # 确保输入张量的维度正确
            if len(img.shape) == 3:  # 如果是3D张量 (C,H,W)
                img = img.unsqueeze(0)  # 添加batch维度

            resized = F.interpolate(
                img,
                size=original_size,
                mode='bilinear',
                align_corners=False)

            if len(img.shape) == 3:  # 如果之前是3D张量
                resized = resized.squeeze(0)  # 移除batch维度

            return resized

        except Exception as e:
            print(f"Resize失败: {str(e)}")
            print(f"输入图像形状: {img.shape}")
            print(f"目标大小: {original_size}")
            raise e

    @torch.no_grad()
    def transfer_display_window(
            self, img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = (img - cut_min) / (cut_max - cut_min)
        return img

    @torch.no_grad()
    def test_PET(self, PET_loader, n_iter, base_output, plane, my_dataset_source):
        opt = self.opt
        self.ema_model.eval()
        base_dir = os.path.join(base_output, f'test_results_{n_iter}')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 初始化总指标
        psnr, ssim, rmse = 0., 0., 0.
        low_dose_psnr, low_dose_ssim, low_dose_rmse = 0., 0., 0.

        # 初始化结果列表，添加标题行
        results = [["Image_Index", "PSNR", "SSIM", "RMSE",
                    "LowDose_PSNR", "LowDose_SSIM", "LowDose_RMSE"]]

        # 计算数据集大小以正确累加指标
        dataset_size = len(PET_loader.dataset)
        total_count = 0  # 用于跟踪处理的样本总数

        # 添加进度条
        pbar = tqdm.tqdm(PET_loader, desc=f"PET测试 [{plane}]", ncols=100)
        for i, test_input in enumerate(pbar):
            # 处理图像并移至GPU
            low_dose = test_input['input'].cuda()
            full_dose = test_input['target'].cuda()
            text_prompt = test_input['semantic_info']
            
            # 获取当前批次大小
            current_batch_size = low_dose.shape[0]
            
            # 创建保存路径 - 移到循环内部
            # 确保所有路径组件都是字符串类型
            modality = test_input['modality']
            if isinstance(modality, list):
                modality = modality[0]
            
            # dataset_source = test_input['dataset_source']
            # if isinstance(dataset_source, list):
            #     dataset_source = dataset_source[0]
            
            dataset_source = my_dataset_source
                
            body_part = test_input['body_part']
            if isinstance(body_part, list):
                body_part = body_part[0]
                
            dose_level = test_input['dose_level']
            if isinstance(dose_level, list):
                dose_level = dose_level[0]
            
            # 使用模型生成图像
            with torch.no_grad():
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                    batch_size=current_batch_size,
                    img=low_dose,
                    t=self.T,
                    sampling_routine=self.sampling_routine,
                    n_iter=n_iter,
                    start_adjust_iter=opt.start_adjust_iter,
                )

            # 将图像转回CPU
            gen_full_dose = gen_full_dose.detach().cpu()
            full_dose = full_dose.detach().cpu()
            low_dose = low_dose.detach().cpu()

            # 更新进度条信息
            pbar.set_postfix({"批次": f"{i+1}/{len(PET_loader)}", "样本": f"{total_count+current_batch_size}/{dataset_size}"})
            
            # 同步处理每个样本
            for b in range(current_batch_size):
                dose_level = test_input['dose_level'][b]
                sample_id = test_input['slice_idx'][b].item()
                dataset_dir = os.path.join(
                base_dir,
                modality,
                dataset_source,
                body_part,
                dose_level,
                plane)
                
                # 为当前样本创建保存路径
                low_dose_path = os.path.join(
                    dataset_dir, "low_dose_images", f"low_dose_{sample_id}.png")
                full_dose_path = os.path.join(
                    dataset_dir, "full_dose_images", f"full_dose_{sample_id}.png")
                gen_full_dose_path = os.path.join(
                    dataset_dir, "gen_full_dose_images", f"gen_full_dose_{sample_id}.png")
                os.makedirs(dataset_dir, exist_ok=True)
                os.makedirs(os.path.join(dataset_dir, "low_dose_images"), exist_ok=True)
                os.makedirs(os.path.join(dataset_dir, "full_dose_images"), exist_ok=True)
                os.makedirs(os.path.join(dataset_dir, "gen_full_dose_images"), exist_ok=True)
                # 尝试按照[heights, widths]的结构获取
                height = test_input['original_size'][0][b].item()  # 例如获取178
                width = test_input['original_size'][1][b].item()   # 例如获取673
                
                # 获取当前样本的图像
                current_low_dose = low_dose[b:b+1]
                current_full_dose = full_dose[b:b+1]
                current_gen_full_dose = gen_full_dose[b:b+1]
                
                try:
                    current_low_dose = torch.nn.functional.interpolate(current_low_dose, size=(height, width), mode='bilinear', align_corners=False)
                    current_full_dose = torch.nn.functional.interpolate(current_full_dose, size=(height, width), mode='bilinear', align_corners=False)
                    current_gen_full_dose = torch.nn.functional.interpolate(current_gen_full_dose, size=(height, width), mode='bilinear', align_corners=False)
                except Exception as e:
                    print(f"插值时出错: {e}，使用当前尺寸")
                    height, width = current_low_dose.shape[2], current_low_dose.shape[3]
                
                current_low_dose = transform.denormalize(img=current_low_dose, modality="PET")
                current_full_dose = transform.denormalize(img=current_full_dose, modality="PET")
                current_gen_full_dose = transform.denormalize(img=current_gen_full_dose, modality="PET")
                
                current_low_dose_truncated = transform.truncate_test(img=current_low_dose, modality="PET")
                current_full_dose_truncated = transform.truncate_test(img=current_full_dose, modality="PET")
                current_gen_full_dose_truncated = transform.truncate_test(img=current_gen_full_dose, modality="PET")
                
                data_range = current_full_dose_truncated.max() - current_full_dose_truncated.min()
                
                current_low_dose_np = current_low_dose.squeeze(0).squeeze(0).cpu().numpy()
                current_full_dose_np = current_full_dose.squeeze(0).squeeze(0).cpu().numpy()
                current_gen_full_dose_np = current_gen_full_dose.squeeze(0).squeeze(0).cpu().numpy()
                
                # 保存图像
                try:
                    plt.imsave(low_dose_path, 1 - current_low_dose_np, cmap="gray")
                    plt.imsave(full_dose_path, 1 - current_full_dose_np, cmap="gray")
                    plt.imsave(gen_full_dose_path, 1 - current_gen_full_dose_np, cmap="gray")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
                    continue

                # 将PyTorch张量转换为NumPy数组用于计算指标
                try:
                    psnr_score, ssim_score, rmse_score = compute_measure_pet(current_full_dose_truncated, current_gen_full_dose_truncated, data_range=data_range)
                    low_psnr_score, low_ssim_score, low_rmse_score = compute_measure_pet(current_full_dose_truncated, current_low_dose_truncated, data_range=data_range)
                    
                    # 累加指标
                    psnr += psnr_score / dataset_size
                    ssim += ssim_score / dataset_size
                    rmse += rmse_score / dataset_size
                    low_dose_psnr += low_psnr_score / dataset_size
                    low_dose_ssim += low_ssim_score / dataset_size
                    low_dose_rmse += low_rmse_score / dataset_size
                    
                    # 将当前样本的结果保存到结果列表
                    results.append([
                        sample_id,
                        psnr_score, ssim_score, rmse_score,
                        low_psnr_score, low_ssim_score, low_rmse_score
                    ])
                except Exception as e:
                    print(f"计算指标时出错: {e}")
                    continue
            
            # 更新处理的样本总数
            total_count += current_batch_size
            
            # 每处理10个批次保存一次部分结果
            if i % 10 == 0 and i > 0:
                partial_csv_path = os.path.join(dataset_dir, "metrics_results_partial.csv")
                with open(partial_csv_path, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(results)

        # 按索引排序结果
        results[1:] = sorted(results[1:], key=lambda x: x[0])
        
        # 添加平均指标到结果
        results.append([
            "Average",
            psnr, ssim, rmse,
            low_dose_psnr, low_dose_ssim, low_dose_rmse
        ])

        # 调试输出，确保最后一行为平均值
        print("results最后一行：", results[-1])

        # 保存最终结果到CSV（包含平均值）
        csv_file_path = os.path.join(dataset_dir, "metrics_results.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(results)

        # 打印数据的指标
        print(f"PET Dataset Results:")
        print(f"Average PSNR: {psnr:.4f}")
        print(f"Average SSIM: {ssim:.4f}")
        print(f"Average RMSE: {rmse:.4f}")
        print(f"Average Low_dose PSNR: {low_dose_psnr:.4f}")
        print(f"Average Low_dose SSIM: {low_dose_ssim:.4f}")
        print(f"Average Low_dose RMSE: {low_dose_rmse:.4f}")
        
        # 如果启用wandb，记录结果
        if opt.wandb:
            wandb.log({
                'epoch': n_iter,
                'PET/PSNR': psnr,
                'PET/SSIM': ssim,
                'PET/RMSE': rmse,
                'PET/LowDose_PSNR': low_dose_psnr,
                'PET/LowDose_SSIM': low_dose_ssim,
                'PET/LowDose_RMSE': low_dose_rmse
            })

        return {
            'psnr': psnr,
            'ssim': ssim,
            'rmse': rmse,
            'low_dose_psnr': low_dose_psnr,
            'low_dose_ssim': low_dose_ssim,
            'low_dose_rmse': low_dose_rmse
        }

    @torch.no_grad()
    def test_CT(self, CT_loader, n_iter, base_output, plane):
        opt = self.opt
        self.ema_model.eval()
        base_dir = os.path.join(base_output, f'test_results_{n_iter}')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 初始化总指标
        psnr, ssim, rmse = 0., 0., 0.
        low_dose_psnr, low_dose_ssim, low_dose_rmse = 0., 0., 0.

        # 初始化结果列表，添加标题行
        results = [["Image_Index", "PSNR", "SSIM", "RMSE",
                    "LowDose_PSNR", "LowDose_SSIM", "LowDose_RMSE"]]

        # 计算数据集大小以正确累加指标
        dataset_size = len(CT_loader.dataset)
        total_count = 0  # 用于跟踪处理的样本总数

        # 添加进度条
        pbar = tqdm.tqdm(CT_loader, desc=f"CT测试 [{plane}]", ncols=100)
        for i, test_input in enumerate(pbar):
            # 处理图像并移至GPU
            low_dose = test_input['input'].cuda()
            full_dose = test_input['target'].cuda()
            text_prompt = test_input['semantic_info']
            
            # 获取当前批次大小
            current_batch_size = low_dose.shape[0]
            
            # 创建保存路径 - 移到循环内部
            # 确保所有路径组件都是字符串类型
            modality = test_input['modality']
            if isinstance(modality, list):
                modality = modality[0]
            
            dataset_source = test_input['dataset_source']
            if isinstance(dataset_source, list):
                dataset_source = dataset_source[0]
                
            body_part = test_input['body_part']
            if isinstance(body_part, list):
                body_part = body_part[0]
                
            dose_level = test_input['dose_level']
            if isinstance(dose_level, list):
                dose_level = dose_level[0]
            
            # 使用字符串类型的参数创建路径
            dataset_dir = os.path.join(
                base_dir,
                modality,
                dataset_source,
                body_part,
                dose_level,
                plane)

            # 创建必要的目录
            os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(
                os.path.join(
                    dataset_dir,
                    "low_dose_images"),
                exist_ok=True)
            os.makedirs(
                os.path.join(
                    dataset_dir,
                    "full_dose_images"),
                exist_ok=True)
            os.makedirs(
                os.path.join(
                    dataset_dir,
                    "gen_full_dose_images"),
                exist_ok=True)
            
            
            # 兼容性处理：当original_size不存在时使用当前尺寸
            if 'original_size' in test_input:
                img_org_size = test_input['original_size']
            else:
                img_org_size = [(low_dose.shape[2], low_dose.shape[3])] * current_batch_size

            # 使用模型生成图像
            with torch.no_grad():
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                    batch_size=current_batch_size,
                    img=low_dose,
                    t=self.T,
                    sampling_routine=self.sampling_routine,
                    n_iter=n_iter,
                    start_adjust_iter=opt.start_adjust_iter,
                )

            # 将图像转回CPU
            gen_full_dose = gen_full_dose.detach().cpu()
            full_dose = full_dose.detach().cpu()
            low_dose = low_dose.detach().cpu()
            
            # 更新进度条信息
            pbar.set_postfix({"批次": f"{i+1}/{len(CT_loader)}", "样本": f"{total_count+current_batch_size}/{dataset_size}"})
    
            # 处理批量数据，为每个样本单独保存图像
            for b in range(current_batch_size):
                # 计算当前样本的全局索引
                sample_idx = total_count + b
                
                # 为当前样本创建保存路径
                low_dose_path = os.path.join(
                    dataset_dir, "low_dose_images", f"low_dose_{sample_idx}.png")
                full_dose_path = os.path.join(
                    dataset_dir, "full_dose_images", f"full_dose_{sample_idx}.png")
                gen_full_dose_path = os.path.join(
                    dataset_dir, "gen_full_dose_images", f"gen_full_dose_{sample_idx}.png")

                # 获取当前样本的图像
                current_low_dose = low_dose[b:b+1]
                current_full_dose = full_dose[b:b+1]
                current_gen_full_dose = gen_full_dose[b:b+1]
                current_body_part = test_input['body_part'][b:b+1][0]
                
                current_low_dose = current_low_dose.squeeze(0)
                current_full_dose = current_full_dose.squeeze(0)
                current_gen_full_dose = current_gen_full_dose.squeeze(0)
                
                # 根据不同身体部位使用不同的窗宽窗位
                if current_body_part == 'chest':
                    # 胸部窗口（肺窗）
                    current_low_dose_1 = self.transfer_display_window(current_low_dose, cut_min=-1350, cut_max=150)
                    current_full_dose_1 = self.transfer_display_window(current_full_dose, cut_min=-1350, cut_max=150)
                    current_gen_full_dose_1 = self.transfer_display_window(current_gen_full_dose, cut_min=-1350, cut_max=150)
        
                elif current_body_part == 'abdomen':
                    # 腹窗
                    current_low_dose_1 = self.transfer_display_window(current_low_dose, cut_min=-100, cut_max=200)
                    current_full_dose_1 = self.transfer_display_window(current_full_dose, cut_min=-100, cut_max=200)
                    current_gen_full_dose_1 = self.transfer_display_window(current_gen_full_dose, cut_min=-100, cut_max=200)

                elif current_body_part == 'neuro':
                    # 脑窗
                    current_low_dose_1 = self.transfer_display_window(current_low_dose, cut_min=40, cut_max=80)
                    current_full_dose_1 = self.transfer_display_window(current_full_dose, cut_min=40, cut_max=80)
                    current_gen_full_dose_1 = self.transfer_display_window(current_gen_full_dose, cut_min=40, cut_max=80)
                
                # 如果匹配不到任何部位，使用默认窗宽窗位（软组织窗）
                else:
                    #报错
                    raise ValueError(f"无法匹配到任何身体部位，使用默认窗宽窗位（软组织窗）")
                    
                torchvision.utils.save_image(
                    current_low_dose_1.clone().detach(), low_dose_path)
                torchvision.utils.save_image(
                    current_full_dose_1.clone().detach(), full_dose_path)
                torchvision.utils.save_image(
                    current_gen_full_dose_1.clone().detach(), gen_full_dose_path)

                current_full_dose = self.transfer_calculate_window(current_full_dose)
                current_gen_full_dose = self.transfer_calculate_window(current_gen_full_dose)
                current_low_dose = self.transfer_calculate_window(current_low_dose)
                
                data_range = current_full_dose.max() - current_full_dose.min()
                
                # 将PyTorch张量转换为NumPy数组用于计算指标
                current_full_dose_np = current_full_dose.cpu()
                current_gen_full_dose_np = current_gen_full_dose.cpu()
                current_low_dose_np = current_low_dose.cpu()
                
                # 使用NumPy数组计算指标
                psnr_score, ssim_score, rmse_score = compute_measure(current_full_dose_np, current_gen_full_dose_np, data_range.item())
                
                # 计算每个样本的指标
                low_psnr_score, low_ssim_score, low_rmse_score = compute_measure(current_full_dose_np, current_low_dose_np, data_range.item())
                
                # 累加计算平均值
                psnr += psnr_score / dataset_size
                ssim += ssim_score / dataset_size
                rmse += rmse_score / dataset_size
                low_dose_psnr += low_psnr_score / dataset_size
                low_dose_ssim += low_ssim_score / dataset_size
                low_dose_rmse += low_rmse_score / dataset_size
                
                # 将当前样本的结果保存到结果列表
                results.append([
                    sample_idx,
                    psnr_score, ssim_score, rmse_score,
                    low_psnr_score, low_ssim_score, low_rmse_score
                ])
            
            # 更新处理的样本总数
            total_count += current_batch_size
            
            # 每处理10个批次保存一次部分结果
            if i % 10 == 0 and i > 0:
                partial_csv_path = os.path.join(dataset_dir, "metrics_results_partial.csv")
                with open(partial_csv_path, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(results)

        # 按索引排序结果
        results[1:] = sorted(results[1:], key=lambda x: x[0])
        
        # 添加平均指标到结果
        results.append([
            "Average",
            psnr, ssim, rmse,
            low_dose_psnr, low_dose_ssim, low_dose_rmse
        ])

        # 调试输出，确保最后一行为平均值
        print("results最后一行：", results[-1])

        # 保存最终结果到CSV（包含平均值）
        csv_file_path = os.path.join(dataset_dir, "metrics_results.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(results)

        # 打印数据的指标
        print(f"CT Dataset Results:")
        print(f"Average PSNR: {psnr:.4f}")
        print(f"Average SSIM: {ssim:.4f}")
        print(f"Average RMSE: {rmse:.4f}")
        print(f"Average Low_dose PSNR: {low_dose_psnr:.4f}")
        print(f"Average Low_dose SSIM: {low_dose_ssim:.4f}")
        print(f"Average Low_dose RMSE: {low_dose_rmse:.4f}")
        
        # 如果启用wandb，记录结果
        if opt.wandb:
            wandb.log({
                'epoch': n_iter,
                'CT/PSNR': psnr,
                'CT/SSIM': ssim,
                'CT/RMSE': rmse,
                'CT/LowDose_PSNR': low_dose_psnr,
                'CT/LowDose_SSIM': low_dose_ssim,
                'CT/LowDose_RMSE': low_dose_rmse
            })

        return {
            'psnr': psnr,
            'ssim': ssim,
            'rmse': rmse,
            'low_dose_psnr': low_dose_psnr,
            'low_dose_ssim': low_dose_ssim,
            'low_dose_rmse': low_dose_rmse
        }

    def test_old(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        # 初始化总指标记录
        total_metrics = defaultdict(lambda: {
            'psnr': 0.0,
            'ssim': 0.0,
            'low_psnr': 0.0,
            'low_ssim': 0.0,
            'count': 0
        })

        # 定义需要测试的数据集配置
        test_datasets = {
            'Bern_0_50': {'dose': '0_50_dose', 'mode': 'val'},
            'Bern_0_4': {'dose': '0-4_dose', 'mode': 'val'},
            'Bern_0_20': {'dose': '0-20_dose', 'mode': 'val'},
            'Bern_0_10': {'dose': '0-10_dose', 'mode': 'val'},
            'UI_0_4': {'dose': '0-4_dose', 'mode': 'val'},
            'UI_0_10': {'dose': '0-10_dose', 'mode': 'val'},
            'UI_0_20': {'dose': '0-20_dose', 'mode': 'val'},
            'ct_mayo2016': {'dataset': 'mayo_2016', 'dose': 25, 'mode': 'val'},
            'ct_mayo2020': {'dataset': 'mayo_2020', 'dose': 25, 'mode': 'val'}
        }

        # 初始化总指标记录
        total_metrics = defaultdict(lambda: {
            'psnr': 0.0,
            'ssim': 0.0,
            'low_psnr': 0.0,
            'low_ssim': 0.0,
            'count': 0
        })

        # 创建总结果目录
        base_dir = f'./output/data_result_now/test_results_{n_iter}'
        os.makedirs(base_dir, exist_ok=True)

        for dataset_name, config in test_datasets.items():
            dataset_dir = os.path.join(base_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            if 'mayo' in dataset_name:
                # CT数据集指标初始化
                psnr, ssim, rmse = 0., 0., 0.
                low_dose_psnr, low_dose_ssim, low_dose_rmse = 0., 0., 0.

                # 初始化存储每次迭代结果的列表
                results = [["Image_Index",
                           "PSNR", "SSIM", "RMSE",
                            "LowDose_PSNR", "LowDose_SSIM", "LowDose_RMSE"]]
            else:
                # PET数据集指标
                psnr, ssim = 0., 0.
                low_dose_psnr, low_dose_ssim = 0., 0.

                # 初始化存储每次迭代结果的列表
                results = [["Image_Index",
                           "PSNR", "SSIM",
                            "LowDose_PSNR", "LowDose_SSIM"]]

            # 加载数据集
            if 'ct_' in dataset_name:
                test_dataset = CTDataset(
                    dataset=config.get('dataset'),
                    mode=config['mode'],
                    dose=config['dose']
                )
            else:
                DatasetClass = Bern_Dataset if 'Bern' in dataset_name else UI_Dataset
                test_dataset = DatasetClass(
                    mode=config['mode'],
                    dose=config['dose']
                )

            test_loader = DataLoader(
                test_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                num_workers=opt.num_workers
            )

            # 处理当前数据集
            if 'mayo' in dataset_name:
                for i, test_input in enumerate(test_loader):
                    # 处理图像并移至GPU
                    low_dose = test_input['input'].cuda()
                    full_dose = test_input['target'].cuda()
                    text_prompt = test_input['semantic_info']
                    with torch.no_grad():
                        # 正确使用CLIP的tokenize方法
                        text_tokens = clip.tokenize(
                            text_prompt, truncate=True).cuda()
                        text_features = self.clip_model.encode_text(
                            text_tokens).float()

                    # 兼容性处理：当original_size不存在时使用当前尺寸
                    if 'original_size' in test_input:
                        img_org_size = test_input['original_size']
                    else:
                        img_org_size = low_dose.shape[2:]  # 获取H,W维度

                    # 使用模型生成图像
                    with torch.no_grad():
                        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                            batch_size=low_dose.shape[0],
                            img=low_dose,
                            t=self.T,
                            sampling_routine=self.sampling_routine,
                            n_iter=n_iter,
                            start_adjust_iter=opt.start_adjust_iter,
                            text_features=text_features
                        )

                    # 将图像转回CPU
                    gen_full_dose = gen_full_dose.detach().cpu()
                    full_dose = full_dose.detach().cpu()
                    low_dose = low_dose.detach().cpu()
                    # 保存图像
                    low_dose_path = os.path.join(
                        dataset_dir, "low_dose_images", f"low_dose_{i}.png")
                    full_dose_path = os.path.join(
                        dataset_dir, "full_dose_images", f"full_dose_{i}.png")
                    gen_full_dose_path = os.path.join(
                        dataset_dir, "gen_full_dose_images", f"denoised_{i}.png")
                    os.makedirs(os.path.dirname(low_dose_path), exist_ok=True)
                    os.makedirs(os.path.dirname(full_dose_path), exist_ok=True)
                    os.makedirs(
                        os.path.dirname(gen_full_dose_path),
                        exist_ok=True)
                    low_dose = self.resize_back(low_dose, img_org_size)
                    full_dose = self.resize_back(full_dose, img_org_size)
                    gen_full_dose = self.resize_back(
                        gen_full_dose, img_org_size)

                    # CT数据使用transfer_display_window处理
                    torchvision.utils.save_image(
                        self.transfer_display_window(low_dose), low_dose_path)
                    torchvision.utils.save_image(
                        self.transfer_display_window(full_dose), full_dose_path)
                    torchvision.utils.save_image(
                        self.transfer_display_window(gen_full_dose), gen_full_dose_path)

                    # CT数据计算指标
                    data_range = full_dose.max() - full_dose.min()
                    psnr_score, ssim_score, rmse_score = compute_measure(
                        full_dose, gen_full_dose, data_range)
                    low_psnr_score, low_ssim_score, low_rmse_score = compute_measure(
                        full_dose, low_dose, data_range)

                    # 累加计算平均值
                    psnr += psnr_score / len(test_loader)
                    ssim += ssim_score / len(test_loader)
                    rmse += rmse_score / len(test_loader)
                    low_dose_psnr += low_psnr_score / len(test_loader)
                    low_dose_ssim += low_ssim_score / len(test_loader)
                    low_dose_rmse += low_rmse_score / len(test_loader)

                    # 将当前图像的结果保存到结果列表
                    results.append([
                        i,
                        psnr_score, ssim_score, rmse_score,
                        low_psnr_score, low_ssim_score, low_rmse_score
                    ])

                # CT数据保存结果到CSV
                csv_file_path = os.path.join(
                    dataset_dir, "metrics_results.csv")
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(results)

                # 打印CT数据的指标
                print(f"Dataset: {dataset_name}")
                print(f"PSNR: {psnr:.4f}")
                print(f"SSIM: {ssim:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"Low_dose PSNR: {low_dose_psnr:.4f}")
                print(f"Low_dose SSIM: {low_dose_ssim:.4f}")
                print(f"Low_dose RMSE: {low_dose_rmse:.4f}")

            else:
                for i, test_input in enumerate(test_loader):
                    # 处理图像并移至GPU
                    low_dose = test_input['input'].cuda()
                    full_dose = test_input['target'].cuda()
                    text_prompt = test_input['semantic_info']
                    with torch.no_grad():
                        # 正确使用CLIP的tokenize方法
                        text_tokens = clip.tokenize(
                            text_prompt, truncate=True).cuda()
                        text_features = self.clip_model.encode_text(
                            text_tokens).float()
                    # 兼容性处理：当original_size不存在时使用当前尺寸
                    if 'original_size' in test_input:
                        img_org_size = test_input['original_size']
                    else:
                        img_org_size = low_dose.shape[2:]  # 获取H,W维度

                    # 使用模型生成图像
                    with torch.no_grad():
                        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                            batch_size=low_dose.shape[0],
                            img=low_dose,
                            t=self.T,
                            sampling_routine=self.sampling_routine,
                            n_iter=n_iter,
                            start_adjust_iter=opt.start_adjust_iter,
                            text_features=text_features
                        )

                    # 将图像转回CPU
                    gen_full_dose = gen_full_dose.detach().cpu()
                    full_dose = full_dose.detach().cpu()
                    low_dose = low_dose.detach().cpu()

                    # 保存图像
                    low_dose_path = os.path.join(
                        dataset_dir, "low_dose_images", f"low_dose_{i}.png")
                    full_dose_path = os.path.join(
                        dataset_dir, "full_dose_images", f"full_dose_{i}.png")
                    gen_full_dose_path = os.path.join(
                        dataset_dir, "gen_full_dose_images", f"gen_full_dose_{i}.png")
                    os.makedirs(os.path.dirname(low_dose_path), exist_ok=True)
                    os.makedirs(os.path.dirname(full_dose_path), exist_ok=True)
                    os.makedirs(
                        os.path.dirname(gen_full_dose_path),
                        exist_ok=True)
                    low_dose = self.resize_back(low_dose, img_org_size)
                    full_dose = self.resize_back(full_dose, img_org_size)
                    gen_full_dose = self.resize_back(
                        gen_full_dose, img_org_size)
                    torchvision.utils.save_image(
                        low_dose.clone().detach(), low_dose_path)
                    torchvision.utils.save_image(
                        full_dose.clone().detach(), full_dose_path)
                    torchvision.utils.save_image(
                        gen_full_dose.clone().detach(), gen_full_dose_path)

                    # 去除前两个维度
                    low_dose_img = cv2.imread(
                        low_dose_path, cv2.IMREAD_GRAYSCALE)
                    full_dose_img = cv2.imread(
                        full_dose_path, cv2.IMREAD_GRAYSCALE)
                    gen_full_dose_img = cv2.imread(
                        gen_full_dose_path, cv2.IMREAD_GRAYSCALE)

                    # print("########################")
                    # print(f'low_dose_shape_{low_dose_img.shape}, full_dose_shape_{full_dose_img.shape}')
                    # print("########################")
                    # 计算生成图像的指标
                    psnr_score = calculate_psnr(
                        full_dose_img, gen_full_dose_img)
                    ssim_score = calculate_ssim(
                        full_dose_img, gen_full_dose_img)
                    low_psnr_score = calculate_psnr(
                        full_dose_img, low_dose_img)
                    low_ssim_score = calculate_ssim(
                        full_dose_img, low_dose_img)

                    # 累加计算平均值
                    psnr += psnr_score / len(test_loader)
                    ssim += ssim_score / len(test_loader)
                    low_dose_psnr += low_psnr_score / len(test_loader)
                    low_dose_ssim += low_ssim_score / len(test_loader)

                    # 将当前图像的结果保存到结果列表
                    results.append([
                        i,
                        psnr_score, ssim_score,
                        low_psnr_score, low_ssim_score
                    ])

                # PET数据保存结果到CSV
                csv_file_path = os.path.join(
                    dataset_dir, "metrics_results.csv")
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(results)

                # 打印PET数据的指标
                print(f"Dataset: {dataset_name}")
                print(f"PSNR: {psnr:.4f}")
                print(f"SSIM: {ssim:.4f}")
                print(f"Low_dose PSNR: {low_dose_psnr:.4f}")
                print(f"Low_dose SSIM: {low_dose_ssim:.4f}")

            # 更新总指标
            total_metrics[dataset_name].update({
                'psnr': psnr,
                'ssim': ssim,
                'low_psnr': low_dose_psnr,
                'low_ssim': low_dose_ssim
            })
            total_metrics['all']['psnr'] += psnr
            total_metrics['all']['ssim'] += ssim
            total_metrics['all']['low_psnr'] += low_dose_psnr
            total_metrics['all']['low_ssim'] += low_dose_ssim
            total_metrics['all']['count'] += 1

        # 输出并保存结果
        self.save_test_results(total_metrics, base_dir)
        if opt.wandb:
            self.log_to_wandb(total_metrics, n_iter)

    @torch.no_grad()
    def generate_images(self, n_iter):
        pass

    def train_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        ''' Initialize WeightNet '''
        weightnet = WeightNet(weight_num=10).cuda()
        optimizer_w = torch.optim.Adam(
            weightnet.parameters(), opt.init_lr * 10)
        lossfn = PerceptualLoss()

        ''' get imstep images of diffusion '''
        for i in range(len(self.test_dataset) - 2):
            if i == opt.index:
                if opt.unpair:
                    low_dose, _ = self.test_dataset[i]
                    _, full_dose = self.test_dataset[i + 2]
                else:
                    low_dose, full_dose = self.test_dataset[i]
        low_dose, full_dose = torch.from_numpy(low_dose).unsqueeze(0).cuda(), torch.from_numpy(full_dose).unsqueeze(
            0).cuda()

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            start_adjust_iter=opt.start_adjust_iter,
        )

        inputs = imstep_imgs.transpose(0, 2).squeeze(0)
        targets = full_dose

        ''' train WeightNet '''
        input_patches, target_patches = self.get_patch(
            inputs, targets, patch_size=opt.patch_size, stride=32)
        input_patches, target_patches = input_patches.detach(), target_patches.detach()

        for n_iter in tqdm.trange(1, opt.osl_max_iter):
            weightnet.train()
            batch_ids = torch.from_numpy(
                np.random.randint(
                    0,
                    input_patches.shape[0],
                    opt.osl_batch_size)).cuda()
            input = input_patches.index_select(dim=0, index=batch_ids).detach()
            target = target_patches.index_select(
                dim=0, index=batch_ids).detach()

            out, weights = weightnet(input)
            loss = lossfn(out, target)
            loss.backward()

            optimizer_w.step()
            optimizer_w.zero_grad()
            lr = optimizer_w.param_groups[0]['lr']
            self.logger.msg([loss, lr], n_iter)
            if opt.wandb:
                wandb.log({'epoch': n_iter, 'loss': loss})
        opt_image = weights * inputs
        opt_image = opt_image.sum(dim=1, keepdim=True)
        print(weights)

        ''' Calculate the quantitative metrics before and after weighting'''
        full_dose_cal = self.transfer_calculate_window(full_dose)
        gen_full_dose_cal = self.transfer_calculate_window(gen_full_dose)
        opt_image_cal = self.transfer_calculate_window(opt_image)
        data_range = full_dose_cal.max() - full_dose_cal.min()
        psnr_ori, ssim_ori, rmse_ori = compute_measure(
            full_dose_cal, gen_full_dose_cal, data_range)
        psnr_opt, ssim_opt, rmse_opt = compute_measure(
            full_dose_cal, opt_image_cal, data_range)
        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)

        fake_imgs = torch.cat((low_dose[:, 1].unsqueeze(
            1), full_dose, gen_full_dose, opt_image), dim=0)
        fake_imgs = self.transfer_display_window(fake_imgs)
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=4), test_iter,
                               'test_opt_' + opt.test_dataset + '_{}_{}'.format(self.dose, opt.index))

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + \
                '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + \
                '_{}_{}.npy'.format(self.dose, opt.index)
        np.save(filename, weights.detach().cpu().squeeze().numpy())

    def test_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + \
                '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + \
                '_{}_{}.npy'.format(self.dose, opt.index)
        weights = np.load(filename)
        print(weights)
        weights = torch.from_numpy(weights).unsqueeze(
            1).unsqueeze(2).unsqueeze(0).cuda()

        psnr_ori, ssim_ori, rmse_ori = 0., 0., 0.
        psnr_opt, ssim_opt, rmse_opt = 0., 0., 0.

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=test_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )
            imstep_imgs = imstep_imgs[:self.T]
            inputs = imstep_imgs.squeeze(2).transpose(0, 1)

            opt_image = weights * inputs
            opt_image = opt_image.sum(dim=1, keepdim=True)

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            opt_image = self.transfer_calculate_window(opt_image)

            data_range = full_dose.max() - full_dose.min()
            psnr_ori, ssim_ori, rmse_ori = compute_measure(
                full_dose, gen_full_dose, data_range)
            psnr_opt, ssim_opt, rmse_opt = compute_measure(
                full_dose, opt_image, data_range)

            psnr_ori += psnr_ori / len(self.test_loader)
            ssim_ori += ssim_ori / len(self.test_loader)
            rmse_ori += rmse_ori / len(self.test_loader)

            psnr_opt += psnr_opt / len(self.test_loader)
            ssim_opt += ssim_opt / len(self.test_loader)
            rmse_opt += rmse_opt / len(self.test_loader)

        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)

    def get_patch(self, input_img, target_img, patch_size=256, stride=32):
        input_patches = []
        target_patches = []
        _, c_input, h, w = input_img.shape
        _, c_target, h, w = target_img.shape

        Top = np.arange(0, h - patch_size + 1, stride)
        Left = np.arange(0, w - patch_size + 1, stride)
        for t_idx in range(len(Top)):
            top = Top[t_idx]
            for l_idx in range(len(Left)):
                left = Left[l_idx]
                input_patch = input_img[:, :, top:top +
                                        patch_size, left:left + patch_size]
                target_patch = target_img[:,
                                          :,
                                          top:top + patch_size,
                                          left:left + patch_size]
                input_patches.append(input_patch)
                target_patches.append(target_patch)

        input_patches = torch.stack(input_patches).transpose(
            0, 1).reshape((-1, c_input, patch_size, patch_size))
        target_patches = torch.stack(target_patches).transpose(
            0, 1).reshape((-1, c_target, patch_size, patch_size))
        return input_patches, target_patches

    def save_test_results(self, metrics, save_dir):
        """保存测试结果到CSV文件"""
        csv_path = os.path.join(save_dir, 'test_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'PSNR', 'SSIM',
                            'LowDose_PSNR', 'LowDose_SSIM'])

            for name, data in metrics.items():
                if name == 'all':
                    continue
                writer.writerow([
                    name,
                    f"{data['psnr']:.4f}",
                    f"{data['ssim']:.4f}",
                    f"{data['low_psnr']:.4f}",
                    f"{data['low_ssim']:.4f}"
                ])
            writer.writerow([
                'Average',
                f"{metrics['all']['psnr']:.4f}",
                f"{metrics['all']['ssim']:.4f}",
                f"{metrics['all']['low_psnr']:.4f}",
                f"{metrics['all']['low_ssim']:.4f}"
            ])

    def log_to_wandb(self, metrics, n_iter):
        """记录指标到WandB"""
        log_data = {'epoch': n_iter}

        for name, data in metrics.items():
            prefix = name + '/' if name != 'all' else ''
            log_data.update({
                f'{prefix}PSNR': data['psnr'],
                f'{prefix}SSIM': data['ssim'],
                f'{prefix}LowDose_PSNR': data['low_psnr'],
                f'{prefix}LowDose_SSIM': data['low_ssim']
            })

        wandb.log(log_data)

    @torch.no_grad()
    def test_quick(self, n_iter, base_output=None, sample_only=True, max_samples=5):
        """
        快速评估函数，可选择只生成样本或者计算有限样本的指标
        
        Args:
            n_iter: 当前训练迭代次数
            base_output: 输出目录
            sample_only: 是否只生成样本而不计算指标
            max_samples: 每个数据集最多处理的样本数量
        """
        opt = self.opt
        # 如果没有指定base_output，使用默认的eval_output_dir
        if base_output is None:
            base_output = opt.eval_output_dir
            
        self.ema_model.eval()
        base_dir = os.path.join(base_output, f'test_quick_{n_iter}')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # 选择一部分数据集进行快速测试
        test_items = [
            ('PET', self.test_loaders['UI_plane_1'], 'plane_1'),
            ('CT', self.test_loaders['Mayo2020_chest'], 'plane_1'),
        ]
        
        results = {}
        
        for data_type, loader, plane in test_items:
            # 创建子目录
            sub_dir = os.path.join(base_dir, f"{data_type}_{plane}")
            os.makedirs(sub_dir, exist_ok=True)
            
            # 创建图像保存目录
            img_dir = os.path.join(sub_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            
            # 初始化指标
            metrics = {'psnr': 0, 'ssim': 0, 'rmse': 0}
            
            # 获取少量样本
            limited_loader = []
            for i, batch in enumerate(loader):
                if i >= (max_samples // opt.test_batch_size + 1):
                    break
                limited_loader.append(batch)
            
            print(f"快速评估: {data_type} {plane} (样本数: {min(len(limited_loader) * opt.test_batch_size, max_samples)})")
            
            # 处理每个批次
            for i, test_input in enumerate(limited_loader):
                if i * opt.test_batch_size >= max_samples:
                    break
                    
                # 处理图像并移至GPU
                low_dose = test_input['input'].cuda()
                full_dose = test_input['target'].cuda() 
                text_prompt = test_input['semantic_info']
                
                # 获取当前批次大小
                current_batch_size = min(low_dose.shape[0], max_samples - i * opt.test_batch_size)
                
                # 如果批次大小需要调整
                if current_batch_size < low_dose.shape[0]:
                    low_dose = low_dose[:current_batch_size]
                    full_dose = full_dose[:current_batch_size]
                    if isinstance(text_prompt, list):
                        text_prompt = text_prompt[:current_batch_size]
                
                with torch.no_grad():
                    # CLIP编码
                    text_tokens = clip.tokenize(text_prompt, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_tokens).float()
                    
                    # 模型推理
                    gen_full_dose, _, _ = self.ema_model.sample(
                        batch_size=current_batch_size, 
                        img=low_dose,
                        t=self.T,
                        sampling_routine=self.sampling_routine,
                        n_iter=n_iter,
                        start_adjust_iter=opt.start_adjust_iter,
                        text_features=text_features
                    )
                
                # 拼接图像：[低剂量, 全剂量, 生成图像]
                grid_images = torch.cat([
                    low_dose.cpu()[:current_batch_size], 
                    full_dose.cpu()[:current_batch_size], 
                    gen_full_dose.cpu()[:current_batch_size]
                ], dim=0)
                
                # 保存图像网格
                grid_path = os.path.join(img_dir, f"batch_{i}.png")
                torchvision.utils.save_image(
                    grid_images, 
                    grid_path, 
                    nrow=current_batch_size, 
                    normalize=True
                )
                
                # 如果不仅仅是生成样本，还需要计算指标
                if not sample_only and data_type == 'CT':
                    # 对CT图像应用适当的窗口级别
                    for b in range(current_batch_size):
                        # 获取身体部位
                        current_body_part = test_input['body_part'][b:b+1][0] if isinstance(test_input['body_part'], list) else test_input['body_part'][0]
                        
                        # 处理窗口级别
                        current_low_dose = low_dose[b].cpu()
                        current_full_dose = full_dose[b].cpu()
                        current_gen_full_dose = gen_full_dose[b].cpu()
                        
                        # 应用计算窗口
                        current_full_dose = self.transfer_calculate_window(current_full_dose)
                        current_gen_full_dose = self.transfer_calculate_window(current_gen_full_dose)
                        current_low_dose = self.transfer_calculate_window(current_low_dose)
                        
                        # 转换为NumPy数组
                        current_full_dose_np = current_full_dose.cpu().numpy()
                        current_gen_full_dose_np = current_gen_full_dose.cpu().numpy()
                        
                        # 计算指标
                        data_range = current_full_dose.max() - current_full_dose.min()
                        batch_psnr, batch_ssim, batch_rmse = compute_measure(
                            current_full_dose_np, 
                            current_gen_full_dose_np, 
                            data_range.item()
                        )
                        
                        # 累加指标
                        metrics['psnr'] += batch_psnr / (current_batch_size * len(limited_loader))
                        metrics['ssim'] += batch_ssim / (current_batch_size * len(limited_loader))
                        metrics['rmse'] += batch_rmse / (current_batch_size * len(limited_loader))
                
                elif not sample_only and data_type == 'PET':
                    # 对PET图像计算指标
                    for b in range(current_batch_size):
                        # 获取样本
                        current_low_dose = low_dose[b].cpu()
                        current_full_dose = full_dose[b].cpu()
                        current_gen_full_dose = gen_full_dose[b].cpu()
                        
                        # 转换为NumPy数组
                        current_low_dose_np = current_low_dose.squeeze().numpy()
                        current_full_dose_np = current_full_dose.squeeze().numpy()
                        current_gen_full_dose_np = current_gen_full_dose.squeeze().numpy()
                        
                        # 计算指标
                        data_range = 1.0
                        psnr_score = calculate_psnr(current_full_dose_np, current_gen_full_dose_np, data_range=data_range)
                        ssim_score = calculate_ssim(current_full_dose_np, current_gen_full_dose_np, data_range=data_range)
                        rmse_score = calculate_rmse(current_full_dose_np, current_gen_full_dose_np)
                        
                        # 累加指标
                        metrics['psnr'] += psnr_score / (current_batch_size * len(limited_loader))
                        metrics['ssim'] += ssim_score / (current_batch_size * len(limited_loader))
                        metrics['rmse'] += rmse_score / (current_batch_size * len(limited_loader))
            
            # 保存指标结果
            if not sample_only:
                with open(os.path.join(sub_dir, 'metrics.txt'), 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value:.4f}\n")
                        print(f"{data_type} {plane} {metric}: {value:.4f}")
            
            results[f"{data_type}_{plane}"] = metrics
        
        return results


print(clip.available_models())  # 应该输出['RN50', 'RN101', 'RN50x4',...]
