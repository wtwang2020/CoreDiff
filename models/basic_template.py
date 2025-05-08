# This part builds heavily on https://github.com/Hzzone/DU-GAN.
import torch
import os.path as osp
import tqdm
import argparse
import torch.distributed as dist
import torchvision
import os
import sys
sys.path.append('/data/wangweitao/i_am_yibo/mnt/sdb/i_am_yibo/project/欧核/主模型/OTF_nii')
from utils.dataset import get_mix_train_dataset
# from utils.pet_dataset import CTDataset

from utils.loggerx import LoggerX
from utils.sampler import RandomSampler
from utils.ops import load_network
import torch
from torch.utils.data import random_split, DataLoader
import torch
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, ProfilerActivity
from collections import OrderedDict
import logging
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basic_template')

def compute_and_print_model_metrics(model, inputs):
    """
    计算并打印模型的 FLOPS、参数量和 GPU 内存消耗。

    Args:
        model (nn.Module): 需要评估的模型
        inputs (torch.Tensor): 输入张量
    """
    model.eval()  # 切换到评估模式

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 计算 FLOPS
    flops = FlopCountAnalysis(model, inputs)
    print(f"FLOPS: {flops.total() / 1e9:.2f} GMac")

    # 前向传播以激活 GPU 内存
    with torch.no_grad():
        _ = model(inputs)

    # 统计 GPU 内存消耗
    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Current GPU Memory Usage: {current_memory:.2f} MB")
    print(f"Max GPU Memory Usage: {max_memory:.2f} MB")

class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        # 创建输出目录
        output_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), opt.output_dir, '{}_{}'.format(opt.model_name, opt.run_name))
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = LoggerX(save_root=output_dir)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(f"初始化训练任务 - 模型: {opt.model_name}, 运行名称: {opt.run_name}")
        
        # 设置数据加载器
        try:
            self.set_loader()
            logger.info("数据加载器设置成功")
        except Exception as e:
            logger.error(f"设置数据加载器时出错: {e}")
            raise
        
        # 设置模型
        try:
            self.set_model()
            logger.info("模型设置成功")
        except Exception as e:
            logger.error(f"设置模型时出错: {e}")
            raise

        # 初始化AMP混合精度训练的scaler
        self.use_amp = getattr(opt, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            print("启用自动混合精度训练(AMP)以加速计算")
        
        # 初始化性能指标收集
        self.data_load_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.update_time = 0.0
        self.total_time = 0.0
        self.iter_count = 0
        
        # 设置优先级以提高GPU利用率
        # if torch.cuda.is_available():
        #     # 设置CUDA计算流优先级
        #     torch.cuda.set_stream_priority(torch.cuda.current_stream(), priority=0)
        #     # 设置异步CUDA内存分配
        #     torch.cuda.set_per_process_memory_fraction(0.8)  # 限制内存使用率为80%以避免OOM
        
        # 初始化预取线程池（用于预取批次数据）
        # self.prefetch_pool = None
        # if getattr(opt, 'prefetch_next_batch', False) and torch.cuda.is_available():
        #     self.prefetch_pool = ThreadPoolExecutor(max_workers=1)
        #     self.next_batch = None
        #     self.prefetch_event = threading.Event()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('General arguments for CT projects')

        parser.add_argument('--save_freq', type=int, default=50000,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=20,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=20,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=30,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=15000000,
                            help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=11000000000000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true", help='启用Weights & Biases日志记录')

        # 添加评估频率和输出目录参数
        parser.add_argument('--eval_freq', type=int, default=50000,
                            help='模型评估频率（迭代次数）')
        parser.add_argument('--eval_output_dir', type=str, default='output/output_UI',
                            help='评估结果输出目录')
        parser.add_argument('--output_dir', type=str, default='output/output_UI',
                            help='模型输出主目录')

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='corediff',
                            help='the type of method')

        # training parameters for one-shot learning framework
        parser.add_argument("--osl_max_iter", type=int, default=3001,
                            help='number of training iterations for one-shot learning framework training')
        parser.add_argument("--osl_batch_size", type=int, default=8,
                            help='batch size for one-shot learning framework training')
        parser.add_argument("--index", type=int, default=10,
                            help='slice index selected for one-shot learning framework training')
        parser.add_argument("--unpair", action="store_true",
                            help='use unpaired data for one-shot learning framework training')
        parser.add_argument("--patch_size", type=int, default=512,
                            help='patch size used to divide the image')

        # dataset
        parser.add_argument('--train_dataset', type=str, default='train')
        parser.add_argument('--test_dataset', type=str, default='val')   # mayo_2020, piglte, phantom, mayo_2016
        parser.add_argument('--test_id', type=int, default=9,
                            help='test patient index for Mayo 2016')
        parser.add_argument('--context', action="store_true",
                            help='use contextual information')   #
        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--dose', type=int, default=25,
                            help='dose% data use for training and testing')

        # 添加数据集类型参数
        # parser.add_argument('--dataset_types', type=str, default="Mayo2016,Mayo2020_chest,Mayo2020_abdomen", help='要加载的数据集类型，用逗号分隔，例如"Bern_0,Bern_1,Bern_2"')
        parser.add_argument('--dataset_types', type=str, default="UI_0,UI_1,UI_2", help='要加载的数据集类型，用逗号分隔，例如"Bern_0,Bern_1,Bern_2"')
        # Add FP16 training argument
        parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度(AMP)训练以加速计算')

        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        """
        加载预训练模型权重
        
        Args:
            file_name: 预训练模型文件名
            
        Returns:
            加载的模型状态字典
        """
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        pretrained_path = osp.join(self.project_root, 'pretrained', file_name)
        
        if not osp.exists(pretrained_path):
            logger.error(f"预训练模型文件不存在: {pretrained_path}")
            raise FileNotFoundError(f"找不到预训练模型文件: {pretrained_path}")
        
        try:
            model_dict = load_network(pretrained_path)
            logger.info(f"成功加载预训练模型: {file_name}")
            return model_dict
        except Exception as e:
            logger.error(f"加载预训练模型时出错: {e}")
            raise

    def set_loader(self):
        opt = self.opt
        if opt.mode == 'train':
            logger.info(f"正在设置训练数据加载器，批量大小: {opt.batch_size}")
            
            # 根据参数选择加载哪些数据集
            train_dataset_types = opt.dataset_types.split(',') if hasattr(opt, 'dataset_types') and opt.dataset_types else None
            train_dataset = get_mix_train_dataset(dataset_types=train_dataset_types)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=opt.batch_size, 
                shuffle=True,
                pin_memory= True, 
                num_workers=opt.num_workers
            )
            self.train_loader = train_loader
            logger.info(f"训练数据加载器包含 {len(train_dataset)} 个样本")

    def fit(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                logger.info(f"从迭代 {opt.resume_iter} 恢复训练")
                try:
                    self.logger.load_checkpoints(opt.resume_iter)
                    logger.info(f"成功加载检查点: {opt.resume_iter}")
                except Exception as e:
                    logger.error(f"加载检查点时出错: {e}")
                    raise
            else:
                logger.info("从头开始训练")

            # training routine
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                try:
                    inputs = next(loader)
                except StopIteration:
                    # 重新初始化数据加载器
                    loader = iter(self.train_loader)
                    inputs = next(loader)
                
                try:
                    self.train(inputs, n_iter)
                except Exception as e:
                    logger.error(f"训练迭代 {n_iter} 发生错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                if n_iter % opt.save_freq == 0:
                    logger.info(f"保存检查点: {n_iter}")
                    self.logger.checkpoints(n_iter)

        elif opt.mode == 'test':
            logger.info(f"加载测试检查点: {opt.test_iter}")
            try:
                self.logger.load_test_checkpoints(opt.test_iter)
                logger.info(f"成功加载测试检查点: {opt.test_iter}")
            except Exception as e:
                logger.error(f"加载测试检查点时出错: {e}")
                raise
                
            logger.info("开始测试")
            self.test(opt.test_iter)
            logger.info("开始生成图像")
            self.generate_images(opt.test_iter)

        elif opt.mode == 'train_osl_framework':
            logger.info(f"加载 OSL 框架训练检查点: {opt.test_iter}")
            try:
                self.logger.load_test_checkpoints(opt.test_iter)
                logger.info(f"成功加载测试检查点: {opt.test_iter}")
            except Exception as e:
                logger.error(f"加载测试检查点时出错: {e}")
                raise
                
            logger.info("开始训练 OSL 框架")
            self.train_osl_framework(opt.test_iter)

        elif opt.mode == 'test_osl_framework':
            logger.info(f"加载 OSL 框架测试检查点: {opt.test_iter}")
            try:
                self.logger.load_test_checkpoints(opt.test_iter)
                logger.info(f"成功加载测试检查点: {opt.test_iter}")
            except Exception as e:
                logger.error(f"加载测试检查点时出错: {e}")
                raise
                
            logger.info("开始测试 OSL 框架")
            self.test_osl_framework(opt.test_iter)


    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        # 记录数据加载时间
        data_load_end = time.time()
        self.data_load_time += data_load_end - getattr(self, 'last_iter_end', data_load_end)
        
        # 训练模式
        self.model.train()
        
        optimizer = self.optimizer
        model = self.model
        
        # 获取输入数据
        if isinstance(inputs, dict):
            x, y = inputs['A'].cuda(), inputs['B'].cuda()
            semantic_info = inputs.get('semantic_info', None)
        else:
            x, y = inputs[0].cuda(), inputs[1].cuda()
            semantic_info = inputs[2] if len(inputs) > 2 else None
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 记录前向传播开始时间
        forward_start = time.time()
        
        # 使用混合精度训练
        if self.use_amp:
            with autocast():
                pred = model(x, y, n_iter)
                loss = self.compute_loss(pred, y)
            
            # 记录后向传播开始时间
            backward_start = time.time()
            self.forward_time += backward_start - forward_start
            
            # 使用scaler进行缩放和梯度计算
            self.scaler.scale(loss).backward()
            
            # 记录权重更新开始时间
            update_start = time.time()
            self.backward_time += update_start - backward_start
            
            # 使用scaler更新权重
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # 常规训练流程
            pred = model(x, y, n_iter)
            loss = self.compute_loss(pred, y)
            
            # 记录后向传播开始时间
            backward_start = time.time()
            self.forward_time += backward_start - forward_start
            
            # 计算梯度
            loss.backward()
            
            # 记录权重更新开始时间
            update_start = time.time()
            self.backward_time += update_start - backward_start
            
            # 更新权重
            optimizer.step()
        
        # 记录迭代结束时间
        iter_end = time.time()
        self.update_time += iter_end - update_start
        self.total_time += iter_end - data_load_end
        self.last_iter_end = iter_end
        self.iter_count += 1
        
        # 每100次迭代打印一次性能指标
        if n_iter % 100 == 0:
            avg_data_time = self.data_load_time / max(1, self.iter_count)
            avg_forward_time = self.forward_time / max(1, self.iter_count)
            avg_backward_time = self.backward_time / max(1, self.iter_count)
            avg_update_time = self.update_time / max(1, self.iter_count)
            avg_total_time = self.total_time / max(1, self.iter_count)
            
            print(f"性能统计 (平均每次迭代):")
            print(f"  数据加载时间: {avg_data_time:.4f}秒")
            print(f"  前向传播时间: {avg_forward_time:.4f}秒")
            print(f"  反向传播时间: {avg_backward_time:.4f}秒")
            print(f"  权重更新时间: {avg_update_time:.4f}秒")
            print(f"  总时间: {avg_total_time:.4f}秒")
            print(f"  吞吐量: {1.0/avg_total_time:.2f} 批次/秒")
            
            # 重置计数器
            if n_iter % 1000 == 0:
                self.data_load_time = 0.0
                self.forward_time = 0.0
                self.backward_time = 0.0
                self.update_time = 0.0
                self.total_time = 0.0
                self.iter_count = 0
        
        # 预取下一批数据（如果启用）
        if self.prefetch_pool and n_iter + 1 < self.opt.max_iter:
            self._prefetch_next_batch()
        
        return loss.item()
    
    def compute_loss(self, pred, target):
        """计算损失函数，可在子类中重写"""
        if hasattr(self, 'lossfn'):
            return self.lossfn(pred, target)
        else:
            return nn.MSELoss()(pred, target)
    
    def _prefetch_next_batch(self):
        """预取下一批数据以减少CPU-GPU之间的等待时间"""
        if self.prefetch_pool is None:
            return
        
        # 等待之前的预取完成
        if hasattr(self, 'prefetch_future') and self.prefetch_future is not None:
            self.next_batch = self.prefetch_future.result()
            self.prefetch_future = None
        
        # 提交新的预取任务
        def fetch_next():
            try:
                # 这里需要访问数据加载器获取下一批数据
                # 注意：实际实现需要根据具体的数据加载器结构调整
                for data in self.train_loader:
                    return data
            except Exception as e:
                print(f"预取数据出错: {e}")
                return None
        
        self.prefetch_future = self.prefetch_pool.submit(fetch_next)
        
    def fit_train(self):
        """模型训练流程"""
        opt = self.opt
        
        # 创建优化的数据加载器
        from utils.pet_dataset import CTDataset, create_optimized_dataloader
        
        # 使用优化过的数据集和数据加载器
        train_dataset = CTDataset(
            dataset=opt.train_dataset, 
            mode="train",
            use_cache=True,
            preload=getattr(opt, 'preload_data', False)
        )
        
        # 创建优化的训练数据加载器
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        
        self.train_loader = train_loader
        
        # 常规训练流程
        n_iter = opt.resume_iter
        max_iter = opt.max_iter
        
        print(f"开始训练，从迭代 {n_iter} 到 {max_iter}")
        
        # 预取第一个批次（如果启用）
        if self.prefetch_pool:
            self._prefetch_next_batch()
            
        # 主训练循环
        while n_iter < max_iter:
            # 如果有预取的批次，使用它
            if self.prefetch_pool and self.next_batch is not None:
                inputs = self.next_batch
                self.prefetch_future = None  # 清除引用
                self._prefetch_next_batch()  # 预取下一个批次
            else:
                # 常规方式获取批次
                for inputs in train_loader:
                    break
            
            loss = self.train(inputs, n_iter)
            
            if n_iter % 10 == 0:
                print(f'Iter: {n_iter}, Loss: {loss:.5f}')
            
            if n_iter % opt.save_freq == 0 and n_iter > 0:
                self.save(n_iter)
            
            if hasattr(self, 'step_ema'):
                self.step_ema(n_iter)
            
            n_iter += 1
            
            # 定期执行测试（如果需要）
            if n_iter % opt.test_freq == 0 and hasattr(self, 'test'):
                with torch.no_grad():
                    self.test(n_iter)
        
        # 保存最终模型
        self.save(n_iter)
        print(f"训练完成，共 {n_iter} 次迭代")

    @torch.no_grad()
    def test(self, n_iter):
        pass

    @torch.no_grad()
    def generate_images(self, n_iter):
        print("Entering generate_images")
        pass

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    # denormalize to [0, 255] for calculating PSNR, SSIM and RMSE
    def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
        return img

    def transfer_display_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = (img - cut_min) / (cut_max - cut_min)
        return img

