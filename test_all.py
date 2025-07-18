import datetime
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from hydra.utils import instantiate
import torch
from hydra import initialize, compose
import sys  # 新增导入sys模块


class LogFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def log(self, metrics):
        try:
            with open(self.file_path, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"{timestamp} - " + ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                f.write(log_line + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="gene_expansion_test_m")
def main(cfg: DictConfig):
    # 新增：检查是否在循环测试模式下
    test_names = [f"test{i}" for i in range(2, 8)]  # test1到test7
    seed = cfg.test.seed
    epoch = cfg.test.epoch
    result_path = cfg.test.result_path
    solver = cfg.test.solver

    all_results = {}  # 存储所有测试结果

    for test_name in test_names:
        print(f"测试数据集为： {test_name}")
        # 动态更新配置中的测试集
        test_data_config = OmegaConf.load(f"{cfg.machine.work_dir}/configs/test/{test_name}.yaml")

        # 直接覆盖整个节点
        OmegaConf.set_struct(cfg, False)
        cfg.test = test_data_config
        cfg.test.seed = seed
        cfg.test.epoch = epoch
        cfg.test.result_path = result_path
        cfg.test.solver = solver
        OmegaConf.set_struct(cfg, True)

        # 环境设置（原代码保持不变）
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_MIRROR'] = 'https://hf-mirror.com'
        os.environ["HYDRA_FULL_ERROR"] = str(1)

        # 初始化Fabric
        fabric = instantiate(cfg.machine.fabric)
        fabric.launch()
        fabric.print(f"🏁 开始测试 {test_name} (设备: {fabric.device})")

        # 设置随机种子
        seed_everything(cfg.test.seed)

        # 日志初始化 - 使用当前测试集名称
        log_file = os.path.join("test", test_name, datetime.datetime.now().strftime("%m-%d-%H-%M-%S") + ".log")
        file_logger = None
        if fabric.is_global_zero:
            file_logger = LogFileHandler(log_file)
            fabric.print(f"📝 日志保存到 {log_file}")

        # 准备测试数据
        dataset = instantiate(cfg.test, _recursive_=False)
        test_loader = fabric.setup_dataloaders(dataset.test_dataloader())
        fabric.print(f"📊 测试集大小: {len(test_loader.dataset)} 样本")

        # 构建模型
        model = instantiate(cfg.model)
        model = fabric.setup(model)
        model.scheduler = cfg.test.solver
        fabric.print(f"采样方式: {model.scheduler}")

        # 加载检查点
        target_epoch = cfg.test.epoch
        ckpt_path = f"epoch_{target_epoch}.ckpt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"检查点不存在: {ckpt_path}")
        fabric.print(f"⬇️ 加载检查点: {ckpt_path}")
        fabric.load(ckpt_path, {"model": model})

        # 运行测试
        fabric.print("🔍 开始模型评估...")
        results = test(model, test_loader, fabric, cfg, file_logger)

        # 保存当前测试结果
        all_results[test_name] = results
        fabric.print(f"\n⭐ {test_name} 结果:")
        for metric, value in results.items():
            fabric.print(f"  {metric.upper()}: {value:.6f}")

        # 打印当前累计所有结果
        fabric.print("\n" + "=" * 30 + " 当前累计结果 " + "=" * 30)
        for test_name, res in all_results.items():
            fabric.print(f"🔬 {test_name}: " + ", ".join([f"{k}={v:.4f}" for k, v in res.items()]))
        fabric.print("=" * 70 + "\n")

        # 清理资源
        del model, test_loader, dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 最终报告
    fabric.print("\n" + "🔥🔥🔥 所有测试完成! 最终结果 🔥🔥🔥")
    for test_name, res in all_results.items():
        fabric.print(f"🎯 {test_name}: " + ", ".join([f"{k}={v:.4f}" for k, v in res.items()]))


def test(model, test_loader, fabric, cfg, file_logger=None):
    model.eval()
    total_metrics = {}
    count = 0

    if fabric.is_global_zero:
        pbar = tqdm(total=len(test_loader), desc="Test: ", dynamic_ncols=True)
    else:
        pbar = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):

            if batch_idx > 100:
                break

            # 模型前向传播
            metrics = model.validation_step(batch, batch_idx, fabric, stage="test", solver=cfg.test.solver)
            # 累积指标
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v

            count += 1

            # 更新进度
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            # 记录批次结果
            if file_logger is not None and fabric.is_global_zero:
                log_data = {f"Test_{k}": v for k, v in metrics.items()}
                log_data["Batch"] = batch_idx
                file_logger.log(log_data)

    # 计算平均指标
    avg_metrics = {k: v / count for k, v in total_metrics.items()}

    if pbar is not None:
        pbar.close()

    return avg_metrics

# 修改入口点以支持循环模式
if __name__ == "__main__":
    main()