import datetime
import os
import time
import shutil
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from tqdm import tqdm
from hydra.utils import instantiate
from src.tool.utils import calculate_model_params


class LogFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def log(self, metrics):
        """将指标附加到日志文件"""
        try:
            with open(self.file_path, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"{timestamp} - "
                log_line += ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                f.write(log_line + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

@hydra.main(version_base=None, config_path="configs", config_name="gene_expansion")
def main(cfg: DictConfig):
    import torch
    print(f"支持的架构: {torch.cuda.get_arch_list()}")
    print(f"设备架构: {torch.cuda.get_device_capability(0)}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_MIRROR'] = 'https://hf-mirror.com'
    os.environ["HYDRA_FULL_ERROR"] = str(1)

    # 初始化Fabric (用于分布式训练)
    fabric = instantiate(cfg.machine.fabric)
    fabric.launch()

    # 设置随机种子
    seed_everything(cfg.train.seed)

    # 创建日志处理器 (仅在主进程中初始化)
    log_file_path = os.path.join(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"), "train_detail.log")
    file_logger = None
    if fabric.is_global_zero:
        file_logger = LogFileHandler(log_file_path)
        fabric.print(f"Logging to {log_file_path}")

    # 创建数据模块
    dataset = instantiate(cfg.data, _recursive_=False)

    train_dataloader = fabric.setup_dataloaders(dataset.train_dataloader())
    if cfg.is_val:
        val_dataloader = fabric.setup_dataloaders(dataset.val_dataloader())

    # 创建模型
    model = instantiate(cfg.model)
    calculate_model_params(model)

    optimizer = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )
    model, optimizer = fabric.setup(model, optimizer)

    scheduler = instantiate(cfg.scheduler)

    fabric.print("Start training")
    start_time = time.time()

    for epoch in range(cfg.train.max_epochs):
        scheduler(optimizer, epoch)

        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Epoch {epoch + 1}/{cfg.train.max_epochs}".center(columns))

        train(model, train_dataloader, optimizer, fabric, epoch, cfg, file_logger)

        if cfg.is_val:
            fabric.print("Val...")
            val(model, val_dataloader, fabric)

        if cfg.test:
            pass

        state = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        if cfg.trainer.save_ckpt == "all":
            fabric.save(f"ckpt_{epoch}.ckpt", state)
        elif cfg.trainer.save_ckpt == "last":
            fabric.save("ckpt_last.ckpt", state)

        fabric.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    fabric.print(f"Training time {total_time_str}")

    fabric.logger.finalize("success")
    fabric.print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def train(model, train_loader, optimizer, fabric, epoch, cfg, file_logger=None):
    model.train()

    if fabric.is_global_zero:
        pbar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            position=0
        )
    else:
        pbar = None

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx, fabric)
        fabric.backward(loss)
        optimizer.step()

        if pbar is not None:
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            pbar.update(1)

        if batch_idx % cfg.log.log_interval == 0:
            log_metrics = {
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "batch": batch_idx
            }

            # Fabric的标准日志
            fabric.log_dict(log_metrics)

            # 自定义文件日志 (仅在主进程中)
            if file_logger is not None and fabric.is_global_zero:
                file_logger.log(log_metrics)

    if pbar is not None:
        pbar.close()

def val(model, val_loader, fabric):
    model.eval()
    if fabric.is_global_zero:
        pbar = tqdm(
            total=len(val_loader),
            desc=f"Validation",
            dynamic_ncols=True,
            position=0
        )
    else:
        pbar = None

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx > 20:
            break
        mse = model.validation_step(batch, batch_idx, fabric)
        if pbar is not None:
            pbar.set_postfix({
                "mse": f"{mse:.2f}",
            })
            pbar.update(1)


    if pbar is not None:
        pbar.close()

if __name__ == "__main__":
    main()