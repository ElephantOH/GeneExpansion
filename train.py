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

    # 创建数据模块
    dataset = instantiate(cfg.data, _recursive_=False)

    train_dataloader = fabric.setup_dataloaders(dataset.train_dataloader())
    if cfg.is_val:
        val_dataloader = fabric.setup_dataloaders(dataset.val_dataloader())

    # 创建模型
    model = instantiate(cfg.model)
    print(type(model))
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

        train(model, train_dataloader, optimizer, fabric, epoch, cfg)

        if cfg.is_val:
            fabric.print("Evaluate")
            # instantiate(cfg.evaluate, model, val_dataloader, fabric=fabric)

        if cfg.test:
            pass
            # for dataset in cfg.test:
            #     columns = shutil.get_terminal_size().columns
            #     fabric.print("-" * columns)
            #     fabric.print(f"Testing on {cfg.test[dataset].dataname}".center(columns))
            #
            #     data = instantiate(cfg.test[dataset])
            #     test_loader = fabric.setup_dataloaders(data.test_dataloader())
            #
            #     test = instantiate(cfg.test[dataset].test)
            #     test(model, test_loader, fabric=fabric)

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

    # for dataset in cfg.test:
    #     columns = shutil.get_terminal_size().columns
    #     fabric.print("-" * columns)
    #     fabric.print(f"Testing on {cfg.test[dataset].dataname}".center(columns))
    #
    #     data = instantiate(cfg.test[dataset])
    #     test_loader = fabric.setup_dataloaders(data.test_dataloader())
    #
    #     test = instantiate(cfg.test[dataset].test)
    #     test(model, test_loader, fabric=fabric)

    fabric.logger.finalize("success")
    fabric.print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def train(model, train_loader, optimizer, fabric, epoch, cfg):
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
            fabric.log_dict({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

    if pbar is not None:
        pbar.close()

if __name__ == "__main__":
    main()