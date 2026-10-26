import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# 根据你的实际文件结构调整导入路径
from cs336_basics.Model.transformer_lm import TransformerLM
from cs336_basics.Training.adamw import AdamW
from cs336_basics.Training.cross_entropy import cross_entropy
from cs336_basics.Training_Loop.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.Training_Loop.get_batch import get_batch
from cs336_basics.Training_Loop.gradient_clipping import gradient_clipping


# ---------- 1. 命令行参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Transformer Training")

    # 数据路径
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)

    # 模型超参数
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 日志和保存频率
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    return parser.parse_args()


# ---------- 2. 加载 Token 数据 ----------
def load_tokens(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
    else:
        # 假设原始二进制文件按 uint16 连续存储
        data = np.memmap(path, dtype=np.uint16, mode="r")

    if data.ndim != 1:
        raise ValueError("Expected a 1D token sequence")

    return data


# ---------- 3. 日志初始化 ----------
def setup_logging(checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(checkpoint_dir / "train.log"),
        ],
    )


# ---------- 4. 核心接口 ----------
def build_model(args) -> torch.nn.Module:
    return TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        eps=1e-5,
        theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=None,
    )

def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )
    
def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    data, target = get_batch(train_data, args.batch_size, args.context_length, device)
    optimizer.zero_grad()
    logits = model.forward(data)
    loss = cross_entropy(logits, target)
    loss.backward()
    gradient_clipping(model.parameters(), args.max_grad_norm)
    optimizer.step()
    return loss.item()

def evaluate(
    model: torch.nn.Module,
    val_data: np.ndarray, 
    args: argparse.Namespace,
    device: torch.device,
    num_eval_batches = 20
) -> float:
    model.eval()

    with torch.no_grad():
        lo = 0
        for _ in range(num_eval_batches):
            data, target = get_batch(val_data, args.batch_size, args.context_length, device)
        
            logits = model.forward(data)
            lo += cross_entropy(logits, target)
        
    model.train()
    return lo.item() / num_eval_batches


# ---------- 5. 训练主程序 ----------
def main():
    args = parse_args()
    setup_logging(args.checkpoint_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data)

    logging.info(f"Training tokens: {len(train_data)}")
    logging.info(f"Validation tokens: {len(val_data)}")
    logging.info(f"Device: {device}")

    model = build_model(args).to(device)
    optimizer = build_optimizer(model, args)

    start_step = 0

    # 恢复训练状态
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        logging.info(f"Resumed from step {start_step}")

    for step in range(start_step, args.max_steps):
        train_loss = train_one_step(
            model, optimizer, train_data, args, device, step
        )

        iteration = step + 1  # 已经完成的优化器更新次数

        # 训练日志
        if iteration % args.log_every == 0:
            logging.info(
                f"Step {iteration}/{args.max_steps} | "
                f"Train Loss: {train_loss:.4f}"
            )

        # 验证
        if iteration % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args, device)
            logging.info(
                f"Step {iteration} | Val Loss: {val_loss:.4f}"
            )

        # 定期保存 Checkpoint
        if iteration % args.save_every == 0:
            path = args.checkpoint_dir / f"step_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, path)

    # 保存最终模型
    save_checkpoint(
        model,
        optimizer,
        args.max_steps,
        args.checkpoint_dir / "final.pt",
    )


if __name__ == "__main__":
    main()
