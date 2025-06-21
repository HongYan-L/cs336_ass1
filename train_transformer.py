from .adapters import *
from tqdm import tqdm
import random
import os
import wandb
import argparse
import torch
import time
import pickle
import pathlib
import numpy as np
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
MODULE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "module"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_pkl(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode(file, file_name):
    np.array(file, dtype=np.uint16).tofile(file_name)

def save_encode_stream(token_stream: Iterable[int], file_path: os.PathLike):
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)

def train_bpe_dataset(
    file_name: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    vocab_name: str, 
    merges_name: str
):
    start_time = time.time()
    traindata_path = DATA_PATH / file_name
    vocab, merges = run_train_bpe(
        input_path=traindata_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_pkl(vocab, DATA_PATH / vocab_name)
    save_pkl(merges, DATA_PATH / merges_name)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"执行时间: {minutes} 分 {seconds} 秒")

def Tokenizer_dataset(
    trainfile_name: str | os.PathLike, 
    validfile_name: str | os.PathLike, 
    trainencode_name: str | os.PathLike, 
    validencode_name: str | os.PathLike, 
    vocab_name: str | os.PathLike, 
    merges_name: str | os.PathLike, 
    special_tokens: list[str]
):
    start_time = time.time()
    trainfile_path = DATA_PATH / trainfile_name
    validfile_path = DATA_PATH / validfile_name
    trainencode_path = DATA_PATH / trainencode_name
    validencode_path = DATA_PATH / validencode_name
    tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)

    # 处理训练集（流式编码）
    with open(trainfile_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    # total_bytes = sum(len(line.encode('utf-8')) for line in train_lines)
    # start_time = time.time()

    encode_stream = tokenizer.encode_iterable(train_lines)
    # token_list = list(encode_stream)
    # total_tokens = len(token_list)

    # 计算 tokenizer 压缩比
    # compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    # print(f"Total bytes: {total_bytes}")
    # print(f"Total tokens: {total_tokens}")
    # print(f"Compression ratio (bytes/token): {compression_ratio:.4f}")

    save_encode_stream(encode_stream, trainencode_path)

    # end_time = time.time()
    # elapsed = end_time - start_time
    # 计算 tokenizer 的速度
    # throughput = total_bytes / elapsed / (1024 ** 2)  # MB/s
    # print(f"[Tokenizer Benchmark] Encoded {total_bytes / (1024 ** 3):.2f} GB in {elapsed:.2f}s")
    # print(f"[Tokenizer Benchmark] Throughput: {throughput:.2f} MB/s")

    # 处理验证集（流式编码）
    with open(validfile_path, 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()
    encode_stream = tokenizer.encode_iterable(valid_lines)
    save_encode_stream(encode_stream, validencode_path)

def activation_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f" NaN or Inf in output of {name}")
            if output.abs().max() > 1e3:
                print(f" Large activation in {name}: max={output.abs().max().item():.2f}")
    return hook_fn

def get_param_grad_norm(model):
    total_param_norm = 0.0
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
    return total_param_norm ** 0.5, total_grad_norm ** 0.5

@torch.no_grad()
def evaluate_validloss(model, valid_dataset, batch_size, context_length, device):
    model.eval()
    losses = []
    total_batches = len(valid_dataset) // (batch_size * context_length)

    for i in range(total_batches):
        input_batch, target_batch = run_get_batch(valid_dataset, batch_size, context_length, device)
        logits = model(input_batch)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        losses.append(loss.item())

    model.train()  # 恢复训练状态
    return sum(losses) / len(losses)

def generate_sample_and_log(model, tokenizer, prompt_str, device, iteration, max_gen_tokens=256, temperature=1.0, top_p=0.95):
    model.eval()
    with torch.no_grad():
        prompt_ids = tokenizer.encode(prompt_str)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        eos_token_id = tokenizer.vocab_to_id.get("<|endoftext|>".encode('utf-8'), None)

        gen_ids = model.generate(
            input_tensor,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        full_ids = prompt_ids + gen_ids[0].tolist()
        output_text = tokenizer.decode(full_ids)

        print(f"[Sample @ Iter {iteration}] {output_text}")
        wandb.log({"sample/text": wandb.Html(f"<pre>{output_text}</pre>")})

    model.train()

def model_pipeline(hyperparameters=None):
    with wandb.init(config=hyperparameters):
        config = wandb.config
        set_seed(config.get("seed", 1337))

        model = Transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=n_layers,
            d_model=d_model,
            num_heads=n_heads,
            d_ff=d_ff,
            rope_theta=rope_theta
        ).to(device)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model = torch.compile(model, backend="aot_eager")
        optimizer = AdamW(model.parameters(), lr=config.lr)
        iteration = start_iter
        model.train()
        wandb.watch(model, log="all")

        # 注册 hook：只监控 Linear, LayerNorm, Embedding 这类常见爆炸点
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
                module.register_forward_hook(activation_hook(name))

        pbar = tqdm(total=total_iters)
        best_val_loss = float("inf")
        val_interval = total_iters // 20

        while iteration <= total_iters:
            input_train, target_train = run_get_batch(
                train_dataset, batch_size, context_length, device
            )
            logits = model(input_train)
            # 检测激活值是否为 NaN 或 Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[Iter {iteration}] NaN or Inf in logits!")
                break
            loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_train.view(-1))
            # 检测 loss 是否为 NaN 或 Inf
            if not torch.isfinite(loss):
                print(f"[Iter {iteration}] Loss is not finite: {loss.item()}")
                break

            lr = run_get_lr_cosine_schedule(
                iteration,
                max_learning_rate=config.lr,
                min_learning_rate=max(1e-6, config.lr * 0.01),
                warmup_iters=min(500, total_iters*0.1),
                cosine_cycle_iters=total_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            # 检查梯度是否爆炸
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f" NaN/Inf in gradient of {name}")
            # 追踪出梯度中出现 NaN 的操作来源
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            # loss.backward()
            run_gradient_clipping(model.parameters(), max_l2_norm)
            optimizer.step()
            if iteration % log_interval == 0 or iteration == total_iters:
                param_norm, grad_norm = get_param_grad_norm(model)
                print(f"[Iter {iteration}] loss: {loss.item():.4f}, param_norm: {param_norm:.2e}, grad_norm: {grad_norm:.2e}")
                wandb.log({
                    "train/loss": loss.item(),
                    "lr": lr,
                    "param_norm": param_norm,
                    "grad_norm": grad_norm,
                }, step=iteration)
                # print(f"[Iter {iteration}] loss: {loss.item():.4f}")
                # wandb.log({"train/loss": loss.item(), "lr": lr}, step=iteration)

            if iteration % val_interval == 0 or iteration == total_iters:
                val_loss = evaluate_validloss(model, valid_dataset, batch_size, context_length, device)
                print(f"[Iter {iteration}] Validation loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss}, step=iteration)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(MODULE_PATH)
                    print(f"Saved best model (val_loss={val_loss:.4f})")
                    wandb.run.summary["best_val_loss"] = best_val_loss

                # if val_loss < 1.45:
                #     print(f"[Early Stop] val_loss={val_loss:.4f} < 1.45, stopping early.")
                #     break

            iteration += 1
            pbar.update(1)

        wandb.finish()
    return model

if __name__ == '__main__':
    trainfile_name = 'owt_train.txt'
    validfile_name = 'owt_valid.txt'
    vocab_name = 'owt_vocab.pkl'
    merges_name = 'owt_merges.pkl'
    trainencode_name = 'owttrain_tokens.bin'
    validencode_name = 'owtvalid_tokens.bin'
    ckpt_name = 'owtcheckpoint.pt'
    vocab_size = 32000
    batch_size = 128
    context_length = 256
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    n_layers = 4
    n_heads = 16
    max_l2_norm = 1e-2
    max_gen_tokens = 256
    temperature = 0.8
    top_p = 0.95
    special_tokens = ["<|endoftext|>"]
    # train_bpe_dataset(trainfile_name, vocab_size, special_tokens, vocab_name, merges_name)
    # tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)
    Tokenizer_dataset(trainfile_name, validfile_name, trainencode_name, validencode_name, vocab_name, merges_name, special_tokens)
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = "mps"
    # print(f"using device: {device}")
    # train_dataset = np.memmap(DATA_PATH / trainencode_name, dtype=np.uint16, mode="r")
    # valid_dataset = np.memmap(DATA_PATH / validencode_name, dtype=np.uint16, mode="r")
    # start_iter = 1
    # total_iters = 10000
    # log_interval = total_iters // 200
    # ckpt_interval = total_iters // 20
    # print(f"Total iterations: {total_iters}")
    # 随机超参数搜索 lr
    # sweep_config = {
    #     "method": "random",
    #     "metric": {"name": "val/loss", "goal": "minimize"},
    #     "parameters": {
    #         "lr": {"distribution": "uniform", "min": 3e-3, "max": 5e-3}
    #     }
    # }
    # 网格超参数搜索 lr
    # sweep_config = {
    #     "method": "grid",
    #     "metric": {"name": "val/loss", "goal": "minimize"},
    #     "parameters": {
    #         "lr": {"values": [0.003]},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_config, project="cs336_ass1")
    # wandb.agent(sweep_id, model_pipeline)
    
