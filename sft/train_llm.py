"""
Fine-tune Qwen3-0.6B on SID sequence prediction (LoRA) with wandb monitoring.

Pipeline
--------
1. Load Qwen3 tokenizer → add 384 SID special tokens
2. Load Qwen3-0.6B → resize embeddings → initialise new token rows
3. Wrap with LoRA (r=16, target: attn+MLP; modules_to_save: embed+lm_head)
4. Print 3 formatted prompt samples  ← verify before training
5. Init wandb run
6. Train with bf16 + gradient accumulation + cosine LR
7. Save LoRA adapter + extended tokenizer

Usage
-----
  python train_llm.py
  python train_llm.py --config conf/sft.conf
  python train_llm.py --resume
"""
import argparse
import configparser
import glob
import logging
import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline.llm_dataset import LLMSIDDataset, all_sid_tokens, collate_pad


# ── tokenizer / embedding helpers ─────────────────────────────────────────────

def extend_tokenizer_and_model(tokenizer, model, sid_tokens: list) -> int:
    """
    Add SID special tokens to the tokenizer and grow model embeddings to match.

    New token rows are initialised near the mean of existing embeddings (rather
    than random zeros) so the model starts in a reasonable neighbourhood.

    Returns the number of tokens actually added (0 if all already present).
    """
    n_before = len(tokenizer)
    tokenizer.add_tokens(sid_tokens, special_tokens=True)
    n_after  = len(tokenizer)
    n_added  = n_after - n_before

    if n_added == 0:
        return 0

    model.resize_token_embeddings(n_after)

    with torch.no_grad():
        in_emb  = model.get_input_embeddings().weight       # (vocab, D)
        mean_in = in_emb[:n_before].float().mean(dim=0)
        noise   = torch.randn(n_added, in_emb.shape[1],
                              device=in_emb.device) * 0.01
        in_emb[n_before:] = mean_in.to(in_emb.dtype) + noise.to(in_emb.dtype)

        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != in_emb.data_ptr():
            lm_w    = out_emb.weight
            mean_lm = lm_w[:n_before].float().mean(dim=0)
            noise2  = torch.randn(n_added, lm_w.shape[1],
                                  device=lm_w.device) * 0.01
            lm_w[n_before:] = mean_lm.to(lm_w.dtype) + noise2.to(lm_w.dtype)

    return n_added


# ── prompt inspection ─────────────────────────────────────────────────────────

def print_sample_prompts(dataset: LLMSIDDataset, tokenizer, n: int = 3):
    sep = '=' * 72
    print(f'\n{sep}')
    print(f'  SAMPLE PROMPTS — verifying tokeniser + format  (n={n})')
    print(sep)

    idxs = [int(i * (len(dataset) - 1) / max(n - 1, 1)) for i in range(n)]

    for rank, idx in enumerate(idxs):
        prompt_txt, next_str = dataset.sample_prompt_text(idx)

        p_ids        = tokenizer.encode(prompt_txt, add_special_tokens=False)
        n_ids        = tokenizer.encode(next_str,   add_special_tokens=False)
        decoded_next = [tokenizer.decode([t]) for t in n_ids]

        print(f'\n  ── Sample {rank + 1}  (index {idx}) ──')
        print(f'  Prompt tail (last 300 chars):')
        for line in prompt_txt[-300:].splitlines():
            print(f'    {line}')
        print(f'  Completion  : {next_str}')
        print(f'  Comp tokens : {decoded_next}  ← each must be one SID token')
        print(f'  Token count : prompt={len(p_ids)}  completion={len(n_ids)}'
              f'  total={len(p_ids) + len(n_ids)}')

    print(f'\n{sep}\n')


# ── wandb ─────────────────────────────────────────────────────────────────────

def init_wandb(cfg, run_config: dict, run_suffix: str = ''):
    """
    Initialise a wandb run using [wandb] config section.
    run_suffix (launch time) keeps different experiments distinguishable.
    Returns the run object (or None if disabled / import fails).
    """
    try:
        enabled = cfg.getboolean('wandb', 'enabled', fallback=False)
        if not enabled:
            return None

        import wandb
        api_key  = cfg.get('wandb', 'api_key',  fallback=None)
        project  = cfg.get('wandb', 'project',  fallback='stock-sid-sft')
        run_name = cfg.get('wandb', 'run_name',  fallback='qwen3-lora')
        if run_suffix:
            run_name = f'{run_name}_{run_suffix}'

        if api_key:
            wandb.login(key=api_key, relogin=False)

        run = wandb.init(
            project = project,
            name    = run_name,
            config  = run_config,
            resume  = 'allow',
        )
        return run
    except Exception as e:
        logging.getLogger(__name__).warning(f"wandb init failed: {e} — running without wandb")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-0.6B on SID prediction')
    parser.add_argument('--config', default='conf/sft.conf')
    parser.add_argument('--resume', action='store_true',
                        help='Load latest saved adapter before training')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    log_dir = cfg.get('paths', 'log_dir')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train_llm.log'), mode='a'),
        ],
    )
    log = logging.getLogger(__name__)

    # ── config ────────────────────────────────────────────────
    K           = cfg.getint('rqvae', 'num_codebooks')
    CS          = cfg.getint('rqvae', 'codebook_size')
    model_dir   = cfg.get('llm',   'model_dir')
    adapter_dir = cfg.get('llm',   'adapter_dir')
    lora_r      = cfg.getint('llm',   'lora_r')
    lora_alpha  = cfg.getint('llm',   'lora_alpha')
    lora_drop   = cfg.getfloat('llm', 'lora_dropout')
    batch_size  = cfg.getint('llm',   'batch_size')
    grad_accum  = cfg.getint('llm',   'grad_accum')
    epochs      = cfg.getint('llm',   'epochs')
    lr          = cfg.getfloat('llm', 'learning_rate')
    wd          = cfg.getfloat('llm', 'weight_decay')
    max_len     = cfg.getint('llm',   'max_seq_len')
    warmup_r    = cfg.getfloat('llm', 'warmup_ratio')

    os.makedirs(adapter_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Step 1 : tokenizer + SID special tokens ───────────────
    log.info(f"Loading tokenizer from {model_dir}")
    tokenizer  = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    sid_tokens = all_sid_tokens(K, CS)
    log.info(f"Adding {len(sid_tokens)} SID special tokens "
             f"(<a_0>..<a_{CS-1}>, <b_0>..<b_{CS-1}>, <c_0>..<c_{CS-1}>)")

    # ── Step 2 : model + resize ───────────────────────────────
    log.info(f"Loading model from {model_dir}  (bfloat16, CPU first)")
    model   = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    n_added = extend_tokenizer_and_model(tokenizer, model, sid_tokens)
    log.info(f"  Vocab: {len(tokenizer):,}  (+{n_added} SID tokens)")
    log.info(f"  tie_word_embeddings: {model.config.tie_word_embeddings}")

    model = model.to(device)
    torch.cuda.empty_cache()

    # ── Step 3 : LoRA ─────────────────────────────────────────
    tied = getattr(model.config, 'tie_word_embeddings', False)
    mts  = ['embed_tokens'] if tied else ['embed_tokens', 'lm_head']

    lora_cfg = LoraConfig(
        r               = lora_r,
        lora_alpha      = lora_alpha,
        target_modules  = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout    = lora_drop,
        bias            = 'none',
        task_type       = TaskType.CAUSAL_LM,
        modules_to_save = mts,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── run directory: one folder per training run ────────────
    # Fresh run  → adapter_dir/run_<launch-time>/   (epoch_* + step_* inside)
    # --resume   → reuse the latest existing run_* folder
    run_tag = time.strftime('%Y%m%d_%H%M%S')
    if args.resume:
        prev_runs = sorted(glob.glob(os.path.join(adapter_dir, 'run_*')))
        run_dir   = prev_runs[-1] if prev_runs else adapter_dir   # legacy fallback
        log.info(f"Resume mode → run dir: {run_dir}")
    else:
        run_dir = os.path.join(adapter_dir, f'run_{run_tag}')
        log.info(f"Fresh run → run dir: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    # ── resume: load weights + optimizer state ────────────────
    resume_epoch     = 0
    resume_batch_idx = 0   # how many batches to skip inside the first resumed epoch
    global_opt_step  = 0
    step_ckpts       = []
    epoch_ckpts      = []
    if args.resume:
        step_ckpts  = sorted(glob.glob(os.path.join(run_dir, 'step_*')))
        epoch_ckpts = sorted(glob.glob(os.path.join(run_dir, 'epoch_*')))
        latest = step_ckpts[-1] if step_ckpts else (epoch_ckpts[-1] if epoch_ckpts else None)
        if latest:
            model.load_adapter(latest, adapter_name='default', is_trainable=True)
            log.info(f"Loaded adapter weights from {latest}")

    # ── Step 4 : dataset ──────────────────────────────────────
    data_path = os.path.join(cfg.get('paths', 'data_dir'), 'train.npz')
    if not os.path.exists(data_path):
        log.error(f"Training data not found: {data_path}  → run build_dataset.py first")
        sys.exit(1)

    log.info(f"Loading training data: {data_path}")
    d       = np.load(data_path)
    dataset = LLMSIDDataset(d['contexts'], d['targets'], tokenizer, K, max_len)
    log.info(f"  {len(dataset):,} samples")

    # ── Step 5 : print 3 prompts (before training) ────────────
    print_sample_prompts(dataset, tokenizer, n=3)

    # ── Step 5b : dataloader ──────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = partial(collate_pad, pad_id=tokenizer.pad_token_id or 0),
        pin_memory  = (device.type == 'cuda'),
        drop_last   = False,
    )
    total_opt_steps  = epochs * math.ceil(len(loader) / grad_accum)
    warmup_opt_steps = max(1, int(total_opt_steps * warmup_r))
    log.info(f"Batches/epoch: {len(loader):,} | "
             f"grad_accum: {grad_accum} | "
             f"optimizer steps: {total_opt_steps:,} | "
             f"warmup: {warmup_opt_steps}")

    # ── Step 6 : init wandb ───────────────────────────────────
    wb_config = dict(
        model       = 'Qwen3-0.6B',
        lora_r      = lora_r,
        lora_alpha  = lora_alpha,
        batch_size  = batch_size,
        grad_accum  = grad_accum,
        eff_batch   = batch_size * grad_accum,
        lr          = lr,
        weight_decay= wd,
        epochs      = epochs,
        max_seq_len = max_len,
        warmup_ratio= warmup_r,
        train_samples = len(dataset),
        total_opt_steps = total_opt_steps,
        sid_tokens  = len(sid_tokens),
        vocab_size  = len(tokenizer),
    )
    run = init_wandb(cfg, wb_config, run_suffix=run_tag)
    if run:
        log.info(f"wandb run: {run.url}")

    # ── Step 7 : optimiser + schedule ────────────────────────
    save_steps = cfg.getint('llm', 'save_steps', fallback=500)
    eval_steps = cfg.getint('llm', 'eval_steps', fallback=200)

    # ── val dataset ───────────────────────────────────────────
    val_path = os.path.join(cfg.get('paths', 'data_dir'), 'val.npz')
    val_loader = None
    if os.path.exists(val_path):
        log.info(f"Loading val data: {val_path}")
        dv         = np.load(val_path)
        val_ds     = LLMSIDDataset(dv['contexts'], dv['targets'], tokenizer, K, max_len)
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size * 2,   # no grad → can use larger batch
            shuffle     = False,
            num_workers = 0,
            collate_fn  = partial(collate_pad, pad_id=tokenizer.pad_token_id or 0),
            pin_memory  = (device.type == 'cuda'),
            drop_last   = False,
        )
        log.info(f"  {len(val_ds):,} val samples  |  eval every {eval_steps} opt steps")
    else:
        log.warning(f"val.npz not found at {val_path} — run build_dataset.py to generate it. "
                    f"Training without validation.")

    @torch.no_grad()
    def run_eval() -> float:
        """Compute mean val loss over the full val set."""
        model.eval()
        total, n = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
            total += out.loss.item()
            n     += 1
        model.train()
        return total / max(n, 1)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    def _lr_lambda(step: int) -> float:
        if step < warmup_opt_steps:
            return (step + 1) / warmup_opt_steps
        t = (step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Load optimizer + scheduler state if resuming from a step checkpoint
    if args.resume and step_ckpts:
        state_path = os.path.join(step_ckpts[-1], 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            resume_epoch     = state['epoch']
            global_opt_step  = state['global_opt_step']
            resume_batch_idx = state['batch_idx'] + 1
            log.info(f"Restored optimizer/scheduler state  "
                     f"epoch={resume_epoch} opt_step={global_opt_step} "
                     f"skip_batches={resume_batch_idx}")
    elif args.resume and epoch_ckpts:
        # epoch checkpoint: no optimizer state, but advance the epoch counter
        resume_epoch = int(os.path.basename(epoch_ckpts[-1]).split('_')[-1])
        log.info(f"Resuming from epoch checkpoint — starting epoch {resume_epoch + 1}")

    # ── helper: save mid-epoch checkpoint ─────────────────────
    def _save_step_ckpt(epoch_idx, batch_idx):
        path = os.path.join(run_dir, f'step_{global_opt_step:06d}')
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save({
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
            'epoch':          epoch_idx,
            'global_opt_step': global_opt_step,
            'batch_idx':      batch_idx,
        }, os.path.join(path, 'trainer_state.pt'))
        log.info(f"Step checkpoint saved → {path}")
        # Keep only the 2 most recent step checkpoints to save disk space
        old = sorted(glob.glob(os.path.join(run_dir, 'step_*')))[:-2]
        for o in old:
            import shutil
            shutil.rmtree(o, ignore_errors=True)

    # ── Step 8 : training loop ────────────────────────────────
    model.train()
    t0 = time.time()

    log.info(f"Training start  epochs={epochs}  lr={lr:.2e}  eff_batch={batch_size*grad_accum}")

    for epoch in range(resume_epoch, epochs):
        epoch_loss = 0.0
        n_trained  = 0   # batches actually trained (not skipped)
        optimizer.zero_grad()

        # How many batches to skip at the start of a resumed epoch
        skip_n = resume_batch_idx if epoch == resume_epoch else 0
        if skip_n:
            log.info(f"ep {epoch+1}/{epochs} | skipping first {skip_n} batches (already trained) ...")

        for step, batch in enumerate(loader):
            # ── skip already-trained batches (fast: no GPU involved) ──
            if step < skip_n:
                continue

            if step == skip_n:
                log.info(f"ep {epoch+1}/{epochs} | first batch loaded, starting forward pass ...")

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out  = model(**batch)
                loss = out.loss / grad_accum

            batch_loss = loss.item() * grad_accum
            loss.backward()
            epoch_loss += batch_loss
            n_trained  += 1

            # ── gradient step ──────────────────────────────
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_opt_step += 1

                # ── mid-epoch checkpoint ───────────────────
                if save_steps > 0 and global_opt_step % save_steps == 0:
                    _save_step_ckpt(epoch, step)

                # ── validation ────────────────────────────
                if val_loader is not None and eval_steps > 0 and global_opt_step % eval_steps == 0:
                    val_loss = run_eval()
                    log.info(
                        f"ep {epoch+1}/{epochs} | "
                        f"opt {global_opt_step}/{total_opt_steps} | "
                        f"VAL loss={val_loss:.4f}"
                    )
                    if run:
                        import wandb
                        wandb.log({'val/loss': val_loss, 'train/opt_step': global_opt_step})

                # ── log: first 5 steps immediately, then every 20 ─────
                if global_opt_step <= 5 or global_opt_step % 20 == 0:
                    avg_loss   = epoch_loss / max(n_trained, 1)
                    lr_now     = scheduler.get_last_lr()[0] * lr
                    elapsed    = time.time() - t0
                    steps_left = total_opt_steps - global_opt_step
                    eta_s      = elapsed / global_opt_step * steps_left if global_opt_step else 0

                    log.info(
                        f"ep {epoch+1}/{epochs} | "
                        f"opt {global_opt_step}/{total_opt_steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"lr={lr_now:.3e} | "
                        f"ETA {eta_s/3600:.1f}h"
                    )

                    if run:
                        import wandb
                        wandb.log({
                            'train/loss':      avg_loss,
                            'train/lr':        lr_now,
                            'train/epoch':     epoch + (step / len(loader)),
                            'train/opt_step':  global_opt_step,
                            'train/eta_hours': eta_s / 3600,
                        })
                        run.summary['last_loss'] = avg_loss

        # ── flush leftover gradient ────────────────────────
        if len(loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_opt_step += 1

        epoch_avg = epoch_loss / max(n_trained, 1)

        # ── epoch-end validation ───────────────────────────
        val_msg = ''
        if val_loader is not None:
            val_loss = run_eval()
            val_msg  = f" | val_loss={val_loss:.4f}"
            if run:
                import wandb
                wandb.log({
                    'val/loss':    val_loss,
                    'epoch/index': epoch + 1,
                    'train/opt_step': global_opt_step,
                })

        log.info(f"=== epoch {epoch+1}/{epochs} done | avg_loss={epoch_avg:.4f}{val_msg} ===")

        if run:
            import wandb
            wandb.log({'epoch/train_loss': epoch_avg, 'epoch/index': epoch + 1})

        # ── epoch-end checkpoint ───────────────────────────
        save_path = os.path.join(run_dir, f'epoch_{epoch+1:02d}')
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.info(f"Epoch adapter saved → {save_path}")

    if run:
        import wandb
        wandb.finish()

    log.info("Training complete.")


if __name__ == '__main__':
    main()
