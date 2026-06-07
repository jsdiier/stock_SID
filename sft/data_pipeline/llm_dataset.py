"""
Dataset for LLM fine-tuning on SID sequence prediction.

Perf design
-----------
Special tokens added to the vocab make tokenizer.encode() very slow (O(n*m)
pattern scan). To avoid this bottleneck in __getitem__, we precompute in
__init__:
  - chat-template prefix/suffix token-ID lists  (one-time encode, fast)
  - SID token → token_id lookup dict            (384 lookups, instant)
  - newline token-ID list

__getitem__ then builds the full input_ids purely by integer list operations —
no string encoding at all. Throughput goes from ~5 s/sample → <0.1 ms/sample.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

LEVEL_NAMES = 'abcdefghijklmnopqrstuvwxyz'
PAD_CODE    = 128   # matches rqvae.codebook_size


# ── helpers (also used by train_llm / predict_llm) ───────────────────────────

def sid_tokens_str(codes, K: int) -> str:
    """(K,) int array → '<a_18><b_4><c_121>'  (human-readable, for display)."""
    return ''.join(f'<{LEVEL_NAMES[k]}_{int(codes[k])}>' for k in range(K))


def all_sid_tokens(num_codebooks: int, codebook_size: int) -> list:
    """
    Full list of SID special tokens to register in the tokenizer.
    Order: <a_0>..<a_127>, <b_0>..<b_127>, <c_0>..<c_127>  →  3×128 = 384 tokens.
    """
    return [
        f'<{LEVEL_NAMES[k]}_{i}>'
        for k in range(num_codebooks)
        for i in range(codebook_size)
    ]


def collate_pad(batch, pad_id: int):
    """Dynamic-length collator: right-pad each field to the longest sample."""
    from torch.nn.utils.rnn import pad_sequence
    return {
        'input_ids':      pad_sequence([b['input_ids']      for b in batch], True, pad_id),
        'attention_mask': pad_sequence([b['attention_mask'] for b in batch], True, 0),
        'labels':         pad_sequence([b['labels']         for b in batch], True, -100),
    }


# ── dataset ───────────────────────────────────────────────────────────────────

class LLMSIDDataset(Dataset):
    """
    Each sample = (prompt_ids, completion_ids) assembled from integer lookup
    tables — no tokenizer.encode() call in __getitem__.

    Prompt structure (token IDs):
        [chat-prefix] <a_X><b_Y><c_Z> \\n <a_X><b_Y><c_Z> \\n … [chat-suffix]

    Completion:
        <a_X><b_Y><c_Z> <|im_end|>
    """

    def __init__(self, contexts, targets, tokenizer, K: int = 3, max_length: int = 256):
        self.contexts   = contexts   # (N, H, K) int64 numpy
        self.targets    = targets    # (N, K)    int64 numpy
        self.K          = K
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ── one-time setup ────────────────────────────────────
        # 1. Chat-template prefix / suffix token IDs
        #    Use a 3-char null-byte marker that can't appear in any template.
        _MARKER = '\x00\x01\x02'
        template_str = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': _MARKER}],
            tokenize=False,
            add_generation_prompt=True,
        )
        pre_text, suf_text = template_str.split(_MARKER)
        self.prefix_ids = list(tokenizer.encode(pre_text, add_special_tokens=False))
        self.suffix_ids = list(tokenizer.encode(suf_text, add_special_tokens=False))

        # 2. <|im_end|> closes the assistant turn
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.im_end_id = (im_end if im_end != tokenizer.unk_token_id
                          else tokenizer.eos_token_id)

        # 3. Newline separator between weeks (may be multiple tokens)
        self.newline_ids = list(tokenizer.encode('\n', add_special_tokens=False))

        # 4. SID (level, code) → single token ID  (look-up table, O(1) in __getitem__)
        self.sid_ids: dict = {}
        for k in range(K):
            for i in range(PAD_CODE):          # 0 .. 127
                tok_str = f'<{LEVEL_NAMES[k]}_{i}>'
                tok_id  = tokenizer.convert_tokens_to_ids(tok_str)
                self.sid_ids[(k, i)] = tok_id

        self._tok = tokenizer   # kept only for sample_prompt_text()

    # ── fast integer builders ─────────────────────────────────

    def _content_ids(self, ctx: np.ndarray) -> list:
        """History token IDs: SID triplets separated by newline, PAD rows stripped."""
        valid = [i for i in range(len(ctx)) if not (ctx[i] >= PAD_CODE).all()]
        ids: list = []
        for j, row in enumerate(valid):
            for k in range(self.K):
                ids.append(self.sid_ids[(k, int(ctx[row, k]))])
            if j < len(valid) - 1:
                ids.extend(self.newline_ids)
        return ids

    def _completion_ids(self, tgt: np.ndarray) -> list:
        return [self.sid_ids[(k, int(tgt[k]))] for k in range(self.K)]

    # ── Dataset API ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx):
        content_ids    = self._content_ids(self.contexts[idx])
        completion_ids = self._completion_ids(self.targets[idx])

        prompt_ids = self.prefix_ids + content_ids + self.suffix_ids
        full_ids   = prompt_ids + completion_ids + [self.im_end_id]
        prompt_len = len(prompt_ids)

        # Truncate from left (preserve recent history + target)
        if len(full_ids) > self.max_length:
            n_drop    = len(full_ids) - self.max_length
            full_ids  = full_ids[n_drop:]
            prompt_len = max(0, prompt_len - n_drop)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attn_mask = torch.ones(len(full_ids), dtype=torch.long)
        labels    = input_ids.clone()
        labels[:prompt_len] = -100        # prompt tokens → ignored in loss

        return {'input_ids': input_ids, 'attention_mask': attn_mask, 'labels': labels}

    # ── Debug helper ──────────────────────────────────────────

    def sample_prompt_text(self, idx) -> tuple:
        """Human-readable (prompt_str, next_str) — slow path, only for display."""
        ctx  = self.contexts[idx]
        tgt  = self.targets[idx]
        valid = [i for i in range(len(ctx)) if not (ctx[i] >= PAD_CODE).all()]
        history_str = '\n'.join(sid_tokens_str(ctx[i], self.K) for i in valid)
        next_str    = sid_tokens_str(tgt, self.K)
        prompt_text = self._tok.apply_chat_template(
            [{'role': 'user', 'content': history_str}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text, next_str
