# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig
import torch
from enum import Enum

class DSAStage(str, Enum):
    WARMUP = "warmup" # 一阶段 warmup 仅训练 indexer
    JOINT = "joint"   # 二阶段 indexer 和基模联合训练

class MiniMindDSAConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = False,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            max_batch_size: int = 32,
            max_seq_len: int = 340,
            # DSA config
            use_dsa: bool = False,
            dsa_stage: DSAStage = DSAStage.WARMUP,
            index_n_heads: int = 4,
            index_topk: int = 32,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        # DeepSeek Sparse Attention 参数
        self.use_dsa = use_dsa
        self.dsa_stage = dsa_stage
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.index_n_heads = index_n_heads
        self.index_head_dim = self.hidden_size // self.index_n_heads
        self.index_topk = index_topk


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class Indexer(nn.Module):
    def __init__(self, config: MiniMindDSAConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.hidden_size // config.num_attention_heads # 为了复用主注意力的旋转编码，indexer 每个注意力头的维度与主注意力维度一致
        self.index_topk = config.index_topk

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.w_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        self.q_norm = LayerNorm(self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)

        self.softmax_scale = self.head_dim**-0.5
        self.k_cache = None
        self.max_seq_len = config.max_position_embeddings

    def _fp16_index(self, q, weights, k):
        # q: (bsz, seqlen, n_heads, head_dim)
        # weights: (bsz, seq_len, n_heads, 1)
        # k: (bsz, seqlen_k, head_dim)
        index_score = torch.einsum(
            "bsnd,btd->bsnt", q, k
        )  # (bsz, seqlen, n_heads, seqlen_k)
        index_score = F.relu(index_score)
        weighted = index_score * weights  # (bsz, seqlen, n_heads, seqlen_k)
        index_score = weighted.sum(dim=2)  # (bsz, seqlen, seqlen_k)
        return index_score

    def _update(self, k, start_pos, end_pos):
        bsz, seqlen, _ = k.shape
        assert seqlen == end_pos - start_pos, "k length must match [start_pos, end_pos)"
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                bsz, self.max_seq_len, self.head_dim, dtype=k.dtype, device=k.device
            )
        self.k_cache[:, start_pos:end_pos] = k

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        end_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        use_cache: bool=False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        hidden_shape = (bsz, seqlen, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape))#.transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape))#.transpose(1, 2)
        # q (bsz, seqlen, n_heads, head_dim)
        # k (bsz, seqlen, 1,       head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos[:seqlen], sin[:seqlen])  # (bsz, n_heads, seqlen, head_dim)
        k = k.squeeze(2)
        # q (bsz, seqlen, n_heads, head_dim)
        # k (bsz, seqlen, head_dim)

        if use_cache and start_pos >= 0 and end_pos >= 0:
            self._update(k, start_pos, end_pos)
            k = self.k_cache[:, :end_pos] # k (bsz, pastlen+seqlen, head_dim) (bsz, seqlen_k, head_dim)

        weights = self.w_proj(x) * self.n_heads**-0.5  # (bsz, seqlen, n_heads)
        weights = (
            weights.unsqueeze(-1) * self.softmax_scale
        )  # (bsz, seqlen, n_heads, 1)

        index_score = self._fp16_index(q, weights, k)  # (bsz, seqlen, seqlen_k)
        # padding 掩码
        if attention_mask is not None:
            index_score = index_score.masked_fill(
                attention_mask[:, None, :] == 0,
                float("-inf")
            )

        # 因果掩码
        seqlen_k = index_score.shape[-1]
        mask = (
            torch.full((seqlen, seqlen_k), float("-inf"), device=x.device).triu_(1)
            if seqlen > 1
            else None
        )
        if mask is not None:
            index_score += mask

        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        # topk_indices_ = topk_indices.clone()
        # dist.broadcast(topk_indices_, src=0)
        # assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices, index_score


class DSAAttention(nn.Module):
    def __init__(self, args: MiniMindDSAConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        self.use_dsa = args.use_dsa
        self.dsa_stage = args.dsa_stage
        self.indexer = Indexer(args)
        self.indexer_loss = 0
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        start_pos = past_key_value[0].shape[1] if past_key_value is not None else 0
        end_pos = start_pos + seq_len

        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        self.indexer_loss = 0
        if self.use_dsa:
            # topk_indices (bsz, curlen, pastlen+curlen) if (topk > pastlen+curlen) else (bsz, curlen, topk)
            # xq (bsz, n_q_heads, curlen, head_dim)
            # xk (bsz, n_kv_heads, pastlen+curlen, head_dim) -> (bsz, n_q_heads, pastlen+curlen, head_dim)
            # x  (bsz, curlen, embed_dim)
            topk_indices, index_score = self.indexer(x.detach(), start_pos, end_pos, cos, sin, attention_mask, use_cache)
            mask_shape = (*x.shape[:-1], xk.shape[-2]) # (bsz, curlen, pastlen+culen)
            topk_mask = torch.zeros(
                mask_shape, dtype=torch.bool, device=x.device
            ) # (bsz, curlen, pastlen+culen)
            topk_mask = topk_mask.scatter_(-1, topk_indices, True) # 给有效的 topk 打上 True
            topk_mask = topk_mask.unsqueeze(1) # (bsz, 1, curlen, pastlen+curlen)

            if self.training:
                # 算 indexer 的 loss
                # topk mask
                eps = 1e-8
                topk_mask = topk_mask.squeeze(1)
                with torch.no_grad(): # 主注意力不参与梯度反向传播
                    attention_weights = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # (bsz, n_q_heads, curlen, pastlen+curlen)
                    # causal mask
                    causal_mask = torch.triu(
                        torch.full((seq_len, seq_len), float("-inf"), device=attention_weights.device),
                        diagonal=1
                    ).unsqueeze(0).unsqueeze(0)  # scores+mask
                    attention_weights = attention_weights + causal_mask
                    # padding mask
                    if attention_mask is not None:
                        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                        attention_weights = attention_weights + extended_attention_mask
                    # index 已经过 causal mask 和 padding mask，不再mask

                    attention_weights = F.softmax(attention_weights.float(), dim=-1).type_as(xq)
                    attention_weights = attention_weights.sum(1)

                    # TODO(wangjintao): 需要打 mask 吗，主要目的是为了让 indexer 注意力分布逼近主主干注意力分布
                    #attention_weights = attention_weights.masked_fill(~topk_mask, eps) # 无效的 token 打上极小值
                    attn_dist = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(eps)

                #index_score = index_score.masked_fill(~topk_mask, float('-inf')) # 无效的 token 打上极小值

                index_score = torch.clamp(index_score, min=-1e9, max=1e9)
                log_index_dist = F.log_softmax(index_score, dim=-1)

                kl_loss = F.kl_div(
                    log_index_dist, attn_dist, reduction="batchmean", log_target=False
                )

                self.indexer_loss = kl_loss

        if (not self.use_dsa) and self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            if self.use_dsa and (not (self.training and self.dsa_stage == DSAStage.WARMUP)): # dsa warmup 训练时不用稀疏注意力
                T = xq.shape[2]
                B, H, S, D = xk.shape
                K = topk_indices.shape[2]
                L = T * K
                topk_flat = topk_indices.reshape(B, L)
                idx = topk_flat[:, None, :, None].expand(B, H, L, D)
                xk_topk = torch.gather(xk, dim=2, index=idx).reshape(B, H, T, K, D)
                xv_topk = torch.gather(xv, dim=2, index=idx).reshape(B, H, T, K, D)
                scores = torch.einsum('bhtd,bhtkd->bhtk', xq, xk_topk) / math.sqrt(D) # (bsz, n_q_heads, curlen, topk)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # (bsz, n_q_heads, curlen, pastlen+curlen)
                scores = scores + torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)  # scores+mask

                if attention_mask is not None:
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                    scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            if self.use_dsa and (not (self.training and self.dsa_stage == DSAStage.WARMUP)): # dsa warmup 训练时不用稀疏注意力
                output = torch.einsum('bhtk,bhtkd->bhtd', scores, xv_topk) # (bsz, n_q_heads, curlen, topk)
            else:
                output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindDSAConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindDSAConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindDSAConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindDSABlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindDSAConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = DSAAttention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindDSAModel(nn.Module):
    def __init__(self, config: MiniMindDSAConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindDSABlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindDSAForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindDSAConfig

    def __init__(self, config: MiniMindDSAConfig = None):
        self.config = config or MiniMindDSAConfig()
        super().__init__(self.config)
        self.model = MiniMindDSAModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        dsa_loss = 0
        if self.training and self.config.use_dsa:
            dsa_loss = sum(layer.self_attn.indexer_loss for layer in self.model.layers)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        output.dsa_loss = dsa_loss
        return output