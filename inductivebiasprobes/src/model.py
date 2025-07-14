"""
GPT Implementation: taken and modified from Andrej Karpathy's nanogpt repository.
https://github.com/karpathy/nanoGPT

Mamba, RNN, LSTM Implementations: taken and modified from alxndrTL's othello_mamba repository.
https://github.com/alxndrTL/othello_mamba
"""

from dataclasses import dataclass
from functools import partial
import inspect
import logging
import math
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, random_seed=0):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class ModelConfig:
    # Common configs
    model_type: Literal["gpt", "mamba", "mamba2", "rnn", "lstm"]
    n_embd: int  # embedding dimension
    n_layer: int  # number of layers
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_dim: int | None
    output_dim: int | None
    block_size: int | None
    mask_id: int | float | None = None

    # Discrete input configs
    input_vocab_size: int | None = None

    # Discrete output configs
    output_vocab_size: int | None = None

    # GPT configs
    n_head: int | None = None
    dropout: float | None = None

    # Mamba configs
    dt_rank: int | str | None = None  # rank of the diffusion tensor
    d_state: int | None = None  # N in paper/comments
    expand_factor: int | None = None  # E in paper/comments
    d_conv: int | None = None
    dt_min: float | None = None
    dt_max: float | None = None
    dt_init: Literal["random", "constant"] | None = None
    dt_scale: float | None = None
    dt_init_floor: float | None = None
    rms_norm_eps: float | None = None
    conv_bias: bool | None = None
    inner_layernorms: bool | None = None  # apply layernorms to internal activations
    pscan: bool | None = None  # use parallel scan mode or sequential mode when training
    use_cuda: bool | None = None  # use official CUDA implementation when training

    def __post_init__(self):
        self.d_inner = None
        if self.expand_factor is not None:
            self.d_inner = self.expand_factor * self.n_embd  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.n_embd / 16)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._validate_config()

        # Initialize input layers
        if config.input_vocab_size is not None:
            self._init_discrete_input()
        else:
            self._init_continuous_input()

        # Initialize output layers
        if config.output_vocab_size is not None:
            self._init_discrete_output()
        else:
            self._init_continuous_output()

        self._init_architecture()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        logger.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def reset_output_head(self):
        """Reset the output head to random initialization while preserving other weights."""
        if self.config.output_vocab_size is not None:
            self._init_discrete_output()
        else:
            self._init_continuous_output()
        self._init_weights(self.output_head)
        logger.info(
            "Output head reset: \nnumber of parameters: %.2fM"
            % (self.get_num_params() / 1e6)
        )

    def _validate_config(self):
        """Validate that only relevant configs are set based on model type and predict type."""
        # Check common configs
        required_common = {
            "model_type",
            "n_embd",
            "n_layer",
            "bias",
            "input_dim",
            "output_dim",
            "block_size",
        }
        for field in required_common:
            if getattr(self.config, field) is None:
                raise ValueError(f"Common config {field} must be set")

        # Check model type specific configs
        gpt_configs = {"n_head", "dropout"}
        mamba_configs = {
            "dt_rank",
            "d_state",
            "expand_factor",
            "d_conv",
            "dt_min",
            "dt_max",
            "dt_init",
            "dt_scale",
            "dt_init_floor",
            "rms_norm_eps",
            "conv_bias",
            "inner_layernorms",
            "pscan",
            "use_cuda",
        }

        if self.config.model_type == "gpt":
            for field in gpt_configs:
                if getattr(self.config, field) is None:
                    raise ValueError(f"GPT config {field} must be set")
            # Ensure mamba configs are None
            for field in mamba_configs:
                if getattr(self.config, field) is not None:
                    raise ValueError(f"{field} should be None for GPT")
        elif self.config.model_type in ("mamba", "mamba2", "rnn", "lstm"):
            for field in mamba_configs:
                if getattr(self.config, field) is None:
                    raise ValueError(f"Mamba config {field} must be set")
            # Ensure GPT configs are None
            for field in gpt_configs:
                if getattr(self.config, field) is not None:
                    raise ValueError(f"{field} should be None for Mamba")

    def _init_discrete_input(self):
        """Initialize input layers for discrete input."""
        # Support 1+ discrete input channels.  For a single channel we keep the original
        # attribute name (`self.embedding`) so that weight-tying logic elsewhere continues
        # to work.  For multi-channel inputs we store the embedding tables in a ModuleList
        # to avoid a proliferation of hard-coded attribute names.
        assert (
            self.config.input_dim >= 1
        ), "input_dim must be at least 1 for discrete input"

        if self.config.input_dim == 1:
            self.embedding = nn.Embedding(
                self.config.input_vocab_size, self.config.n_embd
            )
            return

        # Each channel gets full embedding dimension for better expressiveness
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(self.config.input_vocab_size, self.config.n_embd)
                for _ in range(self.config.input_dim)
            ]
        )
        # Learned combination layer to project concatenated embeddings back to n_embd
        self.input_combiner = nn.Linear(
            self.config.input_dim * self.config.n_embd, self.config.n_embd
        )

    def _init_continuous_input(self):
        """Initialize input layers for continuous input."""
        self.input_proj = nn.Linear(self.config.input_dim, self.config.n_embd)

    def _init_discrete_output(self):
        """Initialize output layers for discrete output (classification)."""
        self.output_head = nn.Linear(
            self.config.n_embd, self.config.output_dim * self.config.output_vocab_size
        )

    def _init_continuous_output(self):
        """Initialize output layers for continuous output (regression)."""
        output_size = self.config.output_dim
        if isinstance(output_size, tuple):
            output_size = math.prod(output_size)
        self.output_head = nn.Linear(self.config.n_embd, output_size)

    def _init_architecture(self):
        """Initialize the core architecture (GPT, Mamba, RNN, or LSTM)."""
        if self.config.model_type == "gpt":
            from .nanogpt import Block, LayerNorm

            self.pos_emb = nn.Embedding(self.config.block_size, self.config.n_embd)
            self.drop = nn.Dropout(self.config.dropout)
            self.blocks = nn.ModuleList(
                [Block(self.config) for _ in range(self.config.n_layer)]
            )
            self.ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias)

            if hasattr(self, "embedding"):
                # Weight-tying: https://paperswithcode.com/method/weight-tying
                # TODO: make sure that this works even with continuous output
                # Only do weight tying if vocab sizes match
                if (
                    self.config.input_vocab_size == self.config.output_vocab_size
                    and self.embedding.weight.shape == self.output_head.weight.shape
                    and self.config.input_dim == 1
                ):
                    self.embedding.weight = self.output_head.weight
                    logger.info("Weight tying successful")
                else:
                    logger.warning(
                        "Skipping weight tying due to mismatched vocab sizes or shapes or input dimension"
                    )
            else:
                logger.info("No embedding tying performed")
        elif self.config.model_type in ("mamba", "mamba2"):
            from .mamba import ResidualBlock, RMSNorm

            self.blocks = nn.ModuleList(
                [ResidualBlock(self.config) for _ in range(self.config.n_layer)]
            )
            self.ln_f = RMSNorm(self.config.n_embd, eps=self.config.rms_norm_eps)

        elif self.config.model_type == "lstm":
            self.core = nn.LSTM(
                self.config.n_embd,
                self.config.n_embd,
                self.config.n_layer,
                batch_first=True,
            )

        elif self.config.model_type == "rnn":
            self.core = nn.RNN(
                self.config.n_embd,
                self.config.n_embd,
                self.config.n_layer,
                batch_first=True,
            )

    def forward(self, x, targets=None, target_callback=None, loss_name=None, return_reps=False):
        """Forward pass through the model."""
        b, t, input_dim = x.size()  # batch size, sequence length
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Input projection
        if hasattr(self, "embedding"):
            # Single discrete channel
            assert input_dim == 1
            x = x.view(b, t)
            x = self.embedding(x)
        elif hasattr(self, "embeddings"):
            # Multi-channel discrete input
            assert input_dim == len(self.embeddings)
            parts = [emb(x[:, :, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat(parts, dim=-1)  # (b, t, input_dim * n_embd)
            x = self.input_combiner(x)  # (b, t, n_embd)
        else:
            # Continuous input
            x = self.input_proj(x)

        if hasattr(self, "pos_emb"):
            assert hasattr(self, "drop")
            pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  # shape (1, t)
            pos = pos.to(x.device)
            pos_emb = self.pos_emb(pos)  # shape (1, t, n_embd)
            x = self.drop(x + pos_emb)  # shape (b, t, n_embd)

        # Core architecture forward pass
        if self.config.model_type in ("gpt", "mamba", "mamba2"):
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
        else:  # rnn or lstm
            x, _ = self.core(x)

        # Output projection
        if return_reps:
            return x
        output = self.output_head(x)
        if targets is not None:
            if target_callback is not None:
                targets = target_callback(targets)
                output = target_callback(output)
            if self.config.output_vocab_size is not None:  # Classification
                # Compute per-element losses with reduction='none'
                # if self.config.output_dim == 1:
                element_losses = F.cross_entropy(
                    output.contiguous().view(-1, self.config.output_vocab_size),
                    targets.view(-1),
                    ignore_index=(
                        -1
                        if self.config.mask_id == float("inf")
                        else self.config.mask_id
                    ),
                    reduction="none",
                ).reshape(targets.shape)

                # Create mask for non-ignored elements
                mask = (targets != self.config.mask_id).float()

                # Sum losses and divide by number of non-masked elements per batch
                loss = (element_losses * mask).sum(axis=(1, 2)) / (
                    mask.sum(axis=(1, 2)) + 1e-8
                )
            else:  # Regression
                loss_fn = partial(F.mse_loss, reduction="none")
                mask = torch.where(targets == self.config.mask_id, 0.0, 1.0)
                element_losses = loss_fn(output * mask, torch.where(mask == 0.0, 0.0, targets))
                loss = (element_losses * mask).sum(axis=(1, 2)) / (
                    mask.sum(axis=(1, 2)) + 1e-8
                )
            return output, loss
        return output

    def _init_weights(self, module):
        """Initialize the weights - the same for all architectures."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position/token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self, "embedding"):
                n_params -= self.embedding.weight.numel()  # subtract token embedding
            if hasattr(self, "transformer") and hasattr(self.transformer, "wpe"):
                n_params -= (
                    self.transformer.wpe.weight.numel()
                )  # subtract position embedding for GPT
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        logger.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        Adapted from Andrej Karpathy's nanoGPT.
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        if self.config.model_type == "gpt":
            N = sum(p.numel() for p in self.parameters())
            cfg = self.config
            L, H, Q, T = (
                cfg.n_layer,
                cfg.n_head,
                cfg.n_embd // cfg.n_head,
                cfg.block_size,
            )
            flops_per_token = 6 * N + 12 * L * H * Q * T
            flops_per_fwdbwd = flops_per_token * T
        elif self.config.model_type in ("mamba", "mamba2"):
            N = sum(p.numel() for p in self.parameters())
            cfg = self.config
            L, D, E = cfg.n_layer, cfg.n_embd, cfg.expand_factor
            flops_per_token = 6 * N + L * D * E  # this is very approximate...
            # Note: Mamba's actual FLOP count might be different, this is a rough estimate
            flops_per_fwdbwd = flops_per_token * D
        else:
            return None  # MFU estimation not implemented for RNN/LSTM

        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_fwdbwd * fwdbwd_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
