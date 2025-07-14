"""
Taken and modified from alxndrTL's othello_mamba repository:
https://github.com/alxndrTL/othello_mamba
"""

import math
import inspect
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # noqa: FURB184
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)  # noqa: FURB184
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


@dataclass
class MambaConfig:
    n_embd: int  # D
    n_layer: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4
    vocab_size: int = 64

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = True  # use official CUDA implementation when training

    model_type: str = "mamba"  # mamba or mamba_ssm

    # For transformer
    num_states: int = 64
    num_state_dimensions: int = 1
    predict_type: str = "next_token"  # "next_token" or "state"
    pad_id: int = -1
    freeze_reps: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.n_embd  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.n_embd / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        if config.model_type in ("mamba", "mamba2"):
            self.layers = nn.ModuleList(
                [ResidualBlock(config) for _ in range(config.n_layer)]
            )
            self.out_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        elif config.model_type == "lstm":
            self.layers = nn.LSTM(
                config.n_embd, config.n_embd, config.n_layer, batch_first=True
            )
        elif config.model_type == "rnn":
            self.layers = nn.RNN(
                config.n_embd, config.n_embd, config.n_layer, batch_first=True
            )
        else:
            raise ValueError("Invalid model_type")

        if config.predict_type == "state":
            self.state_predictor = nn.Linear(
                config.n_embd,
                config.num_states * config.num_state_dimensions,
                bias=True,
            )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith(("fc_3.weight", "c_proj.weight")):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

        if self.config.freeze_reps:
            for name, param in self.named_parameters():
                if "lm_head" not in name and "state_predictor" not in name:
                    param.requires_grad = False

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def forward(self, idx, targets=None):
        # x : (B, L, D)

        # y : (B, L, D)
        b, t = idx.size()
        x = self.embedding(idx)

        if self.config.model_type in ("mamba", "mamba2"):
            for layer in self.layers:
                x = layer(x)
            x = self.out_norm(x)
        elif self.config.model_type in ("lstm", "rnn"):
            x, _ = self.layers(x)

        if self.config.freeze_reps:
            x = x.detach()

        if self.config.predict_type == "state":
            logits = self.state_predictor(x)
            if self.config.num_state_dimensions > 1:
                logits = logits.view(
                    b, t, self.config.num_state_dimensions, self.config.num_states
                )
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            )
            mask = idx != self.config.pad_id
            if self.config.num_state_dimensions > 1:
                loss = loss.view(b, t, self.config.num_state_dimensions).sum(-1)
            else:
                loss = loss.view(b, t)
            loss = (loss * mask).sum() / mask.sum()  # mean only over unmasked elements
        else:
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config.pad_id,
                )
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(
                    x[:, [-1], :]
                )  # note: using list [-1] to preserve the time dim
                loss = None

        return logits, loss

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        if config.model_type == "mamba":
            self.mixer = MambaBlock(config)
        elif config.model_type == "mamba2":
            from mamba_ssm import Mamba2 as Mamba2SSM

            self.mixer = Mamba2SSM(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=config.n_embd,  # Model dimension d_model
                d_state=config.d_state,  # SSM state expansion factor
                d_conv=config.d_conv,  # Local convolution width
                expand=config.expand_factor,  # Block expansion factor
            )

        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        assert isinstance(config.dt_rank, int)
        assert isinstance(self.config.dt_rank, int)

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.n_embd, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.n_embd, bias=config.bias)

        self.dt_layernorm: RMSNorm | None = None
        self.B_layernorm: RMSNorm | None = None
        self.C_layernorm: RMSNorm | None = None

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)

        if self.config.use_cuda:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            self.selective_scan_cuda = selective_scan_fn

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # noqa: FURB184

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(
            1, 2
        )  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)

        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2).to(x.dtype)  # NOTE: casting added by KV
            C = C.transpose(1, 2).to(x.dtype)
            z = z.transpose(1, 2).to(x.dtype)
            y = self.selective_scan_cuda(
                x,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = PScan.apply(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float()
        )  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output
