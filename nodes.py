from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch


class ChebyshevForecaster:
    def __init__(self, m: int = 4, k_history: int = 16, lam: float = 1e-3):
        self.m = m
        self.k_history = k_history
        self.lam = lam
        self.history: List[Tuple[float, torch.Tensor]] = []
        self.feature_shape: Optional[torch.Size] = None
        self.feature_dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

    def ready(self) -> bool:
        return len(self.history) >= max(3, min(self.m + 1, 4))

    def update(self, t: float, feature: torch.Tensor):
        feat = feature.detach()
        if self.feature_shape is None:
            self.feature_shape = feat.shape
            self.feature_dtype = feat.dtype
            self.device = feat.device
        self.history.append((float(t), feat))
        if len(self.history) > self.k_history:
            self.history.pop(0)

    def _cheb_matrix(self, taus: torch.Tensor, m: int) -> torch.Tensor:
        cols = [torch.ones_like(taus)]
        if m > 1:
            cols.append(taus)
        for _ in range(2, m):
            cols.append(2 * taus * cols[-1] - cols[-2])
        return torch.stack(cols[:m], dim=1)

    def predict(self, t_star: float, mix_w: float = 0.5) -> torch.Tensor:
        assert self.feature_shape is not None
        assert self.feature_dtype is not None
        assert self.device is not None

        xs = [x for _, x in self.history]
        ts = torch.tensor([t for t, _ in self.history], device=self.device, dtype=torch.float32)
        xmat = torch.stack([x.reshape(-1).to(torch.float32) for x in xs], dim=0)

        tmin = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        tmax = torch.tensor(50.0, device=self.device, dtype=torch.float32)
        mid = 0.5 * (tmin + tmax)
        rng = (tmax - tmin).clamp_min(1e-6)

        tsn = (ts - mid) * 2.0 / rng
        ttn = (torch.tensor(float(t_star), device=self.device, dtype=torch.float32) - mid) * 2.0 / rng

        n = tsn.shape[0]
        m = min(self.m + 1, n)
        T = self._cheb_matrix(tsn, m).to(torch.float32)
        TT = self._cheb_matrix(ttn.reshape(1), m).reshape(-1).to(torch.float32)

        reg = self.lam * torch.eye(m, device=self.device, dtype=torch.float32)
        lhs = T.transpose(0, 1) @ T + reg
        rhs = T.transpose(0, 1) @ xmat
        coeff = torch.linalg.solve(lhs, rhs)
        spec_pred = (TT.unsqueeze(0) @ coeff).squeeze(0).reshape(self.feature_shape)

        if len(self.history) >= 2:
            t1, x1 = self.history[-2]
            t2, x2 = self.history[-1]
            x1f = x1.to(torch.float32)
            x2f = x2.to(torch.float32)
            if abs(t2 - t1) > 1e-8:
                alpha = (float(t_star) - t2) / (t2 - t1)
                lin_pred = x2f + alpha * (x2f - x1f)
                out = mix_w * spec_pred + (1.0 - mix_w) * lin_pred
            else:
                out = spec_pred
        else:
            out = spec_pred

        return out.to(dtype=self.feature_dtype)


@dataclass
class SpectrumConfig:
    enabled: bool = True
    backend: str = "flux"
    m: int = 4
    lam: float = 0.1
    mix_w: float = 0.5
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 0
    k_history: int = 16
    debug: bool = False


class SpectrumRuntime:
    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg
        self.forecaster = ChebyshevForecaster(m=cfg.m, k_history=cfg.k_history, lam=cfg.lam)
        self.run_id = 0
        self._last_schedule_signature = None
        self.last_info: Dict[str, Any] = {}
        self.reset_all()

    def reset_cycle(self):
        self.step_idx = 0
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.decisions_by_sigma: Dict[float, Dict[str, Any]] = {}
        self.seen_sigmas: List[float] = []
        self.cycle_finished = False

    def reset_all(self):
        self.reset_cycle()
        self.last_info = {
            "enabled": self.cfg.enabled,
            "backend": self.cfg.backend,
            "patched": False,
            "hook_target": None,
            "forecasted_passes": 0,
            "actual_forward_count": 0,
            "curr_ws": float(self.cfg.window_size),
            "last_sigma": None,
            "num_steps": 0,
            "run_id": self.run_id,
        }

    def _schedule_signature(self, transformer_options: Dict[str, Any]):
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        if sample_sigmas is None:
            return None
        try:
            vals = sample_sigmas.detach().float().cpu().flatten().tolist()
            return tuple(round(float(v), 8) for v in vals)
        except Exception:
            return None

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]):
        sig = self._schedule_signature(transformer_options)
        if sig is None:
            return
        if self._last_schedule_signature is None:
            self._last_schedule_signature = sig
            self.last_info["num_steps"] = max(len(sig) - 1, 1)
            return
        if sig != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = sig
            self.reset_cycle()
            self.last_info["forecasted_passes"] = 0
            self.last_info["actual_forward_count"] = 0
            self.last_info["curr_ws"] = float(self.cfg.window_size)
            self.last_info["run_id"] = self.run_id
            self.last_info["num_steps"] = max(len(sig) - 1, 1)

    def num_steps(self) -> int:
        n = int(self.last_info.get("num_steps", 0))
        return max(n, 1)

    def sigma_key(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> float:
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is not None:
            try:
                return round(float(sigmas.detach().flatten()[0].item()), 8)
            except Exception:
                pass
        try:
            return round(float(timesteps.detach().flatten()[0].item()), 8)
        except Exception:
            return float(self.step_idx)

    def _finish_cycle_if_needed(self):
        if len(self.seen_sigmas) >= self.num_steps() and not self.cycle_finished:
            self.cycle_finished = True

    def begin_step(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> Dict[str, Any]:
        self._ensure_run_sync(transformer_options)

        sigma = self.sigma_key(transformer_options, timesteps)
        self.last_info["last_sigma"] = sigma

        if sigma in self.decisions_by_sigma:
            return self.decisions_by_sigma[sigma]

        self._finish_cycle_if_needed()

        if self.cycle_finished:
            self.reset_cycle()
            self.last_info["forecasted_passes"] = 0
            self.last_info["actual_forward_count"] = 0
            self.last_info["curr_ws"] = float(self.cfg.window_size)
            self.last_info["run_id"] = self.run_id

        step_idx = len(self.seen_sigmas)
        self.seen_sigmas.append(sigma)

        actual_forward = True
        if self.forecaster.ready() and step_idx >= self.cfg.warmup_steps:
            ws_floor = max(1, int(torch.floor(torch.tensor(self.curr_ws)).item()))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if not self.forecaster.ready():
            actual_forward = True

        if actual_forward:
            self.num_consecutive_cached_steps = 0
            self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 3)
            self.last_info["actual_forward_count"] += 1
        else:
            self.num_consecutive_cached_steps += 1
            self.last_info["forecasted_passes"] += 1

        self.step_idx = step_idx
        self.last_info["curr_ws"] = self.curr_ws

        decision = {
            "sigma": sigma,
            "step_idx": step_idx,
            "actual_forward": actual_forward,
            "run_id": self.run_id,
        }
        self.decisions_by_sigma[sigma] = decision
        return decision


def _clone_model(model: Any) -> Any:
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    opts = _ensure_model_options(model)
    if "transformer_options" not in opts or opts["transformer_options"] is None:
        opts["transformer_options"] = {}
    return opts["transformer_options"]


def _locate_flux_inner_model(model: Any):
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return outer.diffusion_model, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model, "diffusion_model"
    return None, None


def _wrap_flux_forward_orig(inner, runtime: SpectrumRuntime):
    if getattr(inner, "_spectrum_wrapped_forward_orig_final", False):
        return

    def wrapped_forward_orig(
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance=None,
        control=None,
        transformer_options={},
        attn_mask=None,
    ):
        from comfy.ldm.flux.layers import timestep_embedding

        decision = runtime.begin_step(transformer_options, timesteps)
        step_idx = decision["step_idx"]
        actual_forward = decision["actual_forward"]

        img = inner.img_in(img)
        vec = inner.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if inner.params.guidance_embed and guidance is not None:
            vec = vec + inner.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        if inner.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], inner.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + inner.vector_in(y[:, :inner.params.vec_in_dim])
        vec_orig = vec

        if not actual_forward and runtime.forecaster.ready():
            pred_feature = runtime.forecaster.predict(step_idx, mix_w=runtime.cfg.mix_w)
            return inner.final_layer(pred_feature.to(img.dtype), vec_orig)

        transformer_options_local = transformer_options.copy()
        patches = transformer_options_local.get("patches", {})
        patches_replace = transformer_options_local.get("patches_replace", {})

        if inner.txt_norm is not None:
            txt = inner.txt_norm(txt)
        txt = inner.txt_in(txt)

        if inner.params.global_modulation:
            vec = (inner.double_stream_modulation_img(vec_orig), inner.double_stream_modulation_txt(vec_orig))

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = inner.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})

        transformer_options_local["total_blocks"] = len(inner.double_blocks)
        transformer_options_local["block_type"] = "double"
        for i, block in enumerate(inner.double_blocks):
            transformer_options_local["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        attn_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                    )
                    return out

                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options_local},
                    {"original_block": block_wrap},
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options_local)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        if inner.params.global_modulation:
            vec, _ = inner.single_stream_modulation(vec_orig)

        transformer_options_local["total_blocks"] = len(inner.single_blocks)
        transformer_options_local["block_type"] = "single"
        transformer_options_local["img_slice"] = [txt.shape[1], img.shape[1]]
        for i, block in enumerate(inner.single_blocks):
            transformer_options_local["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"],
                        vec=args["vec"],
                        pe=args["pe"],
                        attn_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                    )
                    return out

                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options_local},
                    {"original_block": block_wrap},
                )
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options_local)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

        prehead_feature = img[:, txt.shape[1] :, ...]
        runtime.forecaster.update(step_idx, prehead_feature)
        return inner.final_layer(prehead_feature, vec_orig)

    inner.forward_orig = wrapped_forward_orig
    inner._spectrum_wrapped_forward_orig_final = True


class FluxSpectrumPatcher:
    @staticmethod
    def patch(model: Any, cfg: SpectrumConfig) -> Any:
        patched = _clone_model(model)

        tr_opts = _ensure_transformer_options(patched)
        prev_runtime = tr_opts.get("spectrum_runtime")
        if prev_runtime is not None:
            runtime = prev_runtime
            runtime.cfg = cfg
            runtime.forecaster.k_history = cfg.k_history
            runtime.forecaster.m = cfg.m
            runtime.forecaster.lam = cfg.lam
            runtime.reset_all()
        else:
            runtime = SpectrumRuntime(cfg)

        tr_opts["spectrum_cfg"] = cfg
        tr_opts["spectrum_runtime"] = runtime
        tr_opts["spectrum_enabled"] = cfg.enabled
        tr_opts["spectrum_backend"] = "flux"
        tr_opts["spectrum_cfg_dict"] = asdict(cfg)

        inner, inner_name = _locate_flux_inner_model(patched)
        runtime.last_info["hook_target"] = inner_name

        if inner is not None:
            _wrap_flux_forward_orig(inner, runtime)
            try:
                setattr(inner, "_spectrum_enabled", True)
                setattr(inner, "_spectrum_cfg", cfg)
                setattr(inner, "_spectrum_runtime", runtime)
                runtime.last_info["patched"] = True
            except Exception:
                runtime.last_info["patched"] = False
        else:
            runtime.last_info["patched"] = False

        return patched


class SpectrumApplyFlux:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "m": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "lam": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mix_w": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "k_history": ("INT", {"default": 16, "min": 8, "max": 512, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        enabled,
        m,
        lam,
        mix_w,
        window_size,
        flex_window,
        warmup_steps,
        k_history,
        debug,
    ):
        if not enabled:
            return (model,)

        cfg = SpectrumConfig(
            enabled=enabled,
            backend="flux",
            m=m,
            lam=lam,
            mix_w=mix_w,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            k_history=k_history,
            debug=debug,
        )
        return (FluxSpectrumPatcher.patch(model, cfg),)


NODE_CLASS_MAPPINGS = {
    "SpectrumApplyFlux": SpectrumApplyFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumApplyFlux": "Spectrum Apply Flux",
}
