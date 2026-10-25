"""Budgeted FP32 training with finite-step checks and a final checkpoint."""
import copy
import math
import torch
from .update_counter import UpdateCounter
from .checkpoint import save_last
from .seed import set_seed


class ModelEMA:
    def __init__(self, model, decay):
        if not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay must be finite and in (0,1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA model structure differs")
        with torch.no_grad():
            for key, value in state.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite EMA source")
                dest = self.shadow[key]
                if value.shape != dest.shape or value.dtype != dest.dtype:
                    raise ValueError("EMA tensor structure differs")
                if value.is_floating_point() or value.is_complex():
                    dest.lerp_(value, 1-self.decay)
                else:
                    dest.copy_(value)
        self.updates += 1

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)


def build_optimizer(model, recipe):
    phases = recipe["offline_phases"]
    if len(phases) != 1 or phases[0]["scheduler"]["type"] not in ("constant", "CosineAnnealingLR"):
        raise ValueError("Expected one phase with a constant or cosine learning rate")
    spec = copy.deepcopy(phases[0]["optimizer"])
    kind = spec.pop("type")
    if kind not in ("Adam", "AdamW"):
        raise ValueError("Supported optimizers are Adam and AdamW")
    spec["betas"] = tuple(spec["betas"])
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    return getattr(torch.optim, kind)(params, **spec)


def learning_rate_at(recipe, completed_updates):
    """Rate after successful updates; skipped updates do not advance the schedule."""
    if type(completed_updates) is not int or not 0 <= completed_updates <= 10000:
        raise ValueError("Successful-update count must be in [0, 10000]")
    phase = recipe["offline_phases"][0]
    initial = phase["optimizer"]["lr"]
    schedule = phase["scheduler"]
    if schedule["type"] == "constant":
        return initial
    if schedule["type"] != "CosineAnnealingLR" or schedule["T_max"] != 10000:
        raise ValueError("Expected the declared 10000-update cosine schedule")
    minimum = schedule["eta_min"]
    return minimum + (initial-minimum)*(1+math.cos(math.pi*completed_updates/10000))/2


def finite_step(model, optimizer, batch, completed_updates, clip=None):
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch, completed_updates)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        return False, None
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        return False, float(loss.detach())
    if clip is not None:
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("Gradient clipping threshold must be positive")
        norm = torch.stack([g.detach().double().norm(2) for g in gradients]).norm(2)
        factor = min(1.0, float(clip/(norm+1e-6)))
        with torch.no_grad():
            for g in gradients:
                g.mul_(factor)
    optimizer.step()
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise FloatingPointError("Nonfinite optimizer result; no completed run is saved")
    return True, float(loss.detach())


def train(model, batches, recipe, output, *, device="cpu", seed=0):
    if (recipe.get("successful_optimizer_updates") != 10000 or seed != 0
            or recipe.get("seed") != 0 or recipe.get("effective_batch_size") != 1
            or recipe.get("checkpoint_selection") != "last"):
        raise ValueError("Expected 10000 updates, seed 0, batch size one and last checkpoint")
    set_seed(seed)
    model = model.to(device=device, dtype=torch.float32).train()
    optimizer = build_optimizer(model, recipe)
    clip = recipe["offline_phases"][0].get("gradient_clip_norm")
    decay = recipe.get("ema_decay")
    ema = ModelEMA(model, decay) if decay is not None else None
    counter = UpdateCounter(10000)
    batches = iter(batches)
    last_loss = None
    while not counter.done:
        batch = next(batches)
        if any(v.shape[0] != 1 for v in batch.values()):
            raise ValueError("Effective batch size must be one")
        batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
        success, last_loss = finite_step(model, optimizer, batch, counter.completed, clip)
        counter.record(success)
        if success:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(recipe, counter.completed)
        if success and ema is not None:
            ema.update(model)
        if counter.skipped > 1000:
            raise FloatingPointError("Repeated unsuccessful steps; budget is not extended")
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 successful_updates=counter.completed, skipped_steps=counter.skipped,
                 seed=seed, recipe=recipe, last_training_loss=last_loss,
                 ema=ema.state_dict() if ema is not None else None,
                 torch_rng=torch.get_rng_state())
    save_last(output, state)
    return dict(successful_updates=counter.completed, skipped_steps=counter.skipped,
                last_training_loss=last_loss)
