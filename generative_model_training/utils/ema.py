# =====================================================================================================================
# Reference:    https://github.com/huggingface/diffusers/blob/f1d4289be80c5acfc8a1404c01fd324d8011e319
#               /src/diffusers/training_utils.py#L42
# =====================================================================================================================

import copy
import torch


class EMAModel(torch.nn.Module):
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, model, update_after_step=0, inv_gamma=1.0, power=2.0 / 3.0, min_value=0.0, max_value=0.9999, device=None):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        super().__init__()
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.register_buffer("update_after_step", torch.tensor(update_after_step))
        self.register_buffer("inv_gamma", torch.tensor(inv_gamma))
        self.register_buffer("power", torch.tensor(power))
        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)

        self.register_buffer("decay", torch.tensor(0.0))
        self.register_buffer("optimization_step", torch.tensor(0))

    def state_dict(self, *args, **kwargs):
        return self.averaged_model.state_dict(*args, **kwargs)

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay.value = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param

            if not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.averaged_model.load_state_dict(ema_state_dict, strict=True)
        self.optimization_step += 1