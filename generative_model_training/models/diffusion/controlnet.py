
import torch
from models.diffusion.unet import Upsample

class ControlledUnetModel(torch.nn.Module):

    def __init__(self, only_mid_control: bool = False):
        super().__init__()
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.iDiff_unet = None
        self.control_unet = None

    def load_state_dict(self, iDiff_unet, control_unet):
        self.iDiff_unet = iDiff_unet
        self.control_unet = control_unet

    def forward(self, x, t, context = None, hint = None, dropout_mask = None):
        hs = []
        ### freezed iDiff unet ###
        with torch.no_grad():
            # time & first block
            t_s = self.iDiff_unet.time_emb(t)
            x_s = self.iDiff_unet.image_proj(x)

            # use context only if the model is context_conditional
            if self.iDiff_unet.is_context_conditional:
                if context is None:
                    c = self.iDiff_unet.empty_context_embedding.unsqueeze(0).repeat(len(x), 1).to(x.device)
                else:
                    c = self.iDiff_unet.context_emb(context)

                    # if entire samples is dropped out, use the empty context embedding instead
                    if dropout_mask is not None:
                        c[dropout_mask] = self.iDiff_unet.empty_context_embedding.type(c.dtype).to(c.device)

                    # maybe apply component dropout to counter context overfitting
                    c = self.iDiff_unet.context_dropout(c)
            else:
                c = None
            hs.append(x_s)

            # down blocks
            for m in self.iDiff_unet.down:
                x_s = m(x_s, t_s, c)
                hs.append(x_s)
            # middle block
            x_s = self.iDiff_unet.middle(x_s, t_s, c)
            ### end ###

        
        ### control unet ###
        control = self.control_unet(x, hint, t, context, dropout_mask) 
        if control is not None:
            x_s += control.pop()
        for m in self.iDiff_unet.up:
            if isinstance(m, Upsample):
                x_s = m(x_s, t_s, c)
            else:
                ### ControlNet core ###
                if self.only_mid_control or control is None:
                    x_s = torch.cat([x_s, hs.pop()], dim=1)
                else:
                    x_s = torch.cat([x_s, hs.pop() + control.pop()], dim=1)
                x_s = m(x_s, t_s, c)
        ### end ###
        
        # out 
        noise_pred_controlled = self.iDiff_unet.final(self.iDiff_unet.act(self.iDiff_unet.norm(x_s)))
        
        return noise_pred_controlled


class ControlledUnetModel4Sample(torch.nn.Module):
    def __init__(self,
                 eps_model: torch.nn.Module,
                 T: int = 1000,
                 criterion: torch.nn.Module = torch.nn.MSELoss(),
                 schedule_type: str = 'linear',
                 schedule_k: float = 1.0,
                 schedule_beta_min: float = 0.0,
                 schedule_beta_max: float = 0.1) -> None:

        super(ControlledUnetModel4Sample, self).__init__()
        self.eps_model = eps_model
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        betas = compute_beta_schedule(T, schedule_type,
                                      k=schedule_k, beta_min=schedule_beta_min, beta_max=schedule_beta_max)
        for k, v in precompute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        self.T = T
        self.criterion = criterion
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k


    def forward(self, x0: torch.Tensor, context: torch.Tensor = None, hint: torch.Tensor = None, dropout_mask: torch.Tensor = None ) -> torch.Tensor:
        # t ~ U(0, T)
        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        # eps ~ N(0, 1)
        eps = torch.randn_like(x0)

        # get mean and standard deviation of p(x_t|x_0)
        mean = self.sqrt_alpha_bars[t, None, None, None] * x0
        sd = self.sqrt_one_minus_alpha_bars[t, None, None, None]

        # sample from p(x_t|x_0)
        x_t = mean + sd * eps

        return self.criterion(eps, self.eps_model(x_t, t, context, hint, dropout_mask))

    def sample(self, n_samples, size, x_T: torch.Tensor = None, context: torch.Tensor = None, hint: torch.Tensor = None, dropout_mask: torch.Tensor = None) -> torch.Tensor:  
        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        # this samples accordingly to Algorithm 2
        self.eval()
        with torch.no_grad():

            # pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            # pbar.set_description("DDPM Sampling")
            # for i in pbar:
            for i in reversed(range(0, self.T)):

                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()
                eps = self.eps_model(x_t, t, context, hint, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + self.sigmas[i] * z
        self.train()
        return x_t

    @staticmethod
    def sample_prior(n_samples, size):
        return torch.randn(n_samples, *size)
    

def compute_beta_schedule(
        T: int, schedule_type: str = 'linear', k: float = 1.0,
        beta_min: float = None, beta_max: float = None) -> torch.Tensor:

    if schedule_type.lower() == 'linear':
        scale = 1000 / T
        beta_1 = scale * 0.0001
        beta_T = scale * 0.02
        return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

    elif schedule_type.lower() in ['cosine', 'cosine_warped']:
        # custom modification to cosine schedule -> warped cosine schedule
        # (this is equivalent to original cosine schedule if k=1 and beta_min=0.0)

        s = 0.008
        beta_min = 0.0 if schedule_type.lower() == 'cosine' else beta_min
        k = 1 if schedule_type.lower() == 'cosine' else k

        return betas_for_alpha_bar(
            T, lambda t: math.cos(math.pi / 2 * (t + s) / (1 + s) ** k) ** 2,
            beta_min=beta_min, beta_max=beta_max
        )

    raise NotImplementedError


def betas_for_alpha_bar(T, alpha_bar, beta_min=0.0, beta_max=1.0):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(max(1 - alpha_bar(t2) / alpha_bar(t1), beta_min), beta_max))
    return torch.tensor(betas).float()


def precompute_schedule_constants(betas: torch.Tensor):
    alphas = 1 - betas
    sqrt_alphas_inv = 1 / alphas.sqrt()

    sigmas = betas.sqrt()

    alpha_bars = torch.cumsum(torch.log(alphas), dim=0).exp()
    sqrt_alpha_bars = alpha_bars.sqrt()

    sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()
    one_minus_alphas_over_sqrt_one_minus_alpha_bars = (1 - alphas) / sqrt_one_minus_alpha_bars

    """
    import matplotlib.pyplot as plt
    plt.title("Variance Schedule")
    plt.plot(betas, label="betas")
    plt.plot(alpha_bars, label="alpha_bars")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    """

    return {
        "betas": betas,
        "alphas": alphas,
        "sigmas": sigmas,
        "sqrt_alphas_inv": sqrt_alphas_inv,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "one_minus_alphas_over_sqrt_one_minus_alpha_bars": one_minus_alphas_over_sqrt_one_minus_alpha_bars
    }


