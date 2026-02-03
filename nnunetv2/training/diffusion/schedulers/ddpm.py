import torch
import torch.nn.functional as F
from ..diffusion_strategy import DiffusionStrategy


class DDPMStrategy(DiffusionStrategy):
    """
    Denoising Diffusion Probabilistic Models (DDPM) strategy.
    Reference: Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
    """
    
    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = 'linear', 
                 beta_start: float = 1e-4, beta_end: float = 0.02, **kwargs):
        super().__init__(num_timesteps, **kwargs)
        
        # Get beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            # Will be implemented via noise_schedulers
            from ..noise_schedulers.cosine import cosine_beta_schedule
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """DDPM predicts noise (epsilon prediction)"""
        return noise
    
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple MSE loss on predicted noise"""
        return F.mse_loss(model_output, target)
    
    def sample_step(self, xt: torch.Tensor, t: torch.Tensor, model: torch.nn.Module, 
                   conditioning: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Reverse process: p(x_{t-1} | x_t) 
        """
        # Get model prediction
        if conditioning is not None:
            noise_pred = model(xt, t, conditioning)
        else:
            noise_pred = model(xt, t)
        
        # Compute posterior mean
        posterior_mean = self._get_posterior_mean(xt, noise_pred, t)
        
        # Add noise (except at t=0)
        if t.min() > 0:
            noise = torch.randn_like(xt)
            posterior_variance = self._extract(self.posterior_variance, t, xt.shape)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            return posterior_mean
    
    def _get_posterior_mean(self, xt: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute posterior mean given noise prediction"""
        coef1 = self._extract(self.posterior_mean_coef1, t, xt.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, xt.shape)
        
        # Predict x0 from noise
        sqrt_recip_alphas_cumprod = 1.0 / self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_recipm1_alphas_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        x0_pred = sqrt_recip_alphas_cumprod * xt - sqrt_recipm1_alphas_cumprod / sqrt_recip_alphas_cumprod * noise_pred
        
        return coef1 * x0_pred + coef2 * xt
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
