import torch
import torch.nn.functional as F
from ..diffusion_strategy import DiffusionStrategy


class DDIMStrategy(DiffusionStrategy):
    """
    Denoising Diffusion Implicit Models (DDIM) strategy.
    Reference: Song et al. (2021) - "Denoising Diffusion Implicit Models"
    
    DDIM is a deterministic sampling method that can generate samples faster than DDPM
    by skipping timesteps during the reverse process. It uses the same forward process
    as DDPM but a different (non-Markovian) reverse process.
    """
    
    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = 'linear',
                 beta_start: float = 1e-4, beta_end: float = 0.02, 
                 eta: float = 0.0, **kwargs):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Type of beta schedule ('linear', 'cosine', 'sigmoid')
            beta_start: Starting beta value for linear schedule
            beta_end: Ending beta value for linear schedule
            eta: Controls stochasticity (0 = deterministic DDIM, 1 = stochastic like DDPM)
        """
        super().__init__(num_timesteps, **kwargs)
        self.eta = eta
        
        # Get beta schedule (same as DDPM)
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            from ..noise_schedulers.cosine import cosine_beta_schedule
            self.betas = cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'sigmoid':
            from ..noise_schedulers.sigmoid import sigmoid_beta_schedule
            self.betas = sigmoid_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters (same as DDPM)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for forward process (same as DDPM)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        Same as DDPM.
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """DDIM also predicts noise (epsilon prediction), same as DDPM"""
        return noise
    
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple MSE loss on predicted noise, same as DDPM"""
        return F.mse_loss(model_output, target)
    
    def sample_step(self, xt: torch.Tensor, t: torch.Tensor, model: torch.nn.Module,
                   conditioning: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        DDIM reverse process: deterministic (when eta=0) or semi-stochastic sampling.
        
        The key difference from DDPM is that DDIM can skip timesteps and uses a
        deterministic update rule (when eta=0).
        """
        # Get model prediction (noise)
        if conditioning is not None:
            noise_pred = model(xt, t, conditioning)
        else:
            noise_pred = model(xt, t)
        
        # Predict x0 from noise
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        
        x0_pred = (xt - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        
        # Get previous timestep parameters
        # For single-step sampling, assume prev_t = t - 1
        prev_t = torch.maximum(t - 1, torch.zeros_like(t))
        sqrt_alphas_cumprod_prev = self._extract(self.sqrt_alphas_cumprod, prev_t, xt.shape)
        sqrt_one_minus_alphas_cumprod_prev = self._extract(self.sqrt_one_minus_alphas_cumprod, prev_t, xt.shape)
        
        # Compute direction pointing to xt
        pred_dir = sqrt_one_minus_alphas_cumprod_prev * noise_pred
        
        # DDIM deterministic sampling (eta=0)
        if self.eta == 0.0:
            x_prev = sqrt_alphas_cumprod_prev * x0_pred + pred_dir
        else:
            # Add stochasticity controlled by eta
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t, xt.shape)
            alphas_cumprod_prev = self._extract(self.alphas_cumprod, prev_t, xt.shape)
            
            # Compute variance
            sigma_t = self.eta * torch.sqrt(
                (1 - alphas_cumprod_prev) / (1 - alphas_cumprod_t) * 
                (1 - alphas_cumprod_t / alphas_cumprod_prev)
            )
            
            # Adjust direction for stochastic sampling
            pred_dir = torch.sqrt(1 - alphas_cumprod_prev - sigma_t**2) * noise_pred
            
            # Add noise
            noise = torch.randn_like(xt) if t.min() > 0 else torch.zeros_like(xt)
            x_prev = sqrt_alphas_cumprod_prev * x0_pred + pred_dir + sigma_t * noise
        
        return x_prev
    
    def sample_step_with_skip(self, xt: torch.Tensor, t: torch.Tensor, prev_t: torch.Tensor,
                              model: torch.nn.Module, conditioning: torch.Tensor = None, 
                              **kwargs) -> torch.Tensor:
        """
        DDIM sampling with arbitrary timestep skipping.
        
        This is the key advantage of DDIM - we can go from t to prev_t directly,
        where prev_t can be much smaller than t-1.
        
        Args:
            xt: Noisy data at timestep t
            t: Current timestep
            prev_t: Previous (target) timestep (can skip multiple steps)
            model: Denoising model
            conditioning: Optional conditioning information
        
        Returns:
            x_{prev_t}: Denoised data at timestep prev_t
        """
        # Get model prediction (noise)
        if conditioning is not None:
            noise_pred = model(xt, t, conditioning)
        else:
            noise_pred = model(xt, t)
        
        # Predict x0 from noise
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        
        x0_pred = (xt - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        
        # Get previous timestep parameters
        sqrt_alphas_cumprod_prev = self._extract(self.sqrt_alphas_cumprod, prev_t, xt.shape)
        sqrt_one_minus_alphas_cumprod_prev = self._extract(self.sqrt_one_minus_alphas_cumprod, prev_t, xt.shape)
        
        # Compute direction pointing to xt
        pred_dir = sqrt_one_minus_alphas_cumprod_prev * noise_pred
        
        # DDIM deterministic sampling (eta=0)
        if self.eta == 0.0:
            x_prev = sqrt_alphas_cumprod_prev * x0_pred + pred_dir
        else:
            # Add stochasticity controlled by eta
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t, xt.shape)
            alphas_cumprod_prev = self._extract(self.alphas_cumprod, prev_t, xt.shape)
            
            # Compute variance
            sigma_t = self.eta * torch.sqrt(
                (1 - alphas_cumprod_prev) / (1 - alphas_cumprod_t) * 
                (1 - alphas_cumprod_t / alphas_cumprod_prev)
            )
            
            # Adjust direction for stochastic sampling
            pred_dir = torch.sqrt(1 - alphas_cumprod_prev - sigma_t**2) * noise_pred
            
            # Add noise (only if not at final step)
            noise = torch.randn_like(xt) if prev_t.min() > 0 else torch.zeros_like(xt)
            x_prev = sqrt_alphas_cumprod_prev * x0_pred + pred_dir + sigma_t * noise
        
        return x_prev
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
