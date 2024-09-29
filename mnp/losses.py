import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence


# NOTE Here we are defining losses for minimization, so essentially we negate the
# log-likelihood or the ELBO.


class ReconstructionLoss(_Loss):
    
    """
    Loss for computing the following reconstruction terms, negated (*):
    
    - In the conditional neural process (CNP),
        
        log p_\theta(y_T | x_D, y_C, x_T),
    
      where C stands for context points, T for target points and D for the
      whole dataset.

    - In the latent neural process (LNP), 
        
        log p_\theta(y_T | z, x_T),
        
      where z is a function-specific latent variable that is sampled
      using information from x_C and y_C, so effectively the above
      could also be written as

        log p_\theta(y_T | z, x_D, y_c).

    (*) NOTE This loss is designed for minimization, so it actually negates
         the quantities above.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.GaussianNLLLoss(full=True, reduction='mean')

    def forward(self, y_t: torch.Tensor, y_t_hat_mean: torch.Tensor,
                y_t_hat_var: torch.Tensor) -> torch.Tensor:
        return self.loss(y_t_hat_mean, y_t, y_t_hat_var)



class RegularizationLoss(_Loss):

    """
    Loss for computing the following regularization term in the latent neural
    process (LNP):

        - KL( q(z | x_D, y_D) || q(z | x_C, y_C) ),

    where C stands for context points and D for the whole dataset.

    (*) NOTE This loss is designed for minimization, so it actually negates/
         /removes the minus sign of the KL.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, full_z_dist: Distribution, 
                context_z_dist: Distribution):
        """
        - full_z_dist: distribution conditioned on all datapoints,
            q(z | x_D, y_D)
        - context_z_dist: distribution conditioned on context datapoints,
            q(z | x_C, y_C)
        - full_z_sample: not used here. Used in ApproximateRegularizationLoss.
            Kept here for compatibility.
        """

        kl = kl_divergence(full_z_dist, context_z_dist).mean()
        reg_term = -1 * kl
        # reg_term is the regularization term as it appears in the LNP objective,
        # which has to be maximized. However, since losses in Pytorch must be
        # minimized, we multiply by -1 once more.
        reg_loss = -1 * reg_term
        return reg_loss


class ApproximateRegularizationLoss(_Loss):

    """
    Loss for approximating by Monte Carlo approximation the following
    regularization term in the latent neural process (LNP):

        - KL( q(z | x_D, y_D) || q(z | x_C, y_C) ) = 
        - KL( q_full(z) || q_context(z) )

    where C stands for context points and D for the whole dataset.

    (*) NOTE This loss is designed for minimization, so it actually negates/
         /removes the minus sign of the KL.
    """
    
    def __init__(self):
        super().__init__()
    

    def forward(self, full_z_dist: Distribution, 
                context_z_dist: Distribution,
                full_z_sample: torch.Tensor):
        """
        - full_z_dist: distribution conditioned on all datapoints,
            q(z | x_D, y_D)
        - context_z_dist: distribution conditioned on context datapoints,
            q(z | x_C, y_C)
        - full_z_sample: sample from q_full(z) used in the current forward
            pass. Will be used here to approximate the KL by Monte Carlo
            approximation.
        """

        # KL( q_full(z) || q_context(z) ) =
        # E_{q_full}[ log q_full(z) ] - E_{q_full}[ log q_context(z) ]
        kl = (full_z_dist.log_prob(full_z_sample).sum(axis=1) -
              context_z_dist.log_prob(full_z_sample).sum(axis=1)).mean()
        reg_term = -1 * kl
        reg_loss = -1 * reg_term
        return reg_loss



class ReconstructionRegularizationLoss(_Loss):
    """
    Loss for LNP, computing KL analytically.
    """

    def __init__(self):
        super().__init__()

        self.reconstruction = ReconstructionLoss()
        self.regularization = RegularizationLoss()

    def forward(self, y_t: torch.Tensor,
                y_t_hat_mean: torch.Tensor,
                y_t_hat_var: torch.Tensor,
                full_z_dist: Distribution,
                context_z_dist: Distribution) -> torch.Tensor:

        reconstruction_loss = self.reconstruction(y_t, y_t_hat_mean, y_t_hat_var)
        regularization_loss = self.regularization(full_z_dist, context_z_dist)
        total_loss = reconstruction_loss + regularization_loss
        return total_loss, reconstruction_loss, regularization_loss



class ApproximateReconstructionRegularizationLoss(_Loss):
    """
    Loss for LNP, approximating KL by Monte Carlo.
    """

    def __init__(self):
        super().__init__()

        self.reconstruction = ReconstructionLoss()
        self.regularization = ApproximateRegularizationLoss()

    def forward(self, y_t: torch.Tensor,
                y_t_hat_mean: torch.Tensor,
                y_t_hat_var: torch.Tensor,
                full_z_dist: Distribution,
                context_z_dist: Distribution,
                full_z_sample: torch.Tensor) -> torch.Tensor:
        """
        full_z_sample is not used here. It is used in the
        ApproximateReconstructionRegularization loss and kept here for
        compatibility.
        """

        reconstruction_loss = self.reconstruction(y_t, y_t_hat_mean, y_t_hat_var)
        approximate_regularization_loss = self.regularization(full_z_dist,
                                                              context_z_dist,
                                                              full_z_sample)
        total_loss = reconstruction_loss + approximate_regularization_loss
        return total_loss, reconstruction_loss, approximate_regularization_loss



class KLGaussianLoss(_Loss):
    """
    KL divergence between two Gaussians with diagonal covariance.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu_1, sigma_diag_1, mu_2, sigma_diag_2):
        n = mu_1.shape[1]

        log_frac_det = torch.log(sigma_diag_1).sum() - torch.log(sigma_diag_2).sum()
        trace = torch.sum((1 / sigma_diag_2) * sigma_diag_1, 1)
        quadratic = torch.sum((mu_2 - mu_1) * (1 / sigma_diag_2) * (mu_2 - mu_1), 1)

        kl = 1/2 * (log_frac_det - n + trace + quadratic)

        return kl