import torch
from torch.utils.data import Dataset, DataLoader
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import Distribution
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from mnp.models.graphs import MolecularGraphAttentionEncoder
from gpytorch.constraints import Positive
from gpytorch.priors import Prior
import gc
from tqdm import tqdm
from typing import Union, Optional
from mnp.models.parents import GPModel


# Kernels

class TanimotoKernel(Kernel):
    """
    Tanimoto kernel
    
    k(x,x') = <x,x'> / (<x,x > + <x',x'> - <x,x'>)

    Implemented following the example for a custom kernel from 
    https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html

    NOTE The Tanimoto kernel is invariant to a lengthscale parameter. So without ARD there is no point
    in dividing the x1 and x2 by a lengthscale. But with ARD it can make a difference.
    """

    has_lengthscale = True

    # We will register the parameter when initializing the kernel
    def __init__(self, lengthscale_prior=None, lengthscale_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        ard_num_dims = kwargs['ard_num_dims']

        # register the raw parameter
        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_constraint is None:
                lengthscale_constraint = Positive()
                self.register_constraint("raw_lengthscale", lengthscale_constraint)
            if lengthscale_prior is not None:
                if not isinstance(lengthscale_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
                )
    @property
    def lengthscale(self) -> torch.Tensor:
        if self.has_lengthscale:
            return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, value: torch.Tensor):
        self._set_lengthscale(value)

    def _lengthscale_param(self, m: Kernel) -> torch.Tensor:
        # Used by the lengthscale_prior
        return m.lengthscale

    def _lengthscale_closure(self, m: Kernel, v: torch.Tensor) -> torch.Tensor:
        # Used by the lengthscale_prior
        return m._set_lengthscale(v)

    def _set_lengthscale(self, value: torch.Tensor):
        # Used by the lengthscale_prior
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))
   

    def forward(self, x1, x2, **params):
        # apply lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        # calculate terms
        x1_x1 = (x1_ * x1_).sum(axis=1)
        x2_x2 = (x2_ * x2_).sum(axis=1)
        if params['diag']:
            x1_x2 = (x1_ * x2_).sum(axis=1)
        else:
            x1_x2 = x1_ @ x2_.T
        # calculate numerator <x, x'>
        numerator = x1_x2
        # calculate denominator <x,x > + <x',x'> - <x,x'>
        if params['diag']:
            denominator = - x1_x2 + x1_x1 + x2_x2
        else:
            denominator = - x1_x2 + x1_x1.reshape(-1,1) + x2_x2.reshape(1,-1)
        # prevent divide by 0 errors
        denominator.where(denominator == 0,
                          torch.as_tensor(1e-20).to(denominator.device))

        tanimoto = numerator.div(denominator)

        return tanimoto


# GP models

class ExactGaussianProcess(gpytorch.models.ExactGP):
    """
    Exact GP model. Adapted from 
    https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: Likelihood,
                 basekernel: Kernel=gpytorch.kernels.RBFKernel,
                 ard_num_dims: Optional[int]=None) -> None:
        super().__init__(train_x, train_y.squeeze(), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(basekernel(ard_num_dims=ard_num_dims))

    def forward(self, x: torch.Tensor) -> Distribution:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximateGaussianProcess(gpytorch.models.ApproximateGP):
    """
    Approximate GP using
    - lower triangular covariance matrix.
    - variational strategy from Hensman et al 2015.
    Adaped from
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
    """
    def __init__(self, inducing_points: torch.Tensor,
                 basekernel: Kernel=gpytorch.kernels.RBFKernel) -> None:
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(basekernel())

    def forward(self, x: torch.Tensor) -> Distribution:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# GP models with training loop

class GaussianProcessOnEncoder(GPModel):
    def __init__(self, num_atom_features: int, num_bond_features: int, 
                 num_atom_V_features: int, num_bond_V_features: int, 
                 num_QK_features: int, mp_iterations: int):

        self.encoder = MolecularGraphAttentionEncoder(num_atom_features=num_atom_features,
                            num_bond_features=num_bond_features,
                            num_atom_V_features=num_atom_V_features,
                            num_bond_V_features=num_bond_V_features,
                            num_QK_features=num_QK_features,
                            mp_iterations=mp_iterations, device='cpu').float()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.gp_class = ExactGaussianProcess
        self.likelihood_class = gpytorch.likelihoods.GaussianLikelihood
        self.optimizer_class = torch.optim.Adam
        self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
        self.training_iter = 50


    def train_gp(self, gp, likelihood, train_x, train_y):

        gp.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = self.optimizer_class(gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = self.mll_class(likelihood, gp)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = gp(train_x.double())
            # Calc loss and backprop gradients
            loss = -mll(output, train_y.double())
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, self.training_iter, loss.item(),
                gp.covar_module.base_kernel.lengthscale.item(),
                gp.likelihood.noise.item()
            ))
            optimizer.step()
        gp.eval()
        likelihood.eval()
        return gp, likelihood

    def __call__(self, inputs):

        # Obtain representations for datapoints
        train_x = self.encoder(inputs['atoms_c'].float(), inputs['atoms_mask_c'].float(),
                                        inputs['adjacencies_c'].float(), inputs['bonds_c'].float())
        train_y = inputs['y_c']
        test_x = self.encoder(inputs['atoms_t'].float(), inputs['atoms_mask_t'].float(), 
                                        inputs['adjacencies_t'].float(), inputs['bonds_t'].float())
        
        # Train GP on those representions
        likelihood = self.likelihood_class()
        gp = self.gp_class(train_x=train_x.double(), train_y=train_y.double(),
                                likelihood=likelihood).cpu()
        gp, likelihood = self.train_gp(gp, likelihood, train_x, train_y)

        # Evaluate GP
        y_preds = likelihood(gp(test_x.double()))
        outputs = {}
        outputs['y_t_hat_mean'] = y_preds.mean
        outputs['y_t_hat_var'] = y_preds.variance
        # Delete old
        del gp, likelihood, train_x, train_y, test_x
        gc.collect()
        return outputs


class SimpleDataset(torch.utils.data.Dataset):
    """
    Simple dataset class from x and y.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ApproximateGaussianProcessOnFingerprints(GPModel):
    """
    Steps of initialization:

    - Initialize model, inputting the inducing points DONE
    - Initialize likelihood function DONE

    Steps of training:

    - Activate training mode of GP and likelihood DONE
    - Initialize optimizer with GP and likelihood parameters DONE
    - Initialize the MLL DONE
    """

    def __init__(self, num_inducing_points: int=500,
                 basekernel: Kernel=gpytorch.kernels.RBFKernel,
                 num_epochs: int=250,
                 device: Union[torch.DeviceObjType, str]='cuda',
                 verbose: bool=False) -> None:

        
        # Initialize the inducing point locations to the first locations of the
        # training set
        self.num_inducing_points = num_inducing_points
        self.basekernel = basekernel

        self.likelihood_class = gpytorch.likelihoods.GaussianLikelihood
        self.optimizer_class = torch.optim.Adam
        
        self.num_epochs = num_epochs

        self.device = torch.device(device)
        self.verbose = verbose


    def train_gp(self, train_x, train_y, num_epochs):

        train_dataset = SimpleDataset(train_x, train_y)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

        initial_inducing_points = train_x[:self.num_inducing_points]

        gp = ApproximateGaussianProcess(inducing_points=initial_inducing_points,
                                        basekernel=self.basekernel).double().to(self.device)
        likelihood = self.likelihood_class().double().to(self.device)

        gp.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = self.optimizer_class([{'params': gp.parameters()},
                                          {'params': likelihood.parameters()}],
                                         lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = VariationalELBO(likelihood, gp,
                              num_data=len(train_dataset))

        for i_epoch in tqdm(range(num_epochs), disable=(not self.verbose)):

            for i_batch, batch in enumerate(train_dataloader):

                # get the inputs in double format
                x_train = batch[0].to(self.device)
                y_train = batch[1].to(self.device)
            
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                
                # Forward pass
                output = gp(x_train)
                loss = -mll(output, y_train.squeeze())

                # Calculate gradients
                loss.backward()

                # Take optimization step
                optimizer.step()

            if self.verbose:
                print(f'Epoch {i_epoch}, \n'
                    # f'batch {i_batch}, \n'
                    # f'lengthscale {gp.covar_module.base_kernel.lengthscale.item()}, '
                    f'noise {likelihood.noise.item()}')

        gp.eval()
        likelihood.eval()
        return gp, likelihood



class ExactGaussianProcessOnFingerprints(GPModel):
    """
    Steps of initialization:

    - Initialize model, inputting the inducing points DONE
    - Initialize likelihood function DONE

    Steps of training:

    - Activate training mode of GP and likelihood DONE
    - Initialize optimizer with GP and likelihood parameters DONE
    - Initialize the MLL DONE
    """

    def __init__(self,
                 basekernel: Kernel=gpytorch.kernels.RBFKernel,
                 ard_num_dims: Optional[int]=None,
                 num_epochs: int=2000,
                 device: Union[torch.DeviceObjType, str]='cuda',
                 prediction_mode: str='regression',
                 verbose: bool=False) -> None:

        self.basekernel = basekernel
        self.ard_num_dims = ard_num_dims
        self.likelihood_class = gpytorch.likelihoods.GaussianLikelihood
        self.optimizer_class = torch.optim.Adam  
        self.num_epochs = num_epochs
        self.prediction_mode = prediction_mode
        self.device = torch.device(device)
        self.verbose = verbose


    def train_gp(self, train_x, train_y, num_epochs):

        likelihood = self.likelihood_class().double().to(self.device)
        gp = ExactGaussianProcess(train_x=train_x, train_y=train_y,
                                  likelihood=likelihood,
                                  basekernel=self.basekernel,
                                  ard_num_dims=self.ard_num_dims).double().to(self.device)
        gp.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = self.optimizer_class(params=gp.parameters(), lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, gp)

        for i_epoch in tqdm(range(num_epochs), disable=(not self.verbose)):

            # get the inputs in double format
            # x_train = batch[0].to(self.device)
            # y_train = batch[1].to(self.device)
        
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            output = gp(train_x.to(self.device))
            loss = -mll(output, train_y.squeeze().to(self.device))

            # Calculate gradients
            loss.backward()

            # Take optimization step
            optimizer.step()

            if self.verbose:
                print(f'Epoch {i_epoch}, \n'
                    # f'batch {i_batch}, \n'
                    # f'lengthscale {gp.covar_module.base_kernel.lengthscale.item()}, '
                    f'noise {likelihood.noise.item()}')

        gp.eval()
        likelihood.eval()
        return gp, likelihood