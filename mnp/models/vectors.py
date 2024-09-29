# For fully-connected models

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, Dict
from torch.distributions.distribution import Distribution
from mnp.models.parents import SimpleModel, MetaModel

# The CNP should read the inputs for each batch
# and then output predictions. The batches should be
# of the same size.

class FullyConnectedNP(MetaModel, metaclass=ABCMeta):

    def __init__(self, x_len: int=1024, y_len: int=1, r_len: int=250,
                 lincs_architecture: bool=False, 
                 encoder_1_len: int=1000, encoder_2_len: int=500,
                 decoder_1_len: int=1000, decoder_2_len: int=50,
                 decoder_3_len: int=None, decoder_4_len: int=None,
                 device: str='cuda', use_layernorm: bool=False,
                 prediction_mode: str='regression'):
        # Save arguments
        self.x_len = x_len
        self.y_len = y_len
        self.r_len = r_len
        self.encoder_1_len = encoder_1_len
        self.encoder_2_len = encoder_2_len
        self.decoder_1_len = decoder_1_len
        self.decoder_2_len = decoder_2_len
        self.decoder_3_len = decoder_3_len
        self.decoder_4_len = decoder_4_len
        self.device = torch.device(device)
        self.use_layernorm = use_layernorm
        self.prediction_mode = prediction_mode
        self.lincs_architecture = lincs_architecture
        if lincs_architecture:
            encoder_1_len = 3000
            encoder_2_len = 3000
            decoder_1_len = 3000
            decoder_2_len = 3000
        
        # Initialize model
        super().__init__()

        # initialize linear layers
        encoder_linear_1_shape = (x_len+y_len, encoder_1_len)
        self.encoder_linear_1 = nn.Linear(*encoder_linear_1_shape)
        encoder_linear_2_shape = (encoder_1_len, encoder_2_len)
        self.encoder_linear_2 = nn.Linear(*encoder_linear_2_shape)
        encoder_linear_3_shape = (encoder_2_len, r_len)
        self.encoder_linear_3 = nn.Linear(*encoder_linear_3_shape)
        decoder_linear_1_shape = (x_len+r_len, decoder_1_len)
        self.decoder_linear_1 = nn.Linear(*decoder_linear_1_shape)
        decoder_linear_2_shape = (decoder_1_len, decoder_2_len)
        self.decoder_linear_2 = nn.Linear(*decoder_linear_2_shape)

        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            # in regression, we need one param for mean and one param for variance
            output_len = y_len * 2
        elif self.prediction_mode == 'binary_classification':
            # in binary classification, we need a single param as input to sigmoid
            output_len = y_len
            self.sigmoid = torch.nn.Sigmoid()
        else:
            raise ValueError

        if decoder_3_len is None:
            decoder_linear_3_shape = (decoder_2_len, output_len)
            self.decoder_linear_3 = nn.Linear(*decoder_linear_3_shape)
            self.decoder_linear_4 = lambda x: x
            self.decoder_linear_5 = lambda x: x
        else:
            decoder_linear_3_shape = (decoder_2_len, decoder_3_len)
            self.decoder_linear_3 = nn.Linear(*decoder_linear_3_shape)
            decoder_linear_4_shape = (decoder_3_len, decoder_4_len)
            self.decoder_linear_4 = nn.Linear(*decoder_linear_4_shape)
            decoder_linear_5_shape = (decoder_4_len, output_len)
            self.decoder_linear_5 = nn.Linear(*decoder_linear_5_shape)

        # initialize activations
        self.encoder_activation = nn.ReLU()
        self.decoder_activation_1 = nn.ReLU()
        self.decoder_activation_2 = nn.ReLU()
        
        if decoder_3_len is not None:
            self.decoder_activation_3 = nn.ReLU()
            self.decoder_activation_4 = nn.ReLU()
        else:
            self.decoder_activation_3 = lambda x: x
            self.decoder_activation_4 = lambda x: x

        # initialize layernorm
        # (by initializing it to an identity operation if we don't use layernorm,
        #  we avoid checking self.uselayernorm after layer in the forward method)
        # (we don't use layernorm on the very last layer of the encoder and
        #  decoder in order to retain all expressive power)
        if self.use_layernorm:
            self.encoder_layernorm_1 = nn.LayerNorm(encoder_linear_1_shape[1])
            self.encoder_layernorm_2 = nn.LayerNorm(encoder_linear_2_shape[1])
            self.decoder_layernorm_1 = nn.LayerNorm(decoder_linear_1_shape[1])
            self.decoder_layernorm_2 = nn.LayerNorm(decoder_linear_2_shape[1])
            if decoder_3_len is not None:
                self.decoder_layernorm_3 = nn.LayerNorm(decoder_linear_3_shape[1])
                self.decoder_layernorm_4 = nn.LayerNorm(decoder_linear_4_shape[1])
            else:
                self.decoder_layernorm_3 = lambda x: x
                self.decoder_layernorm_4 = lambda x: x                            
        else:
            self.encoder_layernorm_1 = lambda x: x
            self.encoder_layernorm_2 = lambda x: x
            self.decoder_layernorm_1 = lambda x: x
            self.decoder_layernorm_2 = lambda x: x
            self.decoder_layernorm_3 = lambda x: x
            self.decoder_layernorm_4 = lambda x: x
        # Move to device
        self.to(self.device)

    def datapoint_encoder(self, x_c, y_c):
        """
        x_c has shape (num_functions, num_datapoints, num_features)
        y_c has shape (num_functions, num_datapoints, 1)
        """
        if y_c.dim() == 2:
            c = torch.cat([x_c, y_c[:,:,None]], dim=2)
        else:
            c = torch.cat([x_c, y_c], dim=2)
        c = self.encoder_linear_1(c)
        c = self.encoder_layernorm_1(c)
        c = self.encoder_activation(c)
        c = self.encoder_linear_2(c)
        c = self.encoder_layernorm_2(c)
        c = self.encoder_activation(c)
        r_i = self.encoder_linear_3(c)
        return r_i

    def decoder_net(self, v):
        v = self.decoder_linear_1(v)
        v = self.decoder_layernorm_1(v)
        v = self.decoder_activation_1(v)
        v = self.decoder_linear_2(v)
        v = self.decoder_layernorm_2(v)
        v = self.decoder_activation_2(v)
        v = self.decoder_linear_3(v)
        v = self.decoder_layernorm_3(v)
        v = self.decoder_activation_3(v)
        v = self.decoder_linear_4(v)
        v = self.decoder_layernorm_4(v)
        v = self.decoder_activation_4(v)
        v = self.decoder_linear_5(v)
        # if regression, then the output is mean and variance of a Gaussian
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            y_hat_mean =   v[:, :, :self.y_len]
            y_hat_logvar = v[:, :, self.y_len:]
            y_hat_var = torch.exp(y_hat_logvar)
            return y_hat_mean, y_hat_var
        # if binary classification, then the output should be sigmoid probability
        elif self.prediction_mode == 'binary_classification':
            p_hat = self.sigmoid(v)[:,:,:1]
            return p_hat

    def decoder(self, encoding, x_c: torch.Tensor, x_t: torch.Tensor, 
                max_points_per_iter: int=200) -> Tuple[torch.Tensor]:
        # TODO Change max_points per iter to something reasonable, and confirm that
        # we obtain the same predictions regardless of max_points_per_iter.
        """
        Decoder that takes in function representation and locations x, and returns
        predictions y_hat for those locations.
        
        Args:
        - encoding: could be deterministic representation r in the CNP, or random
            representation z in the LNP.
        - x_c: locations att he
        """
        num_context = x_c.shape[1]
        num_target = x_t.shape[1]
        num_total = num_context + num_target
        x = torch.cat([x_c, x_t], axis=1)

        # evaluate decoder net in chunks if input is too large
        if num_total > max_points_per_iter:
            outputs = []
            indices = torch.tensor(range(num_total))
            index_split = torch.tensor_split(indices,
                                             int(num_total/max_points_per_iter))
            for each_index_split in index_split:
                v = torch.cat([x[:,each_index_split],
                               encoding[:,None,:].expand(encoding.shape[0], 
                                                    len(each_index_split), 
                                                    encoding.shape[1])], dim=2)
                each_output = self.decoder_net(v)
                outputs.append(each_output)
            if (self.prediction_mode == 'regression' or
                self.prediction_mode == 'regression_antibacterials'):
                y_hat_mean, y_hat_var = self.unpack_outputs(outputs)
            elif self.prediction_mode == 'binary_classification':
                y_hat = self.unpack_outputs(outputs)

        # evaluate decoder net in a single pass if input is small
        else:
            v = torch.cat([x,encoding[:,None,:].expand(encoding.shape[0], 
                                                       num_total, 
                                                       encoding.shape[1])], dim=2)
            if (self.prediction_mode == 'regression' or
                self.prediction_mode == 'regression_antibacterials'):
                y_hat_mean, y_hat_var = self.decoder_net(v)
            elif self.prediction_mode == 'binary_classification':
                y_hat = self.decoder_net(v)

        # make and return output dictionary
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            output_dict = self.make_regression_output_dict(y_hat_mean=y_hat_mean,
                                                           y_hat_var=y_hat_var,
                                                           num_context=num_context,
                                                           num_target=num_target)
        if self.prediction_mode == 'binary_classification':
            output_dict = self.make_binary_classification_output_dict(y_hat=y_hat,
                                                           num_context=num_context,
                                                           num_target=num_target)
        return output_dict


    def unpack_outputs(self, outputs: list) -> dict:
        # if regression, then each element in the list of outputs is a tuple
        # with Gaussian mean and variance for several datapoints
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            y_hat_mean = []
            y_hat_var = []
            for each_output in outputs:
                each_y_hat_mean, each_y_hat_var = each_output
                y_hat_mean.append(each_y_hat_mean)
                y_hat_var.append(each_y_hat_var)
            y_hat_mean = torch.concat(y_hat_mean, axis=1)
            y_hat_var = torch.concat(y_hat_var, axis=1)
            return y_hat_mean, y_hat_var
        # if binary classification, each element in the list of outputs is a
        # set of sigmoid probabilities for several datapoints
        elif self.prediction_mode == 'binary_classification':
            y_hat = []
            for each_output in outputs:
                y_hat.append(each_output)
            y_hat = torch.concat(y_hat, axis=1)
            return y_hat


    def make_regression_output_dict(self, y_hat_mean: torch.Tensor,
                                   y_hat_var: torch.Tensor, num_context: int,
                                   num_target: int) -> dict:
        assert y_hat_mean.shape[1] == num_context + num_target
        assert y_hat_var.shape[1] ==  num_context + num_target
        y_c_hat_mean = y_hat_mean[:, :num_context]
        y_c_hat_var = y_hat_var[:, :num_context]
        y_t_hat_mean = y_hat_mean[:, num_context:]
        y_t_hat_var = y_hat_var[:, num_context:]
        output_dict = {'y_c_hat_mean': y_c_hat_mean,
                'y_c_hat_var': y_c_hat_var,
                'y_t_hat_mean': y_t_hat_mean,
                'y_t_hat_var': y_t_hat_var}
        return output_dict

    def make_binary_classification_output_dict(self, y_hat: torch.Tensor,
                                              num_context: int,
                                              num_target: int) -> dict:
        assert y_hat.shape[1] == num_context + num_target
        y_c_hat = y_hat[:, :num_context]
        y_t_hat = y_hat[:, num_context:]
        output_dict = {'p_c_hat': y_c_hat,
                'p_t_hat': y_t_hat}
        return output_dict

    @abstractmethod
    def function_encoder(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class FullyConnectedCNP(FullyConnectedNP):

    def function_encoder(self, r_i):
        # Reduce along the first dimension, which is the context points dimension
        # (each batch b has dimensions N x M_b x D, where N is the number of
        # functions, M_b is the number of points, and D is the dimensions)
        # This yields the global function representation r
        r = torch.mean(r_i, dim=1)
        # The conventions is that function_encoder will return the latent encoding z
        # and also the distribution that created it (or None, if no distribution was used)
        return r

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        - inputs: dictionary with keys `x_c` (for context inputs), `y_c` (for context 
            outputs) and `x_t` (for target inputs).
        """
        x_c = inputs['x_c'].to(self.device)
        y_c = inputs['y_c'].to(self.device)
        x_t = inputs['x_t'].to(self.device)

        # Datapoint encoder. 
        # The CNP creates the function encoder using only the context datapoints
        r_i = self.datapoint_encoder(x_c, y_c)
        # Function encoder
        r = self.function_encoder(r_i)
        # Decoder
        output_dict = self.decoder(r, x_c, x_t)
        return output_dict

    def till_function_encoder(self, x_c, y_c):
        # Till datapoint encoder
        r_i = self.datapoint_encoder(x_c, y_c)
        r = self.function_encoder(r_i)
        return r

    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        super().to(device)


class AttentionCNP(FullyConnectedNP):

    def __init__(self, QK_len=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.QK_len = torch.tensor(QK_len)
        self.query = nn.Linear(self.x_len, QK_len)
        self.key = nn.Linear(self.r_len, QK_len)
        # Move to device
        self.to(self.device)

    def attention(self, r_j, x_t):
        # r_j has dimensions f x j x r (num functions x num contexts x num latent features)
        # query has dimensions f x i x k (num functions x num targets x num QK features)
        # key has dimensions f x j x k (num functionx x num contexts x num QK features)
        # attention coefficients have dimensions f x i x j (num functions x num targets x num contexts)
        # weighted r_i has dimensions f x i x r (functions x contexts x num latent features)
        query = self.query(x_t)
        key = self.key(r_j)
        coeff = torch.einsum('fik,fjk->fij',query,key) / torch.sqrt(self.QK_len)
        weighted_ri = torch.einsum('fij,fjr->fir', coeff, r_j)
        r = torch.einsum('fdl->fl',weighted_ri)
        return r

    def function_encoder(self, r_i):
        # Reduce along the first dimension, which is the context points dimension
        # (each batch b has dimensions N x M_b x D, where N is the number of
        # functions, M_b is the number of points, and D is the dimensions)
        # This yields the global function representation r
        r = torch.mean(r_i, dim=1)
        # The conventions is that function_encoder will return the latent encoding z
        # and also the distribution that created it (or None, if no distribution was used)
        return r

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        - inputs: dictionary with keys `x_c` (for context inputs), `y_c` (for context 
            outputs) and `x_t` (for target inputs).
        """
        x_c = inputs['x_c'].to(self.device)
        y_c = inputs['y_c'].to(self.device)
        x_t = inputs['x_t'].to(self.device)

        # Datapoint encoder. 
        # The CNP creates the function encoder using only the context datapoints
        r_i = self.datapoint_encoder(x_c, y_c)
        # Function encoder
        r = self.function_encoder(r_i)
        # Decoder
        output_dict = self.decoder(r, x_c, x_t)
        return output_dict

    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        super().to(device)


class FullyConnectedLNP(FullyConnectedNP):
    """
    This model is the original latent neural process (LNP), where during training 
    the prediction of the targets y_T is done from a latent variable obtained from
    the whole x_D, y_D.
    """

    def __init__(self, x_len: int=1024, y_len: int=1, r_len: int=250,
                 z_len: int=250, use_layernorm: bool=False,
                 prediction_mode: str='regression',
                 device: str='cuda'):

        super().__init__(x_len=x_len, y_len=y_len, r_len=r_len,
                         use_layernorm=use_layernorm,
                         prediction_mode=prediction_mode,
                         device=device)
        self.z_len = z_len
        self.latent_linear_1 = nn.Linear(r_len, 1000)
        self.latent_linear_2 = nn.Linear(1000, 500)
        self.latent_linear_3 = nn.Linear(500,z_len*2)
        self.latent_activation_1 = nn.ReLU()
        self.latent_activation_2 = nn.ReLU()
        self.prediction_mode = prediction_mode
        # Move to device
        self.to(self.device)

    def get_latent_param(self, r: torch.Tensor):
        x = self.latent_linear_1(r)
        x = self.latent_activation_1(x)
        x = self.latent_linear_2(x)
        x = self.latent_activation_2(x)
        x = self.latent_linear_3(x)
        mu = x[:,:self.z_len]
        log_sigma = x[:,self.z_len:]
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def function_encoder(self,r_i: torch.Tensor) -> Tuple[torch.Tensor, Distribution]:
        # Deterministic function encoding (same as in the CNP)
        r = torch.mean(r_i, dim=1)
        # Probabilistic and coherent function encoding (new in LNP w.r.t. CNP)
        mu, sigma = self.get_latent_param(r)
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        # rsample produces a reparameterized sample that allows backpropagation
        # through the sampling
        z = normal.rsample()
        return z, normal

    def forward(self, inputs: Dict[str, torch.Tensor]):

        """
        - inputs: dictionary with keys `x_c` (for context inputs), `y_c` (for context 
            outputs), `x_t` (for target inputs) and `y_t` (for target outputs).
        """

        x_c = inputs['x_c'].to(self.device)
        y_c = inputs['y_c'].to(self.device)
        x_t = inputs['x_t'].to(self.device)   

        # Create function encoding using just context datapoints
        context_r_i = self.datapoint_encoder(x_c, y_c)
        context_z, context_z_dist = self.function_encoder(context_r_i)

        # If training, y_T is available
        if 'y_t' in inputs.keys():
            y_t = inputs['y_t'].to(self.device)
            x_d = torch.cat([x_c, x_t], axis=1)
            y_d = torch.cat([y_c, y_t], axis=1)
            # Create function encoding using all datapoints (context and target)
            full_r_i = self.datapoint_encoder(x_d, y_d)
            full_z, full_z_dist = self.function_encoder(full_r_i)
            # Decode using the function encoding from both the context and target points
            output_dict = self.decoder(full_z, x_c, x_t)
            output_dict['full_z_sample'] = full_z
            output_dict['full_z_dist'] = full_z_dist           

        # If testing, y_T is not available
        else:
            # Decode using the function encoding from the context points only
            output_dict = self.decoder(context_z, x_c, x_t)

        output_dict['context_z_sample'] = context_z
        output_dict['context_z_dist'] = context_z_dist   

        return output_dict

    def till_function_encoder(self, x_c, y_c):
        # Till datapoint encoder
        r_i = self.datapoint_encoder(x_c, y_c)
        z, distribution = self.function_encoder(r_i)
        return z, distribution


    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        super().to(device)


class FullyConnectedLNP2(FullyConnectedLNP):
    """
    This model is similar to the original latent neural process (LNP) but predictions
    of the targets y_T during training are done from a latent variable obtained from
    only the context datapoints x_C, y_C instead of from all datapoints x_D, y_D.
    """

    def forward(self, x_c, y_c, x_t, y_t):

        output = {}
        # Create function encoding using just context datapoints
        context_r_i = self.datapoint_encoder(x_c, y_c)
        context_z, context_z_dist = self.function_encoder(context_r_i)
        output['context_z'] = context_z
        output['context_z_dist'] = context_z_dist

        # Decode using the function encoding from the context points only
        y_c_hat_mean, y_c_hat_var, y_t_hat_mean, y_t_hat_var = self.decoder(context_z, x_c, x_t)
        output['y_c_hat_mean'] = y_c_hat_mean
        output['y_c_hat_var'] = y_c_hat_var
        output['y_t_hat_mean'] = y_t_hat_mean
        output['y_t_hat_var'] = y_t_hat_var
            
        # If training, y_T is available and the full-data distribution is 
        # needed to compute the regularization term
        if y_t is not None:
            x_d = torch.cat([x_c, x_t], axis=1)
            y_d = torch.cat([y_c, y_t], axis=1)
            # Create function encoding using all datapoints (context and target)
            full_r_i = self.datapoint_encoder(x_d, y_d)
            full_z, full_z_dist = self.function_encoder(full_r_i)
            output['full_z'] = full_z
            output['full_z_dist'] = full_z_dist

        return output



class FullyConnectedNN(SimpleModel, metaclass=ABCMeta):

    def __init__(self, x_len: int=1024, y_len: int=1,
                 linear_1_len: int=1000, linear_2_len: int=1000,
                 linear_3_len: int=1000, linear_4_len: int=500,
                 linear_5_len: int=50,
                 device: str='cuda', use_layernorm: bool=False,
                 leaky_relu: bool=False,
                 prediction_mode: str='regression'):
        # Save arguments
        self.x_len = x_len
        self.y_len = y_len
        self.device = torch.device(device)
        self.use_layernorm = use_layernorm
        self.prediction_mode = prediction_mode
        
        # Initialize model
        super().__init__()

        # initialize linear layers
        linear_1_shape = (x_len, linear_1_len)
        self.linear_1 = nn.Linear(*linear_1_shape)
        linear_2_shape = (linear_1_len, linear_2_len)
        self.linear_2 = nn.Linear(*linear_2_shape)
        if linear_3_len is not None:
            linear_3_shape = (linear_2_len, linear_3_len)
            self.linear_3 = nn.Linear(*linear_3_shape)
            linear_4_shape = (linear_3_len, linear_4_len)
            self.linear_4 = nn.Linear(*linear_4_shape)
            linear_5_shape = (linear_4_len, linear_5_len)
            self.linear_5 = nn.Linear(*linear_5_shape)
            linear_6_shape = (linear_5_len, y_len)
            self.linear_6 = nn.Linear(*linear_6_shape)
        else:
            linear_3_shape = (linear_2_len, y_len)
            self.linear_3 = nn.Linear(*linear_3_shape)
            self.linear_4 = lambda x: x
            self.linear_5 = lambda x: x
            self.linear_6 = lambda x: x

        # initialize activations
        if leaky_relu:
            activation_function = nn.LeakyReLU()
        else:
            activation_function = nn.ReLU()
        self.activation_1 = activation_function
        self.activation_2 = activation_function
        if linear_3_len is not None:
            self.activation_3 = activation_function
            self.activation_4 = activation_function
            self.activation_5 = activation_function
            self.activation_6 = activation_function
        else:
            self.activation_3 = lambda x: x
            self.activation_4 = lambda x: x
            self.activation_5 = lambda x: x
            self.activation_6 = lambda x: x          

        # initialize layernorm
        # (by initializing it to an identity operation if we don't use layernorm,
        #  we avoid checking self.uselayernorm after layer in the forward method)
        # (we don't use layernorm on the very last layer of the encoder and
        #  decoder in order to retain all expressive power)
        self.layernorm_1 = lambda x: x
        self.layernorm_2 = lambda x: x
        self.layernorm_3 = lambda x: x
        self.layernorm_4 = lambda x: x
        self.layernorm_5 = lambda x: x
        if self.use_layernorm:
            self.layernorm_1 = nn.LayerNorm(linear_1_shape[1])
            self.layernorm_2 = nn.LayerNorm(linear_2_shape[1])
            if linear_3_len is not None:
                self.layernorm_3 = nn.LayerNorm(linear_3_shape[1])
                self.layernorm_4 = nn.LayerNorm(linear_4_shape[1])
                self.layernorm_5 = nn.LayerNorm(linear_5_shape[1])

        # Move to device
        self.to(self.device)


    def _forward(self, x):
        """
        x has shape (num_datapoints, num_features)
        """
        x = self.linear_1(x)
        x = self.layernorm_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.layernorm_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.layernorm_3(x)
        x = self.activation_3(x)
        x = self.linear_4(x)
        x = self.layernorm_4(x)
        x = self.activation_4(x)
        x = self.linear_5(x)
        x = self.layernorm_5(x)
        x = self.activation_5(x)
        x = self.linear_6(x)
        return x

    def forward(self, inputs):

        output = {}

        if self.prediction_mode == 'regression':

            if 'x' in inputs:
                x = inputs['x'].to(self.device)
                output['y_hat_mean'] = self._forward(x)

            if 'x_c' in inputs:
                x = inputs['x_c'].to(self.device)
                output['y_c_hat_mean'] = self._forward(x)

            if 'x_t' in inputs:
                x = inputs['x_t'].to(self.device)
                output['y_t_hat_mean'] = self._forward(x)

        elif self.prediction_mode == 'binary_classification':
            if 'x' in inputs:
                x = inputs['x'].to(self.device)
                output['p_hat'] = torch.sigmoid(self._forward(x))

            if 'x_c' in inputs:
                x = inputs['x_c'].to(self.device)
                output['p_c_hat'] = torch.sigmoid(self._forward(x))

            if 'x_t' in inputs:
                x = inputs['x_t'].to(self.device)
                output['p_t_hat'] = torch.sigmoid(self._forward(x))

        else:
            raise NotImplementedError
        return output

    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        super().to(device)