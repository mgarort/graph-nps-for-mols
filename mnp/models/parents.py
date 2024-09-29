import torch
import torch.nn as nn
import gc

class MetaModel(nn.Module):
    """
    Class to differentiate meta-models from simple-models.
    """
    def __init__(self):
        super().__init__()


class SimpleModel(nn.Module):
    """
    Class to differentiate meta-models from simple-models.
    """
    def __init__(self):
        super().__init__()


class ClassicalModel():
    """
    Class to identify classical models from scikit-learn, which don't
    have any parameters.
    """
    def __call__(self, inputs):
        x_c = inputs['x_c'].squeeze()
        y_c = inputs['y_c'].transpose(0,1)
        x_t = inputs['x_t'].squeeze()
        self.model = self.model.fit(x_c, y_c)
        y_c_hat, y_t_hat = self.predict(x_c, x_t)
        if self.prediction_mode == 'regression':
            output = {'y_c_hat_mean': y_c_hat,
                    'y_t_hat_mean': y_t_hat}
        elif self.prediction_mode == 'binary_classification':
            output = {'p_c_hat': y_c_hat,
                    'p_t_hat': y_t_hat}
        return output


class GPModel():
    """
    Class to identify GP models from GPytorch.
    """
    def __call__(self, inputs, num_epochs=None):

        # Obtain representations for datapoints
        train_x = inputs['x_c'].double().squeeze()
        train_y = inputs['y_c'].double().reshape(-1,1)
        test_x = inputs['x_t'].double().squeeze()
        
        # Train GP on those representions
        if num_epochs is None:
            num_epochs = self.num_epochs

        gp, likelihood = self.train_gp(train_x, train_y, num_epochs=num_epochs)

        # Evaluate GP
        y_t_hat = likelihood(gp(test_x.to(self.device)))
        y_c_hat = likelihood(gp(train_x.to(self.device)))
        outputs = {}
        outputs['y_t_hat_mean'] = y_t_hat.mean
        outputs['y_t_hat_var'] = y_t_hat.variance
        outputs['y_c_hat_mean'] = y_c_hat.mean
        outputs['y_c_hat_var'] = y_c_hat.variance
        # Delete old
        del gp, likelihood, train_x, train_y, test_x
        gc.collect()
        return outputs