from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from tqdm import tqdm
from mnp.models.parents import ClassicalModel


class FSS(ClassicalModel):
    '''
    Fingerprint similarity search (FSS) (KNN with 1 neighbour)
    '''
    def __init__(self, prediction_mode: str='regression') -> None:
        self.prediction_mode = prediction_mode
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            self.model = KNeighborsRegressor(n_neighbors=1)
        else:
            raise NotImplementedError

    def predict(self, x_c, x_t):
        y_c_hat_mean = torch.tensor(self.model.predict(x_c)).transpose(0,1)
        y_t_hat_mean = torch.tensor(self.model.predict(x_t)).transpose(0,1)
        return y_c_hat_mean, y_t_hat_mean


class KNN(ClassicalModel):
    '''
    k-nearest neighbours (KNN)
    '''
    def __init__(self, num_neighbours: int=5,
                 prediction_mode: str='regression') -> None:
        self.prediction_mode = prediction_mode
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            self.model = KNeighborsRegressor(n_neighbors=num_neighbours)
        else:
            raise NotImplementedError
        

    def predict(self, x_c, x_t):
        y_c_hat_mean = torch.tensor(self.model.predict(x_c)).transpose(0,1)
        y_t_hat_mean = torch.tensor(self.model.predict(x_t)).transpose(0,1)
        return y_c_hat_mean, y_t_hat_mean

        

class RF(ClassicalModel):
    '''
    Random forest (RF).
    '''
    def __init__(self, num_estimators: int=500,
                 prediction_mode: str='regression') -> None:
        self.prediction_mode = prediction_mode
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            self.model = RandomForestRegressor(n_estimators=num_estimators)
        else:
            self.model = RandomForestClassifier(n_estimators=num_estimators)

    def predict(self, x_c, x_t):
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            y_c_hat_mean = torch.tensor(self.model.predict(x_c))[None,:]
            y_t_hat_mean = torch.tensor(self.model.predict(x_t))[None,:]
            return y_c_hat_mean, y_t_hat_mean
        elif self.prediction_mode == 'binary_classification':
            p_c_hat = torch.tensor(self.model.predict_proba(x_c))[None,:,1:]
            p_t_hat = torch.tensor(self.model.predict_proba(x_t))[None,:,1:]
            return p_c_hat, p_t_hat


class Dummy(ClassicalModel):
    '''
    Dummy model that predicts:
    - in regression, the mean of the training set.
    - in classification, a class at random based on the frequencies in the
      training set.
    '''
    def __init__(self, prediction_mode: str='regression') -> None:
        self.prediction_mode = prediction_mode
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            self.model = DummyRegressor(strategy='mean')
        elif self.prediction_mode == 'binary_classification':
            # The "stratified" makes predictions at random from the distibution
            # observed in the training set
            self.model = DummyClassifier(strategy='stratified')
        else:
            raise NotImplementedError

    def predict(self, x_c, x_t):
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            y_c_hat_mean = torch.tensor(self.model.predict(x_c))[None,:]
            y_t_hat_mean = torch.tensor(self.model.predict(x_t))[None,:]
            return y_c_hat_mean, y_t_hat_mean
        elif self.prediction_mode == 'binary_classification':
            p_c_hat = torch.tensor(self.model.predict_proba(x_c))[None,:,1:]
            p_t_hat = torch.tensor(self.model.predict_proba(x_t))[None,:,1:]
            return p_c_hat, p_t_hat
        


class XGB(ClassicalModel):
    '''
    Dummy model that predicts:
    - in regression, the mean of the training set.
    - in classification, a class at random based on the frequencies in the
      training set.
    '''
    def __init__(self, prediction_mode: str='regression') -> None:
        self.prediction_mode = prediction_mode
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            raise NotImplementedError
        elif self.prediction_mode == 'binary_classification':
            self.model = XGBClassifier()
        else:
            raise NotImplementedError

    def predict(self, x_c, x_t):
        if (self.prediction_mode == 'regression' or
            self.prediction_mode == 'regression_antibacterials'):
            raise NotImplementedError
        elif self.prediction_mode == 'binary_classification':
            p_c_hat = torch.tensor(self.model.predict_proba(x_c))[None,:,1:]
            p_t_hat = torch.tensor(self.model.predict_proba(x_t))[None,:,1:]
            return p_c_hat, p_t_hat