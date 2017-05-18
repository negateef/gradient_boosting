import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostingRegressor(object):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, \
                 min_samples_split=2, min_samples_leaf=1, \
                 min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, \
                 random_state=None, max_features=None, verbose=0, max_leaf_nodes=None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        
        if loss == 'ls':
            self.gradient_loss = lambda y, y_pred : y_pred-y
        elif loss == 'lad':
            self.gradient_loss = lambda y, y_pred : 2.0 * (y_pred - y > 0.0) - 1.0
        else:
            raise ValueError("loss parameters needs to be one of 'ls' or 'lad'")
    
    def fit(self, X, y):
        n_objects, n_features = X.shape
        current_estimation = np.zeros(n_objects)
        self.models = []
        for iter in xrange(self.n_estimators):
            current_model = DecisionTreeRegressor(min_samples_split=self.min_samples_split, \
                                                  min_samples_leaf=self.min_samples_leaf, \
                                                  min_weight_fraction_leaf = self.min_weight_fraction_leaf, \
                                                  max_depth=self.max_depth, \
                                                  min_impurity_split=self.min_impurity_split, \
                                                  max_features=self.max_features, \
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  random_state=self.random_state)
            current_model.fit(X, -self.gradient_loss(y, current_estimation))
            current_estimation += self.learning_rate * current_model.predict(X)
            self.models.append(current_model)
            if self.verbose:
                if iter % (self.n_estimators / 50) == 0:
                    print 'Iter %d. MSE of residuals: %d' % (iter, mean_squared_error(y, current_estimation))
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for iter in xrange(self.n_estimators):
            predictions += self.learning_rate * self.models[iter].predict(X)
        return predictions
    
    def score(self, X, y):
        y_predicted = self.predict(X)
        return mean_squared_error(y, y_predicted)