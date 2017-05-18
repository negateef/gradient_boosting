import numpy as np
import pandas as pd

class CountMinSketch(object):
    def __init__(self, n_hash_tables=10, hash_table_size=100):
        self.n_hash_tables = n_hash_tables
        self.hash_table_size = hash_table_size
        self.coefs = np.random.randint(0, 100000, size=(self.n_hash_tables, 2))
        self.hash_table = np.zeros((self.n_hash_tables, self.hash_table_size))
        
    def add(self, value):
        indices = np.mod(hash(value) * self.coefs[:, 0] + self.coefs[:, 1], self.hash_table_size)
        self.hash_table[:, indices] += 1
        
    def reset(self):
        self.hash_table = np.zeros((self.n_hash_tables, self.hash_table_size))
        
    def getCount(self, value):
        indices = np.mod(hash(value) * self.coefs[:, 0] + self.coefs[:, 1], self.hash_table_size)
        return np.min(self.hash_table[:, indices])
        

def transform_categorical(X, use_columns):
    min_sketch = CountMinSketch()
    X = X.as_matrix()
    for j in range(X.shape[1]):
        if use_columns[j]:
            for i in range(X.shape[0]):
                min_sketch.add(X[i, j])
            for i in range(X.shape[0]):
                X[i, j] = min_sketch.getCount(X[i, j])
            min_sketch.reset()
    X = X.astype(np.float32)
    return X
            
            
def transform(data, has_y=True):
    if has_y:
        X = data.drop(data.columns[-1], axis=1)
        y = data.ix[:, -1].values
    else:
        X = data
    
    use_columns = []
    for i in range(X.shape[1]):
        if (X.dtypes[i] == 'int64') or (X.dtypes[i] == 'float64') or (X.dtypes[i] == 'int32') or (X.dtypes[i] == 'float32'):
            use_columns.append(False)
            X.ix[:, i] = X.ix[:, i].fillna(X.ix[:, i].mean())
        else:
            use_columns.append(True)
            X.ix[:, i] = X.ix[:, i].astype(str)
    use_columns = np.array(use_columns)
    X = transform_categorical(X, use_columns)
    if has_y:
        return X, y
    else:
        return X
                