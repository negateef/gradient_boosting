import click
import pandas as pd
import numpy as np
from gradientBoosting import GradientBoostingRegressor
from utilities import transform

@click.command()
@click.argument('train_data_path')
@click.argument('test_data_path')
@click.option('--output_path', default='predictions.txt')
@click.option('--n_estimators', default=100, help='Number of models in ensemble')
@click.option('--learning_rate', default=0.1, help='Learning rate')
@click.option('--max_depth', default=3, help='Depth of decision tree')
@click.option('--loss', default='ls', type=click.Choice(['ls', 'lad']), help='Loss function to optimize')
@click.option('--verbose', is_flag=True)
def main(train_data_path, test_data_path, output_path, n_estimators, learning_rate, max_depth, loss, verbose):
    if verbose:
        print 'Reading train and test data'
        
    data_train = pd.read_csv(train_data_path)
    data_test = pd.read_csv(test_data_path)
    
    if verbose:
        print 'Transforming datasets using MinCountSketch'
    
    X_train, y_train = transform(data_train)
    X_test = transform(data_test, False)
    
    if verbose:
        print 'Fitting a model on training dataset'
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, \
                                      loss=loss, max_depth=max_depth, verbose=verbose)
    model.fit(X_train, y_train)
    
    if verbose:
        print 'Predicting'
    
    predictions = model.predict(X_test)
    
    with open(output_path, 'w') as out:
        out.write(data_train.columns[0] + ',' + data_train.columns[-1] + '\n')
        for i in range(len(predictions)):
            out.write(str(data_test.ix[i, 0]))
            out.write(',')
            out.write(str(predictions[i]) + '\n')