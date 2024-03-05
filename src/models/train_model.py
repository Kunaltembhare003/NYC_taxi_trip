import pandas as pd
import numpy as np
import yaml
import pathlib
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error

def load_train_data(data):
    train = pd.read_csv(data)
    X_train = train.drop(['trip_duration'],axis=1)
    y_train = train[['trip_duration']]
    return X_train, y_train

# Define the objective function
def objective(param):
    # RandomForestRegressor parameters
    # Create a RandomForestRegressor model with the given parameters
    rf = RandomForestRegressor(**param)
    
    # Define 4-fold cross-validation
    kf = KFold(n_splits=4)

    
    
    # Calculate the negative mean squared error (Hyperopt minimizes the objective function)
    neg_mse = -np.mean(cross_val_score(rf,
                                       X_train, 
                                       y_train.values.ravel(),
                                       cv=kf,
                                       scoring='neg_mean_squared_error'))
    
    return neg_mse

def train_model(X_train, y_train, best_hyperparams):
    rf_model = RandomForestRegressor(**best_hyperparams)
    rf_model.fit(X_train, y_train.values.ravel())
    return rf_model

def save_model(rf_model, output_path):
    # save the trained model to the specified output path
    joblib.dump(rf_model, output_path+'/model.joblib')

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix()+'/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file_train = sys.argv[1]
    X_train, y_train = load_train_data(input_file_train)
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Define the search space
    search_space = {
        'n_estimators': hp.choice('n_estimators', params['n_estimators']['options']),
        'max_depth': hp.choice('max_depth', params['max_depth']['options']),
        'random_state': hp.choice('random_state', params['random_state']['options'])
    }

    # Run the optimization
    trials = Trials()
    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)
    
    # Convert indices to hyperparameter values
    best_hyperparams = {
        'n_estimators': params['n_estimators']['options'][best['n_estimators']],
        'max_depth': params['max_depth']['options'][best['max_depth']],
        'random_state':  params['random_state']['options'][best['random_state']]
        }   
    
    trained_model = train_model(X_train, y_train, best_hyperparams)
    save_model(trained_model, output_path)

if __name__=="__main__":
    main()
    










