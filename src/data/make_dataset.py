# data preproecessing

import pandas as pd
import numpy as np
import pathlib
import sys

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def data_preprocessing(df, label = sys.argv[2]):
    # drop row with NA value
    if label == 'train.csv':
        # Drop  'pickup_datetime' and 'dropoff_datetime' column form DataFrame as it converted to individual feature 
        df.drop(['id','vendor_id', 'dropoff_datetime'], axis=1,
                inplace=True)
    else:
        # Drop  'pickup_datetime' and 'dropoff_datetime' column form DataFrame as it converted to individual feature 
        df.drop(['id','vendor_id'], axis=1,
                inplace=True)
    df.dropna(inplace=True)
    # convert datetime object to 'Datetime' format
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']) 
    # extract the day, month, and year components (for pickup)
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_Hour'] = df['pickup_datetime'].dt.hour
    df['pickup_Minute'] = df['pickup_datetime'].dt.minute
    df['pickup_Second'] = df['pickup_datetime'].dt.second

    # Drop  'pickup_datetime' and 'dropoff_datetime' column form DataFrame as it converted to individual feature 
    df.drop(['pickup_datetime'], axis=1,
            inplace=True)
    # replace on  col   'store_and_fwd_flag'
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'N':0, 'Y':1})
    return df

def save_data(df, output_path, file_name='/train.csv'):
    # save the preprocessed data in specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path+file_name, index=False)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed/'
    
    data = load_data(data_path)
    pro_data = data_preprocessing(data)
    save_data(pro_data, output_path, sys.argv[2])

    ## run file using command
    # python <py file name> <write i/p file path +name> <mention output file name>

if __name__ == "__main__":
    main()