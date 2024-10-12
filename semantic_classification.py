#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import pickle


# %%
def rf_inference(pkl_file, feature_df, output_csv, task):
    pid = feature_df["PID"]
    features = feature_df.drop(["PID"],axis=1)
    
    #load model params
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)
        
    #model parameters
    clf = model_data['model']
    selected_features = model_data['selected_features']
    scaler = model_data['scaler']
    
    #check selected_features are in features.columns
    if not set(selected_features).issubset(features.columns):
        missing_features = set(selected_features) - set(features.columns)
        raise ValueError(f"Missing features in the input data: {missing_features}")
    
    #run inference
    X_scaled = scaler.transform(features)
    X_new = X_scaled[:, [features.columns.get_loc(col) for col in selected_features]]
    y_pred = clf.predict(X_new)
    
    result_df = pd.DataFrame({'PID':pid, f'{task}_pred':y_pred})
    result_df.to_csv(output_csv, index=False)
    
    print(f"Predictions saved to {output_csv}")


# %%


if __name__ == '__main__':
    # setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help = "Classification Task")
    parser.add_argument('--bbox', type=str, help = "Original or fixed bbox approach")
    parser.add_argument('--extractor', type=str, help = "Feature extraction model")
    parser.add_argument('--feature_csv', type=str, help = "Input feature.csv file path")
    parser.add_argument('--output_csv', type=str, help = "Output predictions.csv file path")
    
    args = parser.parse_args()
    feature_csv = args.feature_csv
    output_path = args.output_csv
    if not os.path.exists('/'.join(output_path.split('/')[:-1])):
        os.makedirs('/'.join(output_path.split('/')[:-1]))

    task = args.task
    bbox = args.bbox
    extractor = args.extractor
    
    assert task in ["consistency","conspicuity","reticulation","margins","shape"], "--task argument must be consistency, conspicuity, reticulation, margin, or shape"
    assert bbox in ["orig","fixed"], "--bbox argument must be orig or fixed"
    assert extractor in ["radiomic","kinetics","med3d","lidc","fm"], "--extractor must be radiomic, kinetics, med3d, lidc, or fm"
    
    feature_df = pd.read_csv(feature_csv)
    feature_df.columns
    #ensure correct number of input features
    if extractor in ["kinetics","med3d","lidc"]:
        assert len(feature_df.columns) == 513, "Incorrect number of features"
        
    if extractor in ["radiomic"]:
        assert len(feature_df.columns) == 101, "Incorrect number of features"
        
    if extractor in ["fm"]:
        assert len(feature_df.columns) == 4097, "Incorrect number of features"
        
    if bbox == "orig":
        bx = "og"
    else:
        bx = "fixed"
    
    pickle_path = f"./rf_model_weights/{task}/{bx}_bbox/{bx}_{extractor}_{task}.pkl"
#     print(pickle_path)
    
    print(f"Running inference for {task} classification using {extractor} random forest model")
    rf_inference(pickle_path, feature_df, output_path, task)
    


# %%
