"""
Neural Network Models for Claims Reserving

File: tensorboard_utils.py

This module contains helper functions to log modelling outputs in tensorboard used in the GRU framework
for claims reserving modeling.

The following classes are defined

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

# Scikit-learn imports
from sklearn.pipeline import Pipeline


# Local imports
from utils.config import  ExperimentConfig
from utils.shap import ShapExplainer, log_shap_explanations, create_background_dataset  

SEED = 42 
rng = np.random.default_rng(SEED) 
#writer = SummaryWriter()  


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Enhanced Tensorboard Outputs with SHAP Explanations
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_enhanced_tensorboard_outputs(model, dat, config: ExperimentConfig, writer: SummaryWriter):
    """
    Generate comprehensive tensorboard outputs including SHAP explanations.
    
    Args:
        model: Trained model pipeline
        dat: Original dataset
        config: Experiment configuration
        writer : SummaryWriter, optional TensorBoard writer to log the figure
    """
    print("Generating enhanced tensorboard outputs...")
    youtput = config['data'].output_field
    
    # Training set analysis
    #train = dat.loc[(dat.train_ind_time == 1) & (dat.train_ind == 1) & (dat.train_settled == 1)]
    train = dat
    train_features = train[config['data'].features + ["claim_no"]]
    
    # Generate predictions
    y_pred = model.predict(train)
    
    # Sum if 2D
    if y_pred.ndim == 2:
        y_pred = y_pred.sum(axis=1)

    # Merge predictions back into dataset
    claim_nos_dedup = train["claim_no"].drop_duplicates()
    claim_nos = train["claim_no"]
    
    if(len(claim_nos) == len(y_pred)):
        pred_df = pd.DataFrame({
            "claim_no": claim_nos.values,
            "pred_claims": y_pred
        })
    elif(len(claim_nos_dedup) == len(y_pred)):
        pred_df = pd.DataFrame({
            "claim_no": claim_nos_dedup.values,
            "pred_claims": y_pred
        })
    else:
        print("Length of predictions does not match number of claim numbers.")

    print(pred_df.shape)


    if "pred_claims" in train.columns:
        train = train.drop(columns=["pred_claims"])
    
    train_pred = train.merge(pred_df, on="claim_no", how="left")
    
    # Feature engineering for analysis
    train_pred["log_pred_claims"] = train_pred["pred_claims"].apply(lambda x: np.log(x+1))
    train_pred["log_actual"] = train_pred[youtput].apply(lambda x: np.log(x+1))
    train_pred["rpt_delay"] = np.ceil(train_pred.notidel).astype(int)
    train_pred["diff"] = train_pred[youtput] - train_pred["pred_claims"]
    train_pred["diffp"] = (train_pred[youtput] - train_pred["pred_claims"]) / train_pred[youtput]
    
    
    # Generate SHAP explanations for final model if enabled
    if config['training'].enable_shap:
        try:
            print("Generating SHAP explanations for trained model...")
            
            # Get the underlying neural network model
            nn_model = model.named_steps['model'].module_
            
            # Transform features through the pipeline (excluding the final model step)  
            pipeline_steps = model.steps[:-1]  # All steps except the model
            feature_pipeline = Pipeline(pipeline_steps)
            X_transformed = feature_pipeline.transform(train_features)
            
            # Convert to tensor
            X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
            
            # Create SHAP explainer
            background_data = create_background_dataset(X_tensor, n_samples=100)
            feature_names = config['data'].features
            shap_explainer = ShapExplainer(nn_model, background_data, feature_names)
            
            # Generate SHAP explanations for a sample of training data
            sample_size = min(200, len(X_tensor))
            #sample_indices = np.random.choice(len(X_tensor), sample_size, replace=False)
            sample_indices = rng.choice(len(X_tensor), sample_size, replace=False)
            X_sample = X_tensor[sample_indices]
            
            # Log SHAP explanations to tensorboard
            log_shap_explanations(
                writer, shap_explainer, X_sample, epoch=config['training'].nn_iter,  # Use high epoch number for final analysis
                prefix="Final_Model_SHAP", max_samples=sample_size
            )
            
            print("SHAP explanations logged to tensorboard successfully!")
            
        except Exception as e:
            print(f"Warning: Failed to generate SHAP explanations: {e}")
    
    return train_pred




def create_actual_vs_expected_plot(data, actual_col, predicted_col, title, 
                                   writer=None, tag=None, max_val=3000000, 
                                   fontsize=15, figsize=(6.4, 4.8)):
    """
    Create an Actual vs Expected scatter plot with diagonal reference line.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data to plot
    actual_col : str
        Column name for actual values (x-axis)
    predicted_col : str
        Column name for predicted values (y-axis)
    title : str
        Title for the plot
    writer : SummaryWriter, optional
        TensorBoard writer to log the figure
    tag : str, optional
        Tag for TensorBoard logging (required if writer is provided)
    max_val : float, default 3000000
        Maximum value for the diagonal reference line
    fontsize : int, default 15
        Font size for axis labels and title
    figsize : tuple, default (6.4, 4.8)
        Figure size as (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[actual_col], data[predicted_col])
    ax.plot([0, max_val], [0, max_val], 'r-', alpha=0.7)  # Diagonal reference line
    ax.set_xlabel('Actual', fontsize=fontsize)
    ax.set_ylabel('Expected', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    
    # Log to tensorboard if writer and tag provided
    if writer is not None and tag is not None:
        writer.add_figure(tag, fig)
    
    return fig, ax