import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import seisbench.generate as sbg
import numpy as np
import pandas as pd
import joblib

def load_or_init_history(history_path="history.json"):
    """
    Load existing training history or initialize a new one based on user input.

    Parameters:
        history_path (str): Path to the history JSON file.

    Returns:
        dict: A dictionary containing the training and validation history.
    """
    if os.path.exists(history_path):
        user_input = input(f"\n📁 Found existing history at '{history_path}'.\n"
                           "Do you want to [r]esume or [o]verwrite it? (r/o): ").strip().lower()

        if user_input == "r":
            with open(history_path, "r") as f:
                history = json.load(f)
            print("✅ Resuming training with existing history...\n")
        elif user_input == "o":
            history = {
                "train_loss": [],
                "train_acc": [],
                "dev_acc": [],
                "dev_loss": [],
                "acum_time" : []
            }
            print("🗑️  Previous history cleared. Starting fresh...\n")
        else:
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        history = {
            "train_loss": [],
            "dev_loss": [],
            "train_acc": [],
            "dev_acc": [],
            "acum_time" : []
        }
        print("📄 No previous history found. Starting fresh...\n")

    return history

def handle_checkpoint(checkpoint_path):
    """
    Check if a checkpoint file exists and ask the user whether to remove or keep it.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        str: The checkpoint path (unchanged), or exits if user cancels.
    """
    if os.path.exists(checkpoint_path):
        user_input = input(f"\n📁 Found existing checkpoint at '{checkpoint_path}'.\n"
                           "Do you want to [d]elete it and start fresh, or [k]eep it? (d/k): ").strip().lower()
        
        if user_input == "d":
            os.remove(checkpoint_path)
            print("🗑️  Old checkpoint deleted. Starting fresh...\n")
        elif user_input == "k":
            print("✅ Keeping existing checkpoint...\n")
            exit()
        else:
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        print("📄 No checkpoint found. Training will start from scratch.\n")
    
    return checkpoint_path



def prepare_data_generators(data, scaler_path):
    """
    Prepares data generators and data loaders with augmentation and labelling.

    Parameters
    ----------
    data : object
        Data object that supports `.train_dev_test()` and `.plot_map()` methods.
    scaler_path : str
        Path to the precomputed magnitude scaler (joblib format).
    batch_size : int, optional
        Batch size for the DataLoaders. Default is 100.

    Returns
    -------
    dict
        Dictionary with generators and DataLoaders:
        {
            "generator_train": generator_train,
            "generator_dev": generator_dev,
            "generator_test": generator_test,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        }
    """

    # Load and split the dataset
    train, dev, test = data.train_dev_test()

    # Initialize data generators
    generator_train = sbg.GenericGenerator(train)
    generator_dev = sbg.GenericGenerator(dev)
    generator_test = sbg.GenericGenerator(test)

    # Load magnitude scaler
    scaler = joblib.load(scaler_path)

    # Define magnitude normalization/denormalization
    def normalize_magnitude(magnitude):
        magnitude_df = pd.DataFrame([[magnitude]], columns=["source_magnitude"])
        return scaler.transform(magnitude_df)

    def denormalize_magnitude(magnitude):
        return scaler.inverse_transform(magnitude).flatten()

    # Augmentations
    normalize = sbg.Normalize(
        detrend_axis=0,
        amp_norm_type="peak",
        eps=1e-8,
        key="X"
    )

    detection_label = sbg.DetectionLabeller(
        p_phases="trace_p_arrival_sample",
        s_phases="trace_s_arrival_sample",
        factor=1.5,
        key=("X", "y_detection")
    )

    # Magnitude labeller
    def magnitude_labeler(state_dict):
        waveforms, metadata = state_dict["X"]
        if metadata["trace_category"] == "noise":
            norm_mag = np.array([[0.0]])
        else:
            norm_mag = normalize_magnitude(metadata["source_magnitude"])
        state_dict["y_magnitude"] = norm_mag

    # Event labeller
    def detection_labeler(state_dict):
        waveforms, metadata = state_dict["X"]
        y = np.array([0]) if metadata["trace_category"] == "noise" else np.array([1])
        state_dict["y_scalar_detection"] = [y.reshape(1, 1)]

    # Attach augmentation functions to generators
    for g in [generator_train, generator_dev, generator_test]:
        g.augmentation(magnitude_labeler)
        g.augmentation(detection_labeler)
        g.add_augmentations([
            normalize,
            detection_label
        ])


    return {
        "generator_train": generator_train,
        "generator_dev": generator_dev,
        "generator_test": generator_test,
    }

def create_sample_mask(metadata: pd.DataFrame, category: str,
                       n_samples: int, random_state: int = 42,
                       min_mag: float = None, max_mag: float = None) -> pd.Series:
    """
    Create a boolean mask for sampling rows from a metadata DataFrame 
    that match a given trace_category, with optional magnitude filtering
    if the category includes the word "earthquake".

    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata DataFrame that includes 'trace_category' and 'mag' columns.
    category : str
        The target trace_category value to filter by (e.g., "earthquake_local").
    n_samples : int
        The number of samples to draw from the filtered subset.
    random_state : int, optional
        The seed for random sampling to ensure reproducibility (default is 42).
    min_mag : float, optional
        Minimum magnitude to include (only applied if "earthquake" in category).
    max_mag : float, optional
        Maximum magnitude to include (only applied if "earthquake" in category).

    Returns
    -------
    pd.Series
        A boolean Series mask aligned with the original metadata index,
        with True values for sampled rows matching the filtering conditions.
    """
    # Step 1: Filter by trace_category
    mask = metadata["trace_category"] == category
    filtered_metadata = metadata[mask]

    # Step 2: Apply magnitude filtering if "earthquake" is in the category
    if "earthquake" in category.lower():
        if min_mag is not None:
            filtered_metadata = filtered_metadata[filtered_metadata["source_magnitude"] >= min_mag]
        if max_mag is not None:
            filtered_metadata = filtered_metadata[filtered_metadata["source_magnitude"] <= max_mag]

    # Step 3: Sample rows after filtering
    sampled_metadata = filtered_metadata.sample(n=n_samples, random_state=random_state)

    # Step 4: Build final mask from sampled indices
    sampled_indices = sampled_metadata.index
    final_mask = metadata.index.isin(sampled_indices)

    return final_mask

def load_detection_outputs(model_outputs):
    """
    Load detection outputs from saved files.

    Args:
        model_outputs (dict): A dictionary where keys are model names and
                              values are paths to the saved output files.

    Returns:
        dict: A dictionary where each key is a model name and the value is
              another dictionary with 'y' (true labels) and 'y_pred' (predictions).
    """
    outputs = {}
    
    for model_name, output_path in model_outputs.items():
        # Load the saved file
        data = torch.load(output_path)
        
        # Extract true labels and predictions
        y = data["y"].squeeze().cpu().numpy()
        y_pred = data["y_pred"].squeeze().cpu().numpy()
        y_pred = np.round(y_pred)
        
        print(f"loading: y {y.shape} | y_pred {y_pred.shape} ")
        
        # Store in the outputs dictionary
        outputs[model_name] = {
            "y": y,
            "y_pred": y_pred
        }
    
    return outputs
        