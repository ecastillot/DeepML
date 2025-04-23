import json
import matplotlib.pyplot as plt
import pandas as pd

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
                "train_det_loss": [],
                "train_mag_loss": [],
                "train_acc": [],
                "dev_acc": [],
                "dev_loss": [],
                "dev_det_loss": [],
                "dev_mag_loss": [],
            }
            print("🗑️  Previous history cleared. Starting fresh...\n")
        else:
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        history = {
            "train_loss": [],
            "train_det_loss": [],
            "train_mag_loss": [],
            "dev_loss": [],
            "dev_det_loss": [],
            "dev_mag_loss": [],
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
