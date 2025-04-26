import os
root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root
    
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from utils import load_detection_outputs

def plot_map(metadata, res="110m", connections=False, xlim=None, ylim=None, states=False, save_path=None, **kwargs):
    """
    Plots the dataset onto a map using the Mercator projection. Requires a cartopy installation.

    :param res: Resolution for cartopy features, defaults to 110m.
    :type res: str, optional
    :param connections: If true, plots lines connecting sources and stations. Defaults to false.
    :type connections: bool, optional
    :param xlim: Tuple of (min_lon, max_lon) for x-axis limits.
    :type xlim: tuple, optional
    :param ylim: Tuple of (min_lat, max_lat) for y-axis limits.
    :type ylim: tuple, optional
    :param states: If true, adds U.S. state boundaries to the map.
    :type states: bool, optional
    :param kwargs: Plotting kwargs that will be passed to matplotlib plot. Args need to be prefixed with
                   `sta_`, `ev_` and `conn_` to address stations, events or connections.
    :return: A figure handle for the created figure.
    """
    fig = plt.figure(figsize=(15, 10))
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Plotting the data set requires cartopy. "
            "Please install cartopy, e.g., using conda."
        )

    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    ax.coastlines(res)
    land_50m = cfeature.NaturalEarthFeature(
        "physical", "land", res, edgecolor="face", facecolor=cfeature.COLORS["land"]
    )
    ax.add_feature(land_50m)

    if states:
        ax.add_feature(cfeature.STATES.with_scale(res), edgecolor='gray')

    # Add text with dataset summary
    num_examples = len(metadata)
    num_events = (metadata["trace_category"] != "noise").sum()
    num_noise = (metadata["trace_category"] == "noise").sum()

    summary_text = (
        f"Total examples: {num_examples}\n"
        f"Event signals: {num_events}\n"
        f"Noise signals: {num_noise}"
    )

    ax.text(
        0.,
        0.01,
        summary_text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    def prefix_dict(kws, prefix):
        return {
            k[len(prefix):]: v
            for k, v in kws.items()
            if k.startswith(prefix)
        }

    lines_kws = {
        "marker": "",
        "linestyle": "-",
        "color": "grey",
        "alpha": 0.5,
        "linewidth": 0.5,
    }
    lines_kws.update(prefix_dict(kwargs, "conn_"))

    station_kws = {"marker": "^", "color": "k", "linestyle": "", "ms": 10}
    station_kws.update(prefix_dict(kwargs, "sta_"))

    event_kws = {"marker": ".", "color": "r", "linestyle": ""}
    event_kws.update(prefix_dict(kwargs, "ev_"))

    # Plot connecting lines
    if connections:
        station_source_pairs = metadata[
            [
                "station_longitude_deg",
                "station_latitude_deg",
                "source_longitude_deg",
                "source_latitude_deg",
            ]
        ].values
        for row in station_source_pairs:
            ax.plot(
                [row[0], row[2]],
                [row[1], row[3]],
                transform=ccrs.Geodetic(),
                **lines_kws,
            )

    # Plot stations
    station_locations = np.unique(
        metadata[["station_longitude_deg", "station_latitude_deg"]].values,
        axis=0,
    )
    
    
    ax.plot(
        station_locations[:, 0],
        station_locations[:, 1],
        transform=ccrs.PlateCarree(),
        **station_kws,
    )

    # Plot events
    source_locations = np.unique(
        metadata[["source_longitude_deg", "source_latitude_deg"]].values,
        axis=0,
    )
    ax.plot(
        source_locations[:, 0],
        source_locations[:, 1],
        transform=ccrs.PlateCarree(),
        **event_kws,
    )

    # Set limits if provided
    if xlim is not None:
        ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Save figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_metadata_distributions(metadata, save_path=None, dpi=300, xlims=None):
    """
    Plots distributions of source depth, magnitude, epicentral distance,
    and S-P arrival time difference from a metadata DataFrame.

    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame containing the required columns:
        'source_depth_km', 'source_magnitude',
        'source_latitude_deg', 'source_longitude_deg',
        'station_latitude', 'station_longitude',
        'trace_p_arrival_sample', 'trace_s_arrival_sample'

    save_path : str, optional
        If provided, saves the figure to this path.

    dpi : int, optional
        Dots per inch for saved figure. Default is 300.

    xlims : dict, optional
        Dictionary of x-axis limits for each subplot.
        Keys: 'depth', 'magnitude', 'distance', 'sp_diff'
        Values: tuple of (min, max) or None

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    axes : ndarray of Axes
        Array of matplotlib Axes for further editing if needed.
    """
    # Compute epicentral distance using haversine formula
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371.0  # Earth radius in kilometers
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    epicentral_distance = haversine(
        metadata["source_longitude_deg"].values,
        metadata["source_latitude_deg"].values,
        metadata["station_longitude_deg"].values,
        metadata["station_latitude_deg"].values
    )

    sp_difference = (
        metadata["trace_s_arrival_sample"] - metadata["trace_p_arrival_sample"]
    )

    if xlims is None:
        xlims = {}

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot source depth distribution
    axes[0].hist(metadata["source_depth_km"].dropna(), bins=50, color='skyblue')
    axes[0].set_title("Source Depth (km)")
    axes[0].set_xlabel("Depth (km)")
    axes[0].set_ylabel("Count")
    if xlims.get("depth") is not None:
        axes[0].set_xlim(xlims["depth"])

    # Plot source magnitude distribution
    axes[1].hist(metadata["source_magnitude"].dropna(), bins=30, color='salmon')
    axes[1].set_title("Source Magnitude")
    axes[1].set_xlabel("Magnitude")
    axes[1].set_ylabel("Count")
    if xlims.get("magnitude") is not None:
        axes[1].set_xlim(xlims["magnitude"])

    # Plot epicentral distance distribution
    axes[2].hist(epicentral_distance, bins=50, color='lightgreen')
    axes[2].set_title("Epicentral Distance (km)")
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Count")
    if xlims.get("distance") is not None:
        axes[2].set_xlim(xlims["distance"])

    # Plot S-P difference distribution
    axes[3].hist(sp_difference.dropna(), bins=50, color='plum')
    axes[3].set_title("S - P Arrival Time Difference (samples)")
    axes[3].set_xlabel("S - P (samples)")
    axes[3].set_ylabel("Count")
    if xlims.get("sp_diff") is not None:
        axes[3].set_xlim(xlims["sp_diff"])

    plt.tight_layout()

    # Save figure if path is given
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes

def plot_training_history(json_path, save_path=None, dpi=300):
    """
    Loads training history from a JSON file and plots the loss and accuracy curves.

    Parameters:
    - json_path (str): Path to the history JSON file.
    - save_path (str, optional): If provided, saves the figure to this path.
    - dpi (int, optional): Dots per inch when saving the figure. Default is 300.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - axes (list of matplotlib.axes._axes.Axes): List of axes objects.
    """
    with open(json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Plot Losses ---
    if history.get("train_loss"):
        ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    if history.get("dev_loss"):
        ax1.plot(epochs, history["dev_loss"], label="Dev Loss", marker="o")

    ax1.set_ylabel("Loss",fontsize=18)
    ax1.set_title("Loss History",fontsize=18)
    ax1.legend(fontsize=16,loc="center right")
    ax1.grid(True)

    # --- Plot Accuracies ---
    if history.get("train_acc"):
        ax2.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    if history.get("dev_acc"):
        ax2.plot(epochs, history["dev_acc"], label="Dev Accuracy", marker="o")

    ax2.set_xlabel("Epoch",fontsize=18)
    ax2.set_ylabel("Accuracy",fontsize=18)
    ax2.set_title("Accuracy History",fontsize=18)
    ax2.legend(fontsize=16,loc="center right")
    ax2.grid(True)

    ax1.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)

    # --- Highlight Best Epoch ---
    if "dev_loss" in history:
        best_epoch_idx = history["dev_loss"].index(min(history["dev_loss"]))
        best_epoch = best_epoch_idx + 1

        best_dev_loss = history["dev_loss"][best_epoch_idx]
        best_train_loss = history["train_loss"][best_epoch_idx]
        best_dev_acc = history["dev_acc"][best_epoch_idx] if "dev_acc" in history else None
        best_train_acc = history["train_acc"][best_epoch_idx] if "train_acc" in history else None

        # Vertical lines
        for ax in (ax1, ax2):
            ax.axvline(best_epoch, color="black", linestyle="--", linewidth=3)

        # --- Add a text box outside the top axis (above ax1) ---
        summary_text = (
        f"$\\bf{{Best\\ Epoch:{best_epoch}}}$   |  "
        f"Dev Loss: {best_dev_loss:.4f}, Train Loss: {best_train_loss:.4f}  |  "
        f"Dev Acc: {best_dev_acc:.2%}, Train Acc: {best_train_acc:.2%}"
    )

        fig.text(0.5, 0.97, summary_text, ha="center", va="top",
                 fontsize=11, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for textbox

    # Optional: Save the figure
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path} at {dpi} dpi")

    return fig, (ax1, ax2)


def plot_multiple_histories(json_paths,label_paths,loss_limits=(-1,50),save_path=None, dpi=300):
    """
    Plots loss, accuracy, and acum_time for train and dev from multiple training history JSON files.

    Parameters:
    - json_paths (list of str): List of paths to JSON files.
    - save_path (str, optional): Path to save the resulting figure.
    - dpi (int, optional): Dots per inch for saving the figure.

    Returns:
    - fig: The matplotlib figure.
    - axes: 2D list of axes [rows][columns].
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    colors = plt.cm.viridis_r(range(0, 256, max(1, 256 // len(json_paths))))

    for idx, json_path in enumerate(json_paths):
        with open(json_path, "r") as f:
            history = json.load(f)

        label = label_paths[idx]

        # Safe plotting for each key
        for row_idx, (train_key, dev_key) in enumerate([
            ("train_loss", "dev_loss"),
            ("train_acc", "dev_acc"),
            ("acum_time", "acum_time")
        ]):
            train_vals = history.get(train_key, [])
            dev_vals = history.get(dev_key, [])

            if train_key == "train_loss":
                # print(*loss_limits)
                axes[row_idx][0].set_ylim(*loss_limits)
            if dev_key == "dev_loss":
                # print(*loss_limits)
                axes[row_idx][0].set_ylim(*loss_limits)
            
            if train_vals:
                epochs = range(1, len(train_vals) + 1)
                axes[row_idx][0].plot(epochs, train_vals, label=label, color=colors[idx],linewidth=3)
            if dev_vals:
                epochs = range(1, len(dev_vals) + 1)
                axes[row_idx][1].plot(epochs, dev_vals, label=label, color=colors[idx],linewidth=3)

    # Titles and labels
    titles = [
        ["Train Loss", "Dev Loss"],
        ["Train Accuracy", "Dev Accuracy"],
        ["Train Accumulated Time", "Dev Accumulated Time"]
    ]
    ylabels = ["Loss", "Accuracy", "Acum Time [s]"]

    for row in range(3):
        for col in range(2):
            axes[row][col].set_title(titles[row][col], fontsize=18)
            axes[row][col].set_xlabel("Epoch", fontsize=16)
            axes[row][col].set_ylabel(ylabels[row], fontsize=16)
            axes[row][col].grid(True)
            axes[row][col].tick_params(axis='both',labelsize=16)

    axes[0][0].legend(fontsize=16, loc="best")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path}")

    return fig, axes

def plot_scalar_detection_examples(generator, random_index=True,
                                   start_index=0, alpha=0.15,
                                   rows=10, cols=7,
                                   save_path=None):
    """
    Plot waveform examples with scalar detection labels using a colored background.

    Parameters
    ----------
    generator : seisbench.generate.GenericGenerator
        The data generator containing waveform samples and labels.
    random_index : bool, optional
        Whether to randomly sample from the generator. Default is True.
    start_index : int, optional
        Index to start sampling from the generator (used if random_index=False). Default is 0.
    alpha : float, optional
        Transparency level of the background color (0 to 1). Default is 0.15.
    rows : int, optional
        Number of subplot rows. Default is 10.
    cols : int, optional
        Number of subplot columns. Default is 7.
    save_path : str or None, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further customization or saving.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of subplot axes.
    """
    total_plots = rows * cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
    axs = axs.flatten()

    for idx in range(total_plots):
        ax = axs[idx]
        if random_index:
            sample_index = random.randint(0, len(generator) - 1)
        else:
            sample_index = start_index + idx

        sample = generator[sample_index]
        x = sample["X"]
        y_scalar_detection = sample["y_scalar_detection"]

        label = y_scalar_detection.item()
        color = "green" if label == 1 else "red"

        # Plot waveform
        ax.plot(x.T, color="black", linewidth=0.8)

        # Set colored background
        ax.set_facecolor(color)
        ax.patch.set_alpha(alpha)

        # Formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')

    # Hide any unused subplots
    for idx in range(total_plots, len(axs)):
        axs[idx].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axs


def plot_scalar_detection_test_examples(predictions,
                                        rows=1, cols=1,
                                        savedir_path=None):
    """
    Plot scalar detection test examples for multiple models.

    Args:
        predictions (dict): Dictionary with model names as keys and prediction dicts as values.
        rows (int, optional): Number of rows in the plot grid. Default is 1.
        cols (int, optional): Number of columns in the plot grid. Default is 1.
        savedir_path (str, optional): Directory where the plots will be saved. 
            If None, plots are not saved.

    Returns:
        figs (dict): Dictionary with model names as keys and (figure, axes) tuples as values.
    """
    figs = {}
    for test_label,model_predictions in predictions.items():
        # model_predictions = predictions[test_label]
        total_plots = rows * cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
        axs = axs.flatten()
        
        x = model_predictions["x"]
        y = model_predictions["y"].squeeze()
        y_pred = model_predictions["y_pred"].squeeze()

        for idx in range(total_plots):
            ax = axs[idx]
            
            x_idx = x[idx]
            y_idx = y[idx]
            y_pred_idx = y_pred[idx]
            
            
            if y_idx == np.round(y_pred_idx):
                color = "green"
            else:
                color = "red"

            # Plot waveform
            ax.plot(x_idx.T, color=color, linewidth=0.8)

            # Set colored background
            # ax.set_facecolor(color)
            # ax.patch.set_alpha(alpha)

            # Formatting
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"%.2f"%y_pred_idx, fontsize=14, fontweight='bold')

        # Hide any unused subplots
        for idx in range(total_plots, len(axs)):
            axs[idx].axis("off")

        fig.suptitle(test_label, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if savedir_path:
            os.makedirs(savedir_path,exist_ok=True)
            save_path = os.path.join(savedir_path,f"test_examples_{test_label}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        figs[test_label] = (fig,axs)
        
    return figs

def plot_detection_confusion_matrix(model_outputs, save_path=None, rows=None, cols=None):
    """
    Plot confusion matrices for detection outputs of multiple models.

    Args:
        model_outputs (dict): Dictionary with model names as keys and file paths as values.
        save_path (str, optional): If provided, saves the figure at this path with dpi=300.
        rows (int, optional): Number of rows for subplots grid. Default determined automatically.
        cols (int, optional): Number of columns for subplots grid. Default determined automatically.

    Returns:
        fig (matplotlib.figure.Figure): The created matplotlib figure.
        axes (array-like): The matplotlib axes corresponding to each subplot.
    """
    # Load all outputs
    outputs = load_detection_outputs(model_outputs)
    
    # Create figure
    n_models = len(outputs)
    
    # Define grid size
    if rows is None or cols is None:
        rows = 1
        cols = n_models
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    
    # Flatten axes for easier handling
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # First, compute all confusion matrices
    cms = {}
    for model_name, y in outputs.items():
        y_true, y_pred = y["y"], y["y_pred"]
        y_pred = np.round(y_pred)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        cms[model_name] = cm
    
    # Normalize color limits across all models
    vmax = max(cm.max() for cm in cms.values())
    
    # Plot each confusion matrix
    ims = []
    for ax, (model_name, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=vmax)
        ims.append(im)
        ax.set_title(model_name,fontsize=18,fontweight="bold")
        ax.set_xlabel('Predicted label',fontsize=15,fontweight="bold")
        ax.set_ylabel('True label',fontsize=15,fontweight="bold")
        
        # Set tick marks
        classes = ['noise', 'earthquake']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45,fontsize=15)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes,fontsize=15)
        
        # Write numbers inside cells
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=18)

    # Hide unused axes if any
    for ax in axes[len(cms):]:
        ax.axis('off')
    
    # Add shared colorbar without overlapping
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)  # Adjust 14 to your preferred size
    cbar.set_label('# Examples', fontsize=16)
    
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
    return fig, axes
    
def plot_detection_roc_curves(model_outputs, save_path=None):
    """
    Plot ROC curves for detection outputs of multiple models in a single axis.

    Args:
        model_outputs (dict): Dictionary with model names as keys and file paths as values.
        save_path (str, optional): If provided, saves the figure at this path with dpi=300.

    Returns:
        fig (matplotlib.figure.Figure): The created matplotlib figure.
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axis.
    """
    # Load all outputs
    outputs = load_detection_outputs(model_outputs)
    
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, y in outputs.items():
        y_true = y["y"]
        y_score = np.round(y["y_pred"])  # Probabilities or continuous scores needed for ROC
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=3, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title('ROC Curves', fontsize=20, fontweight="bold")
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
    return fig, ax

if __name__ == "__main__":
    
    ################################# plot history 
    # label_p = "Perceptron"
    # json_path_p = f"/home/edc240000/DeepML/output/models/detection/{label_p}/best/training_history_{label_p}.json"
    # fig_path_p = f"/home/edc240000/DeepML/output/models/detection/training_history_{label_p}.png"
    # plot_training_history(json_path=json_path_p,save_path=fig_path_p)
    
    # label_dnn = "DNN"
    # json_path_dnn = f"/home/edc240000/DeepML/output/models/detection/{label_dnn}/best/training_history_{label_dnn}.json"
    # fig_path_dnn = f"/home/edc240000/DeepML/output/models/detection/training_history_{label_dnn}.png"
    # plot_training_history(json_path=json_path_dnn,save_path=fig_path_dnn)
    
    # label_cnnse = "CNNSE"
    # json_path_cnnse = f"/home/edc240000/DeepML/output/models/detection/{label_cnnse}/best/training_history_{label_cnnse}.json"
    # fig_path_cnnse = f"/home/edc240000/DeepML/output/models/detection/training_history_{label_cnnse}.png"
    # plot_training_history(json_path=json_path_cnnse,save_path=fig_path_cnnse)
    
    
    # label_cnnde = "CNNDE"
    # json_path_cnnde = f"/home/edc240000/DeepML/output/models/detection/{label_cnnde}/best/training_history_{label_cnnde}.json"
    # fig_path_cnnde = f"/home/edc240000/DeepML/output/models/detection/training_history_{label_cnnde}.png"
    # plot_training_history(json_path=json_path_cnnde,save_path=fig_path_cnnde)
    
    # fig_path = "/home/edc240000/DeepML/output/models/detection/training_stats.png"
    # json_paths = [json_path_p,json_path_dnn,json_path_cnnse,json_path_cnnde]
    # label_paths = [label_p,label_dnn,label_cnnse,label_cnnde]
    # fig, ax = plot_multiple_histories(json_paths,  label_paths,
    #                                 save_path=fig_path, 
    #                                 dpi=300)
    
    
    # ###### plot map ##########
    # import sys
    # import os
    
    # path = "/home/edc240000/DeepML"
    # sys.path.append(path)
    # root = "/groups/igonin/.seisbench"
    # os.environ["SEISBENCH_CACHE_ROOT"] = root

    # # ##### tx dataset #####
    # import seisbench.data as sbd
    # import seisbench.generate as sbg
    # from utils import create_sample_mask
    
    # save_path = "/home/edc240000/DeepML/output/figures/original_map.png"
    # filt_save_path = "/home/edc240000/DeepML/output/figures/filtered_map.png"
    # data = sbd.TXED()
    # plot_map(data.metadata,states=True,save_path=save_path,
    #          xlim=(-108.5,-92),ylim=(25,38))
    
    # n_events = 2500
    # n_noise = 2500

    # noise_mask = create_sample_mask(metadata=data.metadata,category="noise",
    #                                 n_samples=n_noise,random_state=42)
    # event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
    #                                 n_samples=n_events,min_mag=2,random_state=42)

    # data.filter(noise_mask | event_mask)
    
    # plot_map(data.metadata,states=True,save_path=filt_save_path,
    #          xlim=(-108.5,-92),ylim=(25,38))


    
    ###### plot distribution  ######################
    
    # import sys
    # import os
    
    # path = "/home/edc240000/DeepML"
    # sys.path.append(path)
    # root = "/groups/igonin/.seisbench"
    # os.environ["SEISBENCH_CACHE_ROOT"] = root

    # # ##### tx dataset #####
    # import seisbench.data as sbd
    # import seisbench.generate as sbg
    # from utils import create_sample_mask
    
    # save_path = "/home/edc240000/DeepML/output/figures/original_distribution.png"
    # filt_save_path = "/home/edc240000/DeepML/output/figures/filtered_distribution.png"
    # data = sbd.TXED()
    
    # xlims = {
    # "depth": (0, 15),
    # "magnitude": (-1, 5),
    # "distance": (0, 400),
    # "sp_diff": (0, 4000)
    # }
    
    # fig, axes = plot_metadata_distributions(data.metadata, 
    #                                         save_path=save_path,
    #                                         xlims=xlims)
    
    # n_events = 2500
    # n_noise = 2500

    # noise_mask = create_sample_mask(metadata=data.metadata,category="noise",
    #                                 n_samples=n_noise,random_state=42)
    # event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
    #                                 n_samples=n_events,min_mag=2,random_state=42)

    # data.filter(noise_mask | event_mask)
    
    
    # ############ plot examples ###############
    
    # import sys
    # import os
    
    # path = "/home/edc240000/DeepML"
    # sys.path.append(path)
    # root = "/groups/igonin/.seisbench"
    # os.environ["SEISBENCH_CACHE_ROOT"] = root

    # # ##### tx dataset #####
    # import seisbench.data as sbd
    # from torch.utils.data import DataLoader
    # from utils import create_sample_mask, prepare_data_generators
    
    
    # magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"
    # save_path = "/home/edc240000/DeepML/output/figures/scalar_detection.png"
    # batch_size = 100
    
    # data = sbd.TXED()
    
    # generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
    
    # plot_scalar_detection_examples(generators["generator_train"],save_path=save_path)
    
    
    ############ plot 3d seismic signal ############3
    # import sys
    # import os
    
    # path = "/home/edc240000/DeepML"
    # sys.path.append(path)
    # root = "/groups/igonin/.seisbench"
    # os.environ["SEISBENCH_CACHE_ROOT"] = root

    # # ##### tx dataset #####
    # import seisbench.data as sbd
    # from torch.utils.data import DataLoader
    # from utils import create_sample_mask, prepare_data_generators
    
    
    # magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"
    # save_path = "/home/edc240000/DeepML/output/figures/scalar_detection.png"
    # # index = 300000 #evenet
    # index = 3
    
    # data = sbd.TXED()
    
    # generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
    # sample = generators["generator_train"][index]
    # fig, axes = plt.subplots(3,1,figsize=(10, 7))
    
    # x = sample["X"].T
    # axes[0].plot(x[:,0],color="black",label="Z")
    # axes[1].plot(x[:,1],color="red",label="N")
    # axes[2].plot(x[:,2],color="blue",label="E")
    
    # axes[0].legend(loc="upper right",fontsize=16)
    # axes[1].legend(loc="upper right",fontsize=16)
    # axes[2].legend(loc="upper right",fontsize=16)
    # # path = "/home/edc240000/DeepML/output/figures/3d_Seismic_signal.png"
    # path = "/home/edc240000/DeepML/output/figures/3d_noise_signal.png"
    # plt.savefig(path,dpi=300)
    
    
    ################# detection confussion matrix and ROC###########
    # model_outputs = {
    #     "Perceptron": "/home/edc240000/DeepML/output/model_outputs/detection/Perceptron/Perceptron_outputs.pt",
    #     "DNN": "/home/edc240000/DeepML/output/model_outputs/detection/DNN/DNN_outputs.pt",
    #     "CNNSE": "/home/edc240000/DeepML/output/model_outputs/detection/CNNSE/CNNSE_outputs.pt",
    #     "CNNDE": f"/home/edc240000/DeepML/output/model_outputs/detection/CNNDE/CNNDE_outputs.pt"
    # }
    # save_path = "/home/edc240000/DeepML/output/figures/detection_confussion_matrix.png"
    # fig, axes = plot_detection_confusion_matrix(model_outputs, save_path=save_path,rows=2, cols=2)
    # save_path = "/home/edc240000/DeepML/output/figures/detection_ROC.png"
    # plot_detection_roc_curves(model_outputs, save_path=save_path)
    
    
    
    ############### PLOT TEST EXAMPLES ############3
    
    
    import sys
    path = "/home/edc240000/DeepML"
    sys.path.append(path)

    # ##### tx dataset #####
    import seisbench.data as sbd
    from torch.utils.data import DataLoader
    from torch.utils.data import Subset, DataLoader
    
    from DeepML.scalar_detection.models import CNNSE,CNNDE,DNN,Perceptron,DetectionLoss
    from utils import (create_sample_mask, prepare_data_generators,
                        load_detection_outputs, get_scalar_detection_predictions)
    
    
    magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"
    batch_size = 100
    
    data = sbd.TXED()
    n_events = 2500
    n_noise = 2500
    nrows, ncols = 5,5

    noise_mask = create_sample_mask(metadata=data.metadata,category="noise",
                                    n_samples=n_noise,random_state=42)
    event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                    n_samples=n_events,min_mag=2,random_state=42)

    data.filter(noise_mask | event_mask)
    
    generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
    test_loader = DataLoader(generators["generator_test"], batch_size=100, shuffle=False)
    
    n_samples = 20
    full_test_dataset = generators["generator_test"]
    all_indices = list(range(len(full_test_dataset)))
    random_indices = random.sample(all_indices, n_samples)

    # Create a smaller dataset
    small_test_dataset = Subset(full_test_dataset, random_indices)

    # New DataLoader just for those samples
    small_test_loader = DataLoader(small_test_dataset, batch_size=n_samples, shuffle=False)
    
    
    model_classes = {
    "Perceptron": Perceptron,
    "DNN": DNN,
    "CNNSE": CNNSE,
    "CNNDE": CNNDE,
    }
    
    model_paths = dict((x, f"/home/edc240000/DeepML/output/models/detection/{x}/best/best_model_{x}.pt") for x in model_classes.keys())
    predictions = get_scalar_detection_predictions(model_classes=model_classes,
                                model_paths=model_paths,
                                data_loader=small_test_loader,
                                load_y=True,load_x=True)
    # print(predictions)
    # n_traces = np.random.randint(low=0, high=len(generators["generator_test"]),
    #                              size=int(nrows*ncols))
    # trace_loader = {}
    # for idx in n_traces:
    #     sample = generators["generator_test"][idx]
        # x = sample["X"]
        # y_scalar_detection = sample["y_scalar_detection"].squeeze()
        # y_pred = test_predictions[sample_index]
        
    # print(n_traces)
    # print(n_traces.shape)
    # predictions = get_predictions(model_classes=model_classes,
    #                               model_paths=model_paths,
    #                               data_loader=DataLoader)
    # print(predictions)
    # outputs = load_detection_outputs(model_outputs=model_outputs)
    name2plot = "CNNDE"
    
    # # # print(len(generators["generator_test"]))
    # # # exit()
    
    # # # save_path = "/home/edc240000/DeepML/output/figures/detection_confussion_matrix.png"
    # # # fig, axes = plot_detection_confusion_matrix(model_outputs, save_path=save_path,rows=2, cols=2)
    savedir_path = "/home/edc240000/DeepML/output/figures/detection_test_examples"
    # # test_predictions = outputs[name2plot]["y_pred"]
    plot_scalar_detection_test_examples(predictions=predictions,
                                        savedir_path=savedir_path,
                                        cols=4,rows=4)
    
    # plot_SCA_roc_curves(model_outputs, save_path=save_path)