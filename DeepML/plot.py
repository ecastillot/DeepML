import json
import matplotlib.pyplot as plt
import numpy as np
import random


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

    if history.get("train_det_loss"):
        ax1.plot(epochs, history["train_det_loss"], label="Train Det Loss", linestyle="--")
    if history.get("dev_det_loss"):
        ax1.plot(epochs, history["dev_det_loss"], label="Dev Det Loss", linestyle="--")

    if history.get("train_mag_loss"):
        ax1.plot(epochs, history["train_mag_loss"], label="Train Mag Loss", linestyle="--")
    if history.get("dev_mag_loss"):
        ax1.plot(epochs, history["dev_mag_loss"], label="Dev Mag Loss", linestyle="--")

    ax1.set_ylabel("Loss")
    ax1.set_title("Loss History")
    ax1.legend()
    ax1.grid(True)

    # --- Plot Accuracies ---
    if history.get("train_acc"):
        ax2.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    if history.get("dev_acc"):
        ax2.plot(epochs, history["dev_acc"], label="Dev Accuracy", marker="o")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy History")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()

    # Optional: Save the figure
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path} at {dpi} dpi")

    return fig, (ax1, ax2)


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

def export_model_info(model, input_shape, export_base_path):
    """
    Exports a PNG of the model architecture (torchviz) and a TXT summary (torchinfo).

    Parameters:
    - model: PyTorch model instance
    - input_shape: tuple, e.g., (1, 3, 6000)
    - export_base_path: str, base file path without extension
                        e.g., "exports/my_model"
    """
    os.makedirs(os.path.dirname(export_base_path), exist_ok=True)

    # Set model to eval
    model.eval()

    # Dummy input for visualization
    dummy_input = torch.randn(*input_shape)

    # -----------------------
    # Export graph (torchviz)
    # -----------------------
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(export_base_path, cleanup=True)
    print(f"[✓] Graph saved at: {export_base_path}.png")

    # -----------------------
    # Export summary (torchinfo)
    # -----------------------
    model_summary = summary(model, input_size=input_shape, verbose=0)
    summary_text = str(model_summary)
    summary_path = f"{export_base_path}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"[✓] Summary saved at: {summary_path}")

# if __name__ == "__main__":
    
    # json_path = "/home/edc240000/DeepML/DeepML/training_history.json"
    # fig_path = "/home/edc240000/DeepML/DeepML/training_history.png"
    # fig, ax = plot_training_history( json_path , 
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
    
    
    
    