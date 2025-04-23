import json
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    
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

    