import pandas as pd
import os

def get_station_info_flat(station, network_code):
    """
    Extract flattened station and channel information from an ObsPy Station object.

    Args:
        station (obspy.core.inventory.station.Station): ObsPy Station object.
        network_code (str): Code of the network the station belongs to.

    Returns:
        list of dict: Each dict contains combined station and channel information for one channel.
    """
    rows = []

    for channel in station:
        row = {
            # Network and station-level information
            "network": network_code,
            "station": station.code,
            "sta_id": f"{network_code}.{station.code}",
            "latitude": station.latitude,
            "longitude": station.longitude,
            "elevation": station.elevation,
            "station_starttime": station.start_date.datetime,
            "station_endtime": station.end_date.datetime if station.end_date else None,
            "site_name": station.site.name if station.site else None,

            # Channel-level information
            "channel": channel.code,
            "location": channel.location_code,
            "sampling_rate": channel.sample_rate,
            "channel_starttime": channel.start_date.datetime,
            "channel_endtime": channel.end_date.datetime if channel.end_date else None
        }

        # Instrument sensitivity (if available)
        sensitivity = (
            channel.response.instrument_sensitivity
            if channel.response and channel.response.instrument_sensitivity
            else None
        )

        if sensitivity:
            row["sensitivity"] = sensitivity.value
            row["sensitivity_input_units"] = sensitivity.input_units
            row["sensitivity_output_units"] = sensitivity.output_units
        else:
            row["sensitivity"] = None
            row["sensitivity_input_units"] = None
            row["sensitivity_output_units"] = None

        rows.append(row)

    return rows


def get_stations_info(inventory, output_folder):
    """
    Extract flattened station and channel information from an ObsPy Inventory object.

    Args:
        inventory (obspy.core.inventory.inventory.Inventory): Inventory containing network and station data.

    Returns:
        pandas.DataFrame: Flattened DataFrame where each row represents a unique station-channel pair.
    """
    stations_folder = os.path.join(output_folder, "stations")
    os.makedirs(stations_folder, exist_ok=True)
    
    rows = []

    for network in inventory:
        for station in network:
            net_sta_id = f"{network.code}.{station.code}"
            inv = inventory.select(network=network.code, station=station.code)
            inv.write(os.path.join(stations_folder, f"{net_sta_id}.xml"),
                      format="STATIONXML")
        
            rows.extend(get_station_info_flat(station, network.code))

    df = pd.DataFrame(rows)

    # Optional: reorder columns for readability
    first_cols = ["sta_id", "network", "station", "channel", "location"]
    df = df[first_cols + [col for col in df.columns if col not in first_cols]]
    # Save the DataFrame to a CSV file
    csv_file = os.path.join(stations_folder, "stations_info.csv")
    df.to_csv(csv_file, index=False)
    
    return df
