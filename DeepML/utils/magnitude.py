import math
import os
import seisbench
import seisbench.data as sbd
import numpy as np
from obspy.core.inventory.inventory import Inventory
from obspy.io.xseed.parser import Parser
from obspy.core.trace import Trace,Stats
from obspy.core.stream import Stream
from obspy.core.inventory.inventory import read_inventory
import glob
from .stations import get_station_info_flat

paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
          'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

def calculate_minus_log_A0(distance_km):
    """
    Calculate -log(A0) based on the given piecewise formula.
    
    Parameters:
    distance_km (float): Hypocentral distance in kilometers
    
    Returns:
    float: Value of -log(A0)
    """
    if distance_km <= 16:
        # For distance_km <= 16 km
        return 2.07 * math.log10(distance_km) + 0.0002 * (distance_km - 100) - 0.72
    elif 16 < distance_km <= 105:
        # For 16 km < distance_km <= 105 km
        return 1.54 * math.log10(distance_km) + 0.0002 * (distance_km - 100) - 0.08
    else:
        # For distance_km > 105 km
        return 0.29 * math.log10(distance_km) + 0.0002 * (distance_km - 100) + 2.45
    
def get_tx_dataset(test=True,debug =False):
    
    if test:
        print(f"Your dataset will be in {seisbench.cache_root}")
        return None
    
    data = sbd.TXED()
    
    if debug:
        print("Cache root:", seisbench.cache_root)
        print("Contents:", os.listdir(seisbench.cache_root))
        print("datasets:", os.listdir(seisbench.cache_root / "datasets"))
        print("txed:", os.listdir(seisbench.cache_root / "datasets" / "txed"))
    
    return data

def get_obspy_waveforms(data, idx, inventory_folder=None):
    
    if isinstance(data, sbd.TXED):
        waveforms = data.get_waveforms(idx)
    else:
        raise ValueError("data must be of type seisbench.data.TXED")
    
    
    metadata = data.metadata.iloc[idx]
    
    stream = Stream()
    for channel in range(len(metadata["trace_component_order"])):
        trace = Trace()
        trace_name_original = metadata["trace_name_original"]
        ev_id, sta_id, ev_type = trace_name_original.split("_")
        trace.data = waveforms[channel]
        trace.stats.network = "TX" 
        trace.stats.station = sta_id
        trace.stats.event_type = ev_type
        trace.stats.source_origin_time = metadata["source_origin_time"]
        trace.stats.event = idx
        trace.stats.event_real = ev_id
        trace.stats.channel = metadata["trace_component_order"][channel]
        trace.stats.sampling_rate = metadata["trace_sampling_rate_hz"]
        trace.stats.starttime = metadata["source_origin_time"]
        
        if inventory_folder is not None:
            query__path = os.path.join(inventory_folder, f"*.{sta_id}.xml")
            
            inventory_path = glob.glob(query__path)
            if len(inventory_path) > 0:
                inventory_path = inventory_path[0]
            else:
                print(f"Inventory file not found for {sta_id} at {query__path}")
                continue
            
            if os.path.exists(inventory_path):
                print(inventory_path)
                inventory = read_inventory(inventory_path,level="response")
                
                reference_time = trace.stats.source_origin_time
                print(inventory)
                inventory = inventory.select(network=trace.stats.network ,
                                             station=trace.stats.station,
                                            location="*",
                                            channel="*",
                                            time=reference_time,
                                            sampling_rate=trace.stats.sampling_rate)
                seed_ids = inventory.get_contents()["channels"]
                print(metadata["trace_component_order"][channel], seed_ids)
                seed_id = seed_ids[0]
                net,sta,loc,chan = seed_id.split(".")
                
                trace.stats.network = net
                trace.stats.station = sta
                trace.stats.channel = chan
                trace.stats.location = loc
                
                print(trace)
                
                print(trace.data)
                
                fig = trace.plot(method="full")
                fig.savefig("/home/edc240000/DeepML/tests/utils/trace_before.png")
                
                trace.remove_response(inventory=inventory, output="VEL")
                trace.simulate(paz_remove=None, paz_simulate=paz_wa)
                
                # trace.remove_response(inventory=inventory, output="VEL")
                # trace.simulate(paz_remove=None, paz_simulate=paz_wa)
                
                
                fig = trace.plot(method="full")
                fig.savefig("/home/edc240000/DeepML/tests/utils/trace_after.png")
                print(trace.data)
                # exit()
            else:
                print(f"Inventory file not found for {sta_id} at {inventory_path}")
        stream.append(trace)
        
    
        
        
    return stream


