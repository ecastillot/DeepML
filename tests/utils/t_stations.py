import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)

##### stations from texnet #####
from obspy.clients.fdsn.client import Client
from DeepML.utils_bck.stations import get_stations_info

provider = "texnet"
output_folder = "/groups/igonin/.seisbench/datasets/txed"

client =  Client(provider)
inv = client.get_stations(network="*", station="FW04", level="response")
print(inv)
# sta_info = get_stations_info(inv,output_folder)
