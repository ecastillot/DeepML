import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)



# ##### tx dataset #####
import os
root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root


from DeepML.utils.magnitude import get_tx_dataset, get_obspy_waveforms
data = get_tx_dataset(test=False)
print(data.metadata.info())


inventory_folder = "/groups/igonin/.seisbench/datasets/txed/stations"
stream = get_obspy_waveforms(data, 340055,inventory_folder)
print(stream)

# waveforms = data.get_waveforms(3)
# print("waveforms.shape:", waveforms.shape)

# import matplotlib.pyplot as plt
# plt.plot(waveforms.T);
# plt.show()
# # #############################

