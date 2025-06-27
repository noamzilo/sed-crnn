import numpy as np

arr = np.load("/home/noams/src/plai_cv/cache/decorte_metadata/features/20230528_VIGO_01_mon.npz")
print(arr['arr_0'].shape)  # train_x
print(arr['arr_1'].shape)  # train_y