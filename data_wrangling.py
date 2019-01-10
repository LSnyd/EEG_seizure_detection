import numpy as np
import glob



def wrangled_data():

    source_dir = "/home/lisa/Dokumente/SU_CD/DAMI2/EEG_seizure_detection"


    label = np.genfromtxt("/home/lisa/Dokumente/SU_CD/DAMI2/EEG_seizure_detection/EEG_Data/labels.txt", delimiter="\n", dtype=int)


    #Map to multivariate time series
    data = []
    file_list = sorted(glob.glob(source_dir + '/EEG_Data' + '/Data' + '/*.txt'))


    for file_path in file_list:
      data.append(
         np.genfromtxt(file_path, delimiter="\n", dtype=int))

    np.stack(data)

    data = np.stack(data)
    data = np.vsplit(data, 25)
    data = np.stack(data)


    #Change data type & reshape data
    data = np.array(data, dtype=np.float64)

    x_samples, x_dimensions, x_timesteps = data.shape

    data = data.reshape(x_samples, x_dimensions * x_timesteps)


    # Random shuffling
    random_state = np.random.RandomState(123)
    order = np.arange(x_samples)
    random_state.shuffle(order)
    data = data[order, :]
    label = label[order]

    return data, label, x_dimensions

