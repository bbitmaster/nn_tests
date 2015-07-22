import numpy as np
import sys

data_dir = '../data/'
co_file = 'gas_data/ethylene_CO.txt'
methane_file = 'gas_data/ethylene_methane.txt'

def load_gas_data(num_samples=1e99,correct=1.0,incorrect=-1.0):
    co_f = open(data_dir + co_file)
    #skip header
    co_f.next()
    co_data = []
    co_sample_count = 0
    for line in co_f:
        data_line = line.split()
        data_line = [float(d) for d in data_line]
        co_data.append(np.array(data_line))
        co_sample_count+=1
        if(co_sample_count == num_samples):
            break
    co_f.close()
    
    methane_f = open(data_dir + methane_file)
    methane_f.next()
    methane_data = []
    methane_sample_count=0
    for line in methane_f:
        data_line = line.split()
        data_line = [float(d) for d in data_line]
        methane_data.append(np.array(data_line))
        methane_sample_count+=1
        if(methane_sample_count == num_samples):
            break
    methane_f.close()

    co_data = np.array(co_data)
    co_data = co_data.astype(np.float32)
    methane_data = np.array(methane_data)
    methane_data = methane_data.astype(np.float32)

    gas_data = np.append(co_data,methane_data,axis=0)

    gas_class = np.append( \
            np.tile(np.array([correct,incorrect],dtype=np.float32),[co_sample_count,1]),
            np.tile(np.array([incorrect,correct],dtype=np.float32),[methane_sample_count,1]),axis=0)
    gas_class = gas_class.astype(np.float32)

    gas_data = gas_data - np.mean(gas_data,axis=0)
    gas_data = gas_data/np.std(gas_data,axis=0)

    return (gas_data,gas_class)

if __name__ == '__main__':
    d = load_gas_data()
    print(d[0].shape)
    print(d[1].shape)
