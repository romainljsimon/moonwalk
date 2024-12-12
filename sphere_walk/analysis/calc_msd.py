import numpy as np
from .read_parquet import ReadParquet
import pyarrow as pa
import pyarrow.parquet as pq
from .file_util import get_all_files
import struct

class MSD:

    def __init__(self, path_read, path_write):
        path_list = get_all_files(path_read)
        init_parquet = ReadParquet(path_list[0])
        self.init_observable_array = init_parquet.data_array
        self.column_names = init_parquet.column_names
        self.len_column = len(self.column_names)
        if self.len_column == 1:
            self.msd_array = np.zeros(len(path_list))
        else:
            self.msd_array = np.zeros((len(path_list), len(self.column_names)))

        self.time_array = np.zeros(len(path_list))
        self.calc_msd(path_list)
        shape = self.msd_array.shape
        if len(shape) == 1:
            observable_arrays = [pa.array(self.msd_array)]
        else:
            observable_arrays = [pa.array(self.msd_array[:, i]) for i in range(self.msd_array.shape[1])]
        observable_arrays = [pa.array(self.time_array)] + observable_arrays
        observable_table = pa.Table.from_arrays(observable_arrays, names=["time"]+self.column_names) 
        pq.write_table(observable_table, f'{path_write}/msd.parquet')

    def calc_msd(self, path_list):
        for i, elt in enumerate(path_list):
            elt_parquet = ReadParquet(elt)
            t_observable_array = elt_parquet.data_array
            if self.len_column == 1:
                self.msd_array[i] = np.nanmean((t_observable_array - self.init_observable_array)**2)
            else:
                mean_observable = np.nanmean(t_observable_array, axis=0)
                diff = t_observable_array - self.init_observable_array #- mean_observable
                self.msd_array[i] = np.nanmean(diff**2, axis=0)
            self.time_array[i] = struct.unpack('f', elt_parquet.metadata[b'time'])[0]