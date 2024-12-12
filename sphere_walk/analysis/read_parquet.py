import numpy as np
import pyarrow.parquet as pq

class ReadParquet:

    def __init__(self, path):
        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read()
        arrays_list = [column.to_numpy() for column in table.columns]
        self.data_array = np.column_stack(arrays_list)
        self.column_names = table.column_names
        self.metadata = table.schema.metadata