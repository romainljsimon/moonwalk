import pyarrow as pa
import pyarrow.parquet as pq
import struct



def write_data(time, j, data, column_names):
    # Convert positions and observables to Apache Arrow arrays
    data_arrays = [pa.array(data[:, i]) for i in range(data.shape[1])]

    
    # Combine position and observable arrays into one array for table creation
    data_table = pa.table(data_arrays, names=column_names)
    
    # Create and store custom metadata for each time snapshot, with the current time
    custom_metadata = {'time': struct.pack('f', time)}
    existing_metadata = data_table.schema.metadata
    merged_metadata = {**custom_metadata, **existing_metadata} if existing_metadata else custom_metadata
    data_table = data_table.replace_schema_metadata(merged_metadata)
    # Write the data to a Parquet file for the current time step
    pq.write_table(data_table, f'out/out{j}.parquet')