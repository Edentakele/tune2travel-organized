# csv_to_parquet.py
import csv
import pyarrow as pa
import pyarrow.parquet as pq

csv_file = './despacito_api_dedup.csv'
parquet_file = './see_you_api_dedup.parquet'
chunksize = 100_000

# Open the CSV file using the csv library
with open(csv_file, mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    # Skip the header if there is one
    next(reader)
    
    # Initialize Parquet writer with an empty table schema
    parquet_writer = None
    for i, chunk in enumerate(iter(lambda: list(next(reader) for _ in range(chunksize)), [])):
        print("Chunk", i)
        if i == 0:
            # Guess the schema of the CSV file from the first chunk
            header = chunk[0]
            parquet_schema = [pa.field(name, pa.string()) for name in header]
            # Open a Parquet file for writing
            parquet_writer = pq.ParquetWriter(parquet_file, schema=pa.schema(parquet_schema), compression='snappy')
        
        # Convert chunk to PyArrow table and write it to the Parquet file
        table = pa.table([chunk])
        parquet_writer.write_table(table)

# Close the Parquet writer
if parquet_writer is not None:
    parquet_writer.close()