#!/usr/bin/env python
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager, cpu_count
from itertools import chain

# Dummy transformation function (replace with your actual logic)
def compute_transformation(data):
    return data * 2

# Worker function for processing chunks
def worker_chunk(chunk_id, indices, data_df, return_dict):
    results = []
    for i in tqdm(indices, position=chunk_id, desc=f"Worker {chunk_id}", leave=True,mininterval=0.5):
        row = data_df.iloc[i]
        transformed = compute_transformation(row["value"])
        results.append((i, transformed))
    return_dict[chunk_id] = results

# Worker function for merging chunks
def merge_worker(results_chunk, return_list):
    df = pd.DataFrame(results_chunk, columns=["index", "result"])
    return_list.append(df.set_index("index"))

def main():
    # Dummy data
    data = pd.DataFrame({"value": np.random.rand(1000000)})
    data["result"] = np.nan

    # Multiprocessing setup
    num_chunks = 8 # cpu_count()
    index_chunks = np.array_split(range(len(data)), num_chunks)

    manager = Manager()
    return_dict = manager.dict()
    processes = []

    # Start worker processes
    for i, idx_chunk in enumerate(index_chunks):
        p = Process(target=worker_chunk, args=(i, idx_chunk, data, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect and split all results
    all_results = list(chain.from_iterable(return_dict.values()))
    result_chunks = np.array_split(all_results, num_chunks)

    # Parallel merging
    partial_results = manager.list()
    merge_processes = []

    for chunk in result_chunks:
        p = Process(target=merge_worker, args=(chunk, partial_results))
        p.start()
        merge_processes.append(p)

    for p in merge_processes:
        p.join()

    # Combine partial DataFrames and update main DataFrame
    merged_df = pd.concat(partial_results)
    data.loc[merged_df.index, "result"] = merged_df["result"].values

    # # Save output
    # data.to_csv("output.csv", index=False)
    # print("Saved results to output.csv")

if __name__ == "__main__":
    main()
