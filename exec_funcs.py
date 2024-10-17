import polars as pl
import numpy as np
import time
import ctypes
import pathlib
from threading import Thread
from pim_process import pim_agg_calc

def get_ordered_dataset(df, col_list):
	selected_df = df[col_list].with_row_index()
	
	sorted_df = selected_df.sample(fraction=1, shuffle=True)
	sorted_df = sorted_df.sort(by=col_list, maintain_order=True)
	sorted_df = sorted_df.rename({"index" : "orig_index"})
	sorted_df = sorted_df.with_row_index()
	sorted_df = sorted_df.rename({"index" : "sorted_index"})
	sorted_df = sorted_df.rename({"orig_index" : "index"})
	sorted_df = sorted_df.select(["index", "sorted_index"])

	ret_df = selected_df.join(sorted_df, on="index", validate="1:1", how="left")

	return ret_df.select("sorted_index")

def preprocess_dataset(df, columns_to_inspect):
	processed = 0
	for columns in columns_to_inspect:
		print("Processed " + str(processed))
		processed += 1
		ordered_df = get_ordered_dataset(df, columns)
		df = pl.concat([df, ordered_df], how="horizontal")
		df = df.rename({'sorted_index' : '+'.join(columns) + "_sorted"})
	df = df.sample(fraction=1.0, shuffle=True)
	return df

def bucketize_dataset(sorted_df, bucket_num, columns_to_inspect, total_rows):
	bucketize_clib = ctypes.CDLL(pathlib.Path().absolute() / "bucketize_data.so")

	start = time.time()

	def bucketize_col(sorted_col, ret_list, index):
		sorted_idxs = sorted_col.to_numpy()
		bucket_idxs = np.zeros(len(sorted_col)).astype(np.uint8)
		bucketize_clib.bucketize_data(sorted_idxs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(sorted_idxs),
		  total_rows, bucket_idxs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), bucket_num)
		bucket_idxs = pl.from_numpy(bucket_idxs, schema=[sorted_col.name + "_bucket"])
		ret_list[index] = bucket_idxs

	bucket_idxs_list = [None] * len(columns_to_inspect)
	threads = []

	for i in range(len(columns_to_inspect)):
		thread = Thread(target=bucketize_col, args=(sorted_df['+'.join(columns_to_inspect[i]) + "_sorted"], bucket_idxs_list, i))
		thread.start()
		threads.append(thread)

	for t in threads:
		t.join()

	bucket_df = pl.concat(bucket_idxs_list, how="horizontal")
	end = time.time()
	print(end - start)
	return bucket_df

def get_val_df(df_analyze, val_columns_and_types):
	casted_dfs = []
	for v in val_columns_and_types:
		if v[0] is not None:
			casted_dfs.append(df_analyze[v[0]].cast(v[1]).to_frame())
		else:
			casted_dfs.append(pl.from_numpy(np.ones(len(df_analyze)), schema=["count"]).cast(v[1]))

	return pl.concat(casted_dfs, how="horizontal")

def sort_pim_agg(pim_agg, cols):
	ret = pim_agg
	tmp_cols = cols.copy()
	index_min = np.argmin(tmp_cols)
	if index_min != 0:
		ret = np.swapaxes(ret, 0, index_min)
		tmp_cols[0] = cols[index_min]
		tmp_cols[index_min] = cols[0]
	index_min = np.argmin(tmp_cols[1:])
	if index_min != 0:
		ret = np.swapaxes(ret, 1, 2)
	cols.sort()
	return (ret, cols)

def sort_pim_aggs(pim_aggs, pim_agg_cols):
	ret_aggs = []
	ret_cols = []
	for i in range(len(pim_agg_cols)):
		sort_ret = sort_pim_agg(pim_aggs[i], pim_agg_cols[i])
		ret_aggs.append(sort_ret[0])
		ret_cols.append(sort_ret[1])
	ret_aggs = [agg for _, agg in sorted(zip(ret_cols, ret_aggs))]
	return (sorted(ret_cols), np.array(ret_aggs))


def equidistance_bin(df, columns_to_inspect, bins):
	result_dfs = []
	for cols in columns_to_inspect:
		s = df[cols[0]]
		bin_col_name = '+'.join(cols) + "_edbinned"
		if s.dtype.is_numeric():
			val_np = s.to_numpy()
			min = val_np.min()
			max = val_np.max()
			boundaries = np.linspace(min, max, bins + 1)
			boundaries[-1] += 1
			binned_data = (np.digitize(val_np, boundaries) - 1).astype(np.uint32)
			result_df = pl.from_numpy(binned_data, schema=[bin_col_name])
		else:
			sub_df = df[cols]
			#sorted_df = sub_df.sort(by=cols, maintain_order=True).group_by(cols[0], maintain_order=True).len()
			cols_to_bins = sub_df.sort(by=cols, maintain_order=True).unique(maintain_order=True).with_row_index()
			cols_to_bins = cols_to_bins.rename({"index": bin_col_name})
			joined_df = sub_df.join(cols_to_bins, left_on=cols, right_on=cols, validate="m:1")
			result_df = joined_df[bin_col_name].to_frame()
		result_dfs.append(result_df)
	return pl.concat(result_dfs, how="horizontal")

def type_to_int(nptype):
	if nptype == np.int32:
		return 0
	elif nptype == np.float32:
		return 1
	elif nptype == np.float64 or nptype == np.double:
		return 2

def print_exec_info(row_num, col_num, bucket_num, data_type):
	print("ROWS: " + str(row_num))
	print("COLS: " + str(col_num))
	print("BUCKETS: " + str(bucket_num))
	print("DATA TYPE: " + str(data_type))

def calculate_aggregates_baseline(exec_method, bucket_df, val_series, bucket_num, threads=1, divide=1):
	if exec_method == 2:
		rows = bucket_df.to_numpy()
	else:
		rows = bucket_df.to_numpy(order="c")
	
	row_num = rows.shape[0]
	col_num = rows.shape[1]

	baseline_clib = ctypes.CDLL(pathlib.Path().absolute() / "baseline.so")

	vals = val_series.to_numpy()
	data_type = type_to_int(vals.dtype)

	num_aggs = int(col_num * (col_num - 1) * (col_num - 2) / 6)
	aggs = np.zeros(num_aggs * bucket_num * bucket_num * bucket_num, dtype=vals.dtype)

	print("BASELINE: " + str(exec_method))
	print("THREADS: " + str(threads))
	print("DIVIDE: " + str(divide))
	print_exec_info(row_num, col_num, bucket_num, vals.dtype)
	
	start = time.time()

	if exec_method == 2:
		rows = rows.T

	agg_info = []
	for i in range(0, col_num - 2):
		for j in range(i + 1, col_num - 1):
			for k in range(j + 1, col_num):
				agg_info.append([i,j,k])

	rows = rows.ravel()

	rows_ptr = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
	vals_ptr = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
	aggs_ptr = aggs.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

	preprocess_end = time.time()

	print("READY TIME: " + str(preprocess_end - start))

	if exec_method == 0:
		baseline_clib.baseline(rows_ptr, vals_ptr, aggs_ptr, row_num, col_num, bucket_num, data_type)
	elif exec_method == 1:
		baseline_clib.baseline_threaded(rows_ptr, vals_ptr, aggs_ptr, row_num, col_num, bucket_num, threads, data_type)
	elif exec_method == 2:
		baseline_clib.baseline_divide_threaded(rows_ptr, vals_ptr, aggs_ptr, row_num, col_num, bucket_num, threads, divide, data_type)

	end = time.time()
	print("EXEC TIME: " + str(end - preprocess_end))

	return agg_info, aggs

def calculate_aggregates_pim(bucket_df, val_series, bucket_num, buffer_size, dpu_num, async_mode=False, simulator=False, correctness_check=False):
	rows = bucket_df.to_numpy(order="c")
	row_num = rows.shape[0]
	col_num = rows.shape[1]

	vals = val_series.to_numpy()
	data_type = type_to_int(vals.dtype)

	num_aggs = int(col_num * (col_num - 1) * (col_num - 2) / 6)
	aggs = np.zeros(num_aggs * bucket_num * bucket_num * bucket_num, dtype=vals.dtype)

	print("PIM")
	print("DPUS: " + str(dpu_num))
	print("ASYNC MODE: " + str(async_mode))
	print("BUFFER SIZE: " + str(buffer_size))
	print_exec_info(row_num, col_num, bucket_num, vals.dtype)

	start = time.time()

	buffers = []
	for _ in range(dpu_num):
		buffers.append(np.zeros(buffer_size, dtype=np.uint8))

	rows = rows.ravel()

	preprocess_end = time.time()

	print("READY TIME: " + str(preprocess_end - start))

	agg_info = pim_agg_calc(rows, vals, aggs, row_num, col_num, buffers, data_type, False, bucket_num, dpu_num, async_mode, simulator)

	end = time.time()
	print("EXEC TIME: " + str(end - preprocess_end))

	if correctness_check:
		check_agg_info, check_aggs = calculate_aggregates_baseline(1, bucket_df, val_series, bucket_num, 20)
		check_aggs = check_aggs.reshape([num_aggs, bucket_num, bucket_num, bucket_num])
		_, check_aggs_sorted = sort_pim_aggs(check_aggs, check_agg_info)

		aggs = aggs.reshape([len(agg_info), bucket_num, bucket_num, bucket_num])
		agg_info, aggs = sort_pim_aggs(aggs, agg_info)
		
		eval_result = np.isclose(check_aggs_sorted, aggs)

		if eval_result.all():
			print("Correct")
		else:
			print("Wrong")

	return agg_info, aggs










