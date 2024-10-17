from dpu import DpuSet
from sys import stdout
import numpy as np
import math
from math_dist import even_dist_dpu
import ctypes
import pathlib
import time
from threading import Thread

batch_size = 40

DPU_HEAP_PTR = '__sys_used_mram_end'

def roundup(x, base=4):
    return int(math.ceil(x / base)) * base

def pad_numpy_array(arr, n):
	return np.pad(arr, (0, roundup(len(arr), n) - len(arr)), 'constant')

def dpu_exec_aggs(agg_it, rows, vals, row_num, col_num, buffers, data_type, buckets, dpu_num, dpus):
	arg_init_time = time.time()
	buffer_time = 0
	copy_time = 0
	dpu_exec_time = 0

	arg_copy_start_time = time.time()
	dpu_max_aggs = len(max(agg_it[1], key=len))
	dpu_args = []
	for aggs in agg_it[1]:
		dpu_aggs = np.pad(np.asarray(aggs, dtype=np.uint8).flatten(), (0, 3 * (dpu_max_aggs - len(aggs))), 'constant')
		dpu_aggs = np.concatenate((np.zeros(12, dtype=np.uint8), dpu_aggs.flatten()))

		# Pad array length to be multiple of 8 for dpu transfers
		dpu_aggs = pad_numpy_array(dpu_aggs, 8)
		dpu_aggs.view(dtype=np.uint32)[0] = 1
		dpu_aggs.view(dtype=np.uint32)[1] = dpu_max_aggs
		dpu_aggs.view(dtype=np.uint32)[2] = len(aggs)
		for _ in range(buckets * agg_it[0]):
			dpu_args.append(memoryview(dpu_aggs))
	dpus.copy(DPU_HEAP_PTR, dpu_args)
	dpus.sync()
	dpus.exec()

	rows_ptr = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
	vals_ptr = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

	agg_cols = len(agg_it[1])
	partitions = agg_it[0]
	buffer_progress = np.zeros(agg_cols * partitions, dtype=np.uint32)
	buffer_ends = np.zeros(agg_cols * partitions, dtype=np.uint32)
	for i in range(agg_cols):
		for j in range(partitions):
			buffer_progress[i * partitions + j] = int(row_num / partitions) * j
			buffer_ends[i * partitions + j] = int(row_num / partitions) * (j + 1) if j != partitions - 1 else row_num

	buffer_size = len(buffers[0])
	buffer_clib = ctypes.CDLL(pathlib.Path().absolute() / "buffer.so")

	#buffers_ptr = buffers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
	buffers_ptr = (ctypes.POINTER(ctypes.c_uint8) * dpu_num)()
	for i in range(dpu_num):
		buffers_ptr[i] = buffers[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
	buffer_progress_ptr = buffer_progress.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
	buffer_ends_ptr = buffer_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

	agg_cols_vals = []
	for i in range(agg_cols):
		agg_cols_vals.append(agg_it[1][i][0][0])
	agg_cols_vals = np.array(agg_cols_vals, dtype=np.uint32)
	agg_cols_vals_ptr = agg_cols_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

	#buf_reshape = buffers.reshape([dpu_num, -1])
	bufm = [memoryview(buffers[i]) for i in range(dpu_num)]

	dpus.sync()

	arg_init_time = time.time() - arg_init_time

	while not np.array_equal(buffer_progress, buffer_ends):
		buffer_start_time = time.time()
		buffer_clib.fill_buffers_c(rows_ptr, vals_ptr, col_num, buckets, buffers_ptr, 
				buffer_size, agg_cols_vals_ptr, agg_cols, partitions, buffer_progress_ptr, buffer_ends_ptr, data_type)
		buffer_end_time = time.time()
		buffer_time += buffer_end_time - buffer_start_time

		dpus.copy(DPU_HEAP_PTR, bufm)
		dpus.sync()
		dpu_copy_end_time = time.time()
		copy_time += dpu_copy_end_time - buffer_end_time

		dpus.exec()
		dpu_exec_end_time = time.time()
		dpu_exec_time += dpu_exec_end_time - dpu_copy_end_time

	dpus.sync()
		
	print("ARG INIT TIME: " + str(arg_init_time))
	print("BUFFER TIME: " + str(buffer_time))
	print("CPU->DPU xfer time: " + str(copy_time))
	print("DPU EXEC TIME: " + str(dpu_exec_time))
	print("ITERATION TIME: " + str(time.time() - arg_copy_start_time))

def pim_agg_calc(rows, vals, ret_aggs, row_num, col_num, buffers, data_type, get_counts, buckets, dpu_num, async_mode, simulator):
	profile_str = ''
	if simulator:
		profile_str += 'backend=simulator'
	with DpuSet(nr_dpus=dpu_num, binary="task", async_mode=async_mode, profile=profile_str) as dpus:
		# Init DPU arguments (one time only)
		dpu_rows = int(dpu_num / buckets)
		agg_dist = even_dist_dpu(col_num, dpu_rows)
		dpu_max_aggs = []
		for agg_it in agg_dist:
			dpu_max_aggs.append(len(max(agg_it[1], key=len)))

		init_start_time = time.time()
		get_counts_arg = 0
		if get_counts:
			get_counts_arg = 1
		buffer_size = len(buffers[0])
		dpu_arg = pad_numpy_array(np.asarray([0, col_num, buffer_size, data_type, get_counts_arg, buckets, sum(dpu_max_aggs)], dtype=np.uint32), 8)
		dpu_args = [memoryview(dpu_arg) for _ in range(dpu_num)]
		dpus.copy(DPU_HEAP_PTR, dpu_args, size=len(dpu_arg)*4)
		dpus.sync()
		dpus.exec()
		dpus.sync()

		dpu_agg_buffers = np.zeros(dpu_num * sum(dpu_max_aggs) * buckets * buckets, dtype=vals.dtype)

		print("Init time: " + str(time.time() - init_start_time))

		exec_start_time = time.time()
		for it in range(len(agg_dist)):
			print("ITERATION: " + str(it))
			dpu_exec_aggs(agg_dist[it], rows, vals, row_num, col_num, buffers, data_type, buckets, dpu_num, dpus)
			print()
		exec_end_time = time.time()
		print("TOTAL Exec time: " + str(exec_end_time - exec_start_time))

		dpu_agg_buffers_reshape = dpu_agg_buffers.reshape([dpu_num, -1])
		dpu_agg_buffers_m = [memoryview(dpu_agg_buffers_reshape[i]) for i in range(dpu_num)]
		dpus.copy(dpu_agg_buffers_m, DPU_HEAP_PTR, offset=buffer_size)
		xfer_end_time = time.time()
		dpus.sync()
		print("DPU->CPU Transfer time: " + str(xfer_end_time - exec_end_time))

	ret_agg_info = []
	dpu_agg_buffers_reshape = dpu_agg_buffers.reshape([dpu_rows, buckets, sum(dpu_max_aggs), -1])
	ret_agg_curr = 0
	curr = 0

	for it in range(len(agg_dist)):
		dpu_agg_buffers_curr_it = dpu_agg_buffers_reshape[:,:,curr:curr+dpu_max_aggs[it],:]
		dpu_row_curr = 0
		hp = agg_dist[it][0]   # Horizontal partitioning
		for aggs in agg_dist[it][1]:
			curr_dpu_aggs = dpu_agg_buffers_curr_it[dpu_row_curr:dpu_row_curr + hp][:,:,:len(aggs),:]
			curr_agg_size = curr_dpu_aggs[0].size
			ret_out = ret_aggs[ret_agg_curr:ret_agg_curr + curr_agg_size].reshape(curr_dpu_aggs.shape[1:])
			np.sum(curr_dpu_aggs, axis=0, out=ret_out)

			dpu_row_curr += hp
			ret_out = np.transpose(ret_out, (1, 0, 2))
			ret_aggs[ret_agg_curr:ret_agg_curr + curr_agg_size] = ret_out.ravel()
			
			ret_agg_curr += curr_agg_size
			ret_agg_info += aggs
		curr += dpu_max_aggs[it]
	reshape_end_time = time.time()
	print("Reshape time: " + str(reshape_end_time - xfer_end_time))
	return ret_agg_info
