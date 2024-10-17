#include <iostream>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

template <typename T>
void fill_buffer(uint8_t* rows, T* vals, const int col_num, const int buckets, 
									const int target_col, unsigned* buffer_progress, unsigned* buffer_ends, uint8_t** buffers,
									const int buffer_size, const int dpu_row_idx) {
	const int dpu_offset = dpu_row_idx * buckets;
	const int start = buffer_progress[dpu_row_idx];
	const int end = buffer_ends[dpu_row_idx];
	for (int i = 0; i < buckets; i++) {
		// Set cmd to 2
		((unsigned*)buffers[dpu_offset + i])[0] = 2;
		// Set number of rows to 0
		((unsigned*)buffers[dpu_offset + i])[1] = 0;
	}
	const int tuple_size = col_num + sizeof(T);
	const int tuples_per_buffer = (buffer_size - 8) / tuple_size;
	int curr = start;
	while (curr < end) {
		const int target_buffer_idx = rows[col_num * curr + target_col];
		uint8_t* target_buffer = buffers[dpu_offset + target_buffer_idx];
		const int curr_tuples = ((unsigned*)target_buffer)[1];
		memcpy(&target_buffer[8 + curr_tuples * tuple_size], &rows[curr * col_num], col_num);
		memcpy(&target_buffer[8 + curr_tuples * tuple_size + col_num], &vals[curr], sizeof(T));
		((unsigned*)target_buffer)[1]++;
		curr++;
		if (((unsigned*)target_buffer)[1] == tuples_per_buffer) {
			break;
		}
	}
	buffer_progress[dpu_row_idx] = curr;
}

template <typename T>
void fill_buffers(uint8_t* rows, T* vals, const int col_num, const int buckets, 
									uint8_t** buffers, const int buffer_size,
									unsigned* agg_cols, const int agg_col_num, const int partitions, unsigned* buffer_progress, unsigned* buffer_ends) {
	
	std::vector<std::thread> thread_vec;
	for (unsigned i = 0; i < agg_col_num; i++) {
		for (unsigned j = 0; j < partitions; j++) {
			const int dpu_row_idx = i * partitions + j;
			thread_vec.emplace_back(fill_buffer<T>, rows, vals, col_num, buckets, agg_cols[i], buffer_progress, buffer_ends, 
			buffers, buffer_size, dpu_row_idx);
		}
	}
	for (auto& t : thread_vec) {
    t.join();
  }
}

void fill_buffers_int(uint8_t* rows, int* vals, const int col_num, const int buckets, 
									uint8_t** buffers, const int buffer_size,
									unsigned* agg_cols, const int agg_col_num, const int partitions, unsigned* buffer_progress, unsigned* buffer_ends) {
	fill_buffers(rows, vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
}

void fill_buffers_float(uint8_t* rows, float* vals, const int col_num, const int buckets, 
									uint8_t** buffers, const int buffer_size,
									unsigned* agg_cols, const int agg_col_num, const int partitions, unsigned* buffer_progress, unsigned* buffer_ends) {
	fill_buffers(rows, vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
}

void fill_buffers_double(uint8_t* rows, double* vals, const int col_num, const int buckets, 
									uint8_t** buffers, const int buffer_size,
									unsigned* agg_cols, const int agg_col_num, const int partitions, unsigned* buffer_progress, unsigned* buffer_ends) {
	fill_buffers(rows, vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
}

extern "C"
void fill_buffers_c(uint8_t* rows, void* vals, const int col_num, const int buckets, 
									uint8_t** buffers, const int buffer_size,
									unsigned* agg_cols, const int agg_col_num, const int partitions, unsigned* buffer_progress, unsigned* buffer_ends, const int data_type) {
	if (data_type == 0) {
		fill_buffers_int(rows, (int*)vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
	} else if (data_type == 1) {
		fill_buffers_float(rows, (float*)vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
	} else if (data_type == 2) {
		fill_buffers_double(rows, (double*)vals, col_num, buckets, buffers, buffer_size, agg_cols, agg_col_num, partitions, buffer_progress, buffer_ends);
	}
}