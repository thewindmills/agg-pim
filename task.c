#include <alloc.h>
#include <stdio.h>
#include <stdint.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <stdbool.h>
#include <barrier.h>
#include <mutex_pool.h>
#include <defs.h>
#include <string.h>

int round_up(int num, int round) { return (num + round - 1) & (-round); }

#define DPU_CACHE_BYTES 256
#define DPU_WRAM_INTERMEDIATE 32768

BARRIER_INIT(init_barrier, NR_TASKLETS);
MUTEX_POOL_INIT(mutex_pool, 32);

__mram_ptr uint8_t* buffer = (__mram_ptr uint8_t*)DPU_MRAM_HEAP_POINTER;

uint8_t cmd;

uint8_t col_num;
unsigned buffer_size;
uint8_t data_type;
bool get_counts;
uint8_t buckets;

uint8_t val_data_size;
uint8_t tuple_size;

unsigned agg_num;
unsigned dpu_max_aggs;
uint8_t* agg_idxs;

__mram_ptr uint8_t* agg_ptr;
uint32_t* agg_ptrs;

int tuples_in_buffer;
void update_aggs();
void update_aggs_local_cache_int(uint8_t local_cache[], unsigned tuples_to_process);
void inline update_int_agg(uint32_t update_address, int val);
void inline update_int_agg_counts(uint32_t update_address, int val);
void update_aggs_local_cache_float(uint8_t local_cache[], unsigned tuples_to_process);
void inline update_float_agg(uint32_t update_address, float val);
void update_aggs_local_cache_double(uint8_t local_cache[], unsigned tuples_to_process);
void inline update_double_agg(uint32_t update_address, double val);

uint8_t* inter_wram;
unsigned processed_tuples;
unsigned curr_tuples_to_process;
uint8_t* wram_buf;

void multithreaded_copy(const int curr_tuples_to_process, const int buffer_idx);
void update_aggs_no_sync(const int start, const int end);
void inline update_int_agg_no_sync(uint32_t update_address, int val);
void inline update_float_agg_no_sync(uint32_t update_address, float val);
void inline update_double_agg_no_sync(uint32_t update_address, double val);

int main() {
	if (me() == 0) {
		cmd = buffer[0];
	}
	barrier_wait(&init_barrier);

	// Initialize variables
	if (cmd == 0 && me() == 0) {
		__mram_ptr unsigned* buffer_uint = (__mram_ptr unsigned*)(&buffer[4]);
		col_num = buffer_uint[0];
		buffer_size = buffer_uint[1];
		data_type = buffer_uint[2];
		get_counts = buffer_uint[3] ? 1 : 0;
		buckets = buffer_uint[4];

		switch(data_type) {
			case 0:
			case 1:
				val_data_size = 4;
				break;
			case 2:
				val_data_size = 8;
				break;
		}
		tuple_size = col_num + val_data_size;
		dpu_max_aggs = 0;
		agg_ptr = buffer + buffer_size;

		const int total_agg_slices = buffer_uint[5];
		if (get_counts) {
			memset(agg_ptr, 0, round_up(total_agg_slices * buckets * buckets * (val_data_size + 4), 8));
		} else {
			memset(agg_ptr, 0, round_up(total_agg_slices * buckets * buckets * val_data_size, 8));
		}

		inter_wram = (uint8_t*)mem_alloc(DPU_WRAM_INTERMEDIATE + 16 + 2048);
		
		printf("%d %d %d %d %d\n", col_num, buffer_size, data_type, buckets, NR_TASKLETS);
	} else if (cmd == 1 && me() == 0) {
		// Move agg ptr by previous iteration number of aggs
		agg_ptr += dpu_max_aggs * buckets * buckets * val_data_size;

		dpu_max_aggs = ((__mram_ptr unsigned*)buffer)[1];
		agg_num = ((__mram_ptr unsigned*)buffer)[2];

		agg_idxs = (uint8_t*)mem_alloc(agg_num * 2 * sizeof(uint8_t));

		printf("%d\n", agg_num);

		for (int i = 0; i < agg_num; i++) {
			agg_idxs[i*2] = buffer[sizeof(int) * 3 + i * 3 + 1];
			agg_idxs[i*2 + 1] = buffer[sizeof(int) * 3 + i * 3 + 2];
		}
		/*
		for (int i = 0; i < agg_num; i++) {
			printf("%d %d, ", agg_idxs[i*2], agg_idxs[i*2 + 1]);
		}
		printf("\n");
		*/
		agg_ptrs = (uint32_t*)mem_alloc(agg_num * sizeof(uint32_t));
    for (int i = 0; i < agg_num; i++) {
      agg_ptrs[i] = ((uint32_t) agg_ptr) + i * buckets * buckets * val_data_size;
    }
	} else if (cmd == 2) {
		/*
		if (me() == 0) {
			for (int i = 0; i < 360; i++) {
				printf("%u ", buffer[i]);
				if (i % 20 == 0) {
					printf("\n");
				}
			}
			printf("\n");
		}
		*/
		update_aggs();
	}
	return 0;
}

void update_aggs() {
	if (me() == 0) {
		tuples_in_buffer = ((__mram_ptr unsigned*)buffer)[1];
		processed_tuples = 0;
	}
	barrier_wait(&init_barrier);

	/*
	__dma_aligned uint8_t local_cache[DPU_CACHE_BYTES];
	const int tuples_per_tasklet =
      (tuples_in_buffer + (NR_TASKLETS - 1)) / NR_TASKLETS;
	int total_tuples_to_process =
      tuples_in_buffer - (int)(me() * tuples_per_tasklet) > tuples_per_tasklet
          ? tuples_per_tasklet
          : tuples_in_buffer - (me() * tuples_per_tasklet);
	total_tuples_to_process = total_tuples_to_process < 0 ? 0 : total_tuples_to_process;
	const unsigned tuples_per_it = (DPU_CACHE_BYTES - 8) / tuple_size;
	const unsigned buffer_offset = sizeof(unsigned) * 2 + (me() * tuples_per_tasklet * tuple_size);

	
	for (unsigned i = 0; i < total_tuples_to_process; i += tuples_per_it) {
		unsigned tuples_to_process = total_tuples_to_process - i > tuples_per_it ? tuples_per_it : total_tuples_to_process - i;
		mram_read(&buffer[(buffer_offset + i * tuple_size) & (-7)], &local_cache, round_up(tuples_to_process * tuple_size, 8) + 8);
		unsigned data_index = (buffer_offset + i * tuple_size) % 8;
		
		if (data_type == 0) {
			update_aggs_local_cache_int(&local_cache[data_index], tuples_to_process);
		} else if (data_type == 1) {
			if (get_counts) {

			} else {
				update_aggs_local_cache_float(&local_cache[data_index], tuples_to_process);
			}
		} else if (data_type == 2) {
			if (get_counts) {

			} else {
				update_aggs_local_cache_double(&local_cache[data_index], tuples_to_process);
			}
		}
	}
		*/
	const int aggs_per_tasklet =
      (agg_num + (NR_TASKLETS - 1)) / NR_TASKLETS;
	int aggs_to_process = agg_num - (int)(me() * aggs_per_tasklet);
	if (aggs_to_process > aggs_per_tasklet) {
		aggs_to_process = aggs_per_tasklet;
	}
	aggs_to_process = aggs_to_process < 0 ? 0 : aggs_to_process;

	const int tuples_in_wram = DPU_WRAM_INTERMEDIATE / tuple_size;
	const int buffer_offset = sizeof(unsigned) * 2;

	while (processed_tuples < tuples_in_buffer) {
		/*
		if (me() == 0) {
			curr_tuples_to_process = tuples_in_buffer - processed_tuples > tuples_in_wram ? 
																		tuples_in_wram : tuples_in_buffer - processed_tuples;
			memcpy(inter_wram, &buffer[buffer_offset + processed_tuples * tuple_size], curr_tuples_to_process * tuple_size);
			processed_tuples += curr_tuples_to_process;
			wram_buf = inter_wram;
		}
		*/
		if (me() == 0) {
			curr_tuples_to_process = tuples_in_buffer - processed_tuples > tuples_in_wram ? 
																		tuples_in_wram : tuples_in_buffer - processed_tuples;
			const int buf_idx = buffer_offset + processed_tuples * tuple_size;
			wram_buf = &inter_wram[buf_idx - (buf_idx & (-7))];
		}
		barrier_wait(&init_barrier);
		multithreaded_copy(curr_tuples_to_process, buffer_offset + processed_tuples * tuple_size);
		barrier_wait(&init_barrier);
		if (me() == 0) {
			processed_tuples += curr_tuples_to_process;
		}
		barrier_wait(&init_barrier);
		update_aggs_no_sync(me() * aggs_per_tasklet, me() * aggs_per_tasklet + aggs_to_process);
		barrier_wait(&init_barrier);
	}		
}

void multithreaded_copy(const int curr_tuples_to_process, const int buffer_idx) {
	const int mem_to_read = curr_tuples_to_process * tuple_size;
	const int read_end = buffer_idx + mem_to_read;
	const int read_start = buffer_idx & (-7);
	const int num_2k_chunks = (read_end - read_start + 2047) / 2048;
	const int chunks_per_thread = (num_2k_chunks + NR_TASKLETS - 1) / NR_TASKLETS;
	int chunks_to_process = chunks_per_thread;
	if (num_2k_chunks - (int)me() * chunks_per_thread < chunks_per_thread) {
		chunks_to_process = num_2k_chunks - me() * chunks_per_thread;
		chunks_to_process = chunks_to_process < 0 ? 0 : chunks_to_process;
	}
	if (chunks_to_process > 0) {
		const int buffer_offset = read_start + me() * chunks_per_thread * 2048;
		const int wram_offset = me() * chunks_per_thread * 2048;
		for (int i = 0; i < chunks_to_process; i++) {
			mram_read(&buffer[buffer_offset + i * 2048], &inter_wram[wram_offset + i * 2048], 2048);
		}
	}
}

void update_aggs_no_sync(const int start, const int end) {
	const int start_i = start*2;
	const int end_i = end*2;
	for (unsigned t = 0; t < curr_tuples_to_process; t++) {
		uint8_t* curr_tuple = &wram_buf[t * tuple_size];

		if (data_type == 0) {
			int val;
    	memcpy(&val, &curr_tuple[col_num], sizeof(int));
			for (int i = start_i; i < end_i; i += 2) {
				uint32_t update_address = agg_ptrs[i >> 1] + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(int);
				update_int_agg_no_sync(update_address, val);
			}
		} else if (data_type == 1) {
			float val;
    	memcpy(&val, &curr_tuple[col_num], sizeof(float));
			for (int i = start_i; i < end_i; i += 2) {
				uint32_t update_address = agg_ptrs[i >> 1] + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(float);
				update_float_agg_no_sync(update_address, val);
			}
		} else if (data_type == 2) {
			double val;
			memcpy(&val, &curr_tuple[col_num], sizeof(double));
			for (int i = start_i; i < end_i; i += 2) {
				uint32_t update_address = agg_ptrs[i >> 1] + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(double);
				update_double_agg_no_sync(update_address, val);
			}
		}

	}
}

void inline update_int_agg_no_sync(uint32_t update_address, int val) {
	__dma_aligned int read_cache[2];
	const uint32_t update_address_round_down = update_address & (-7);
	const int mutex_index = update_address_round_down >> 3;
	char read_idx = update_address & (1 << 2) ? 1 : 0;
	
	mram_read((__mram_ptr void*)(update_address_round_down), read_cache,
            sizeof(read_cache));
  read_cache[read_idx] += val;
  mram_write(read_cache, (__mram_ptr void*)(update_address_round_down),
             sizeof(read_cache));
}

void inline update_float_agg_no_sync(uint32_t update_address, float val) {
	__dma_aligned float read_cache[2];
	const uint32_t update_address_round_down = update_address & (-7);
	const int mutex_index = update_address_round_down >> 3;
	char read_idx = update_address & (1 << 2) ? 1 : 0;
	
	mram_read((__mram_ptr void*)(update_address_round_down), read_cache,
            sizeof(read_cache));
  read_cache[read_idx] += val;
  mram_write(read_cache, (__mram_ptr void*)(update_address_round_down),
             sizeof(read_cache));
}

void inline update_double_agg_no_sync(uint32_t update_address, double val) {
	const int mutex_index = update_address >> 3;
	double tmp;
	mram_read((__mram_ptr void*)(update_address), &tmp,
            sizeof(double));
  tmp += val;
  mram_write(&tmp, (__mram_ptr void*)(update_address),
             sizeof(double));
}



void update_aggs_local_cache_int(uint8_t local_cache[], unsigned tuples_to_process) {
	unsigned agg_num_times_2 = agg_num * 2;
	unsigned pointer_inc = buckets * buckets * sizeof(int);
	for (unsigned t = 0; t < tuples_to_process; t++) {
		uint32_t curr_agg_addr = (uint32_t)agg_ptr;
		uint8_t* curr_tuple = &local_cache[t * tuple_size];
		int val;
    memcpy(&val, &curr_tuple[col_num], sizeof(int));
		if (!get_counts) {
			for (int i = 0; i < agg_num_times_2; i += 2) {
				uint32_t update_address = curr_agg_addr + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(int);
				update_int_agg(update_address, val);
				curr_agg_addr += pointer_inc;
			}
		} else {
			for (int i = 0; i < agg_num_times_2; i += 2) {
				uint32_t update_address = curr_agg_addr + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * (sizeof(int) + sizeof(int));
				update_int_agg_counts(update_address, val);
				curr_agg_addr += pointer_inc;
			}
		}
	}
}

void inline update_int_agg(uint32_t update_address, int val) {
	__dma_aligned int read_cache[2];
	const uint32_t update_address_round_down = update_address & (-7);
	const int mutex_index = update_address_round_down >> 3;
	mutex_pool_lock(&mutex_pool, mutex_index);
	char read_idx = update_address & (1 << 2) ? 1 : 0;
	
	mram_read((__mram_ptr void*)(update_address_round_down), read_cache,
            sizeof(read_cache));
  read_cache[read_idx] += val;
  mram_write(read_cache, (__mram_ptr void*)(update_address_round_down),
             sizeof(read_cache));
  mutex_pool_unlock(&mutex_pool, mutex_index);
}

void inline update_int_agg_counts(uint32_t update_address, int val) {
	__dma_aligned int read_cache[2];
	const int mutex_index = update_address >> 3;
	mutex_pool_lock(&mutex_pool, mutex_index);
	mram_read((__mram_ptr void*)(update_address), read_cache,
            sizeof(read_cache));
  read_cache[0] += val;
	read_cache[1] += 1;
  mram_write(read_cache, (__mram_ptr void*)(update_address),
             sizeof(read_cache));
  mutex_pool_unlock(&mutex_pool, mutex_index);
}

void update_aggs_local_cache_float(uint8_t local_cache[], unsigned tuples_to_process) {
	unsigned agg_num_times_2 = agg_num * 2;
	unsigned pointer_inc = buckets * buckets * sizeof(float);
	for (unsigned t = 0; t < tuples_to_process; t++) {
		uint32_t curr_agg_addr = (uint32_t)agg_ptr;
		uint8_t* curr_tuple = &local_cache[t * tuple_size];
		float val;
    memcpy(&val, &curr_tuple[col_num], sizeof(float));
		for (int i = 0; i < agg_num_times_2; i += 2) {
			uint32_t update_address = curr_agg_addr + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(float);
			update_float_agg(update_address, val);
			curr_agg_addr += pointer_inc;
		}
	}
}

void inline update_float_agg(uint32_t update_address, float val) {
	__dma_aligned float read_cache[2];
	const uint32_t update_address_round_down = update_address & (-7);
	const int mutex_index = update_address_round_down >> 3;
	mutex_pool_lock(&mutex_pool, mutex_index);
	char read_idx = update_address & (1 << 2) ? 1 : 0;
	
	mram_read((__mram_ptr void*)(update_address_round_down), read_cache,
            sizeof(read_cache));
  read_cache[read_idx] += val;
  mram_write(read_cache, (__mram_ptr void*)(update_address_round_down),
             sizeof(read_cache));
  mutex_pool_unlock(&mutex_pool, mutex_index);
}

void update_aggs_local_cache_double(uint8_t local_cache[], unsigned tuples_to_process) {
	unsigned agg_num_times_2 = agg_num * 2;
	unsigned pointer_inc = buckets * buckets * sizeof(double);
	for (unsigned t = 0; t < tuples_to_process; t++) {
		uint32_t curr_agg_addr = (uint32_t)agg_ptr;
		uint8_t* curr_tuple = &local_cache[t * tuple_size];
		double val;
    memcpy(&val, &curr_tuple[col_num], sizeof(double));
		for (int i = 0; i < agg_num_times_2; i += 2) {
			uint32_t update_address = curr_agg_addr + (buckets * curr_tuple[agg_idxs[i]] + curr_tuple[agg_idxs[i + 1]]) * sizeof(double);
			update_double_agg(update_address, val);
			curr_agg_addr += pointer_inc;
		}
	}
}

void inline update_double_agg(uint32_t update_address, double val) {;
	const int mutex_index = update_address >> 3;
	mutex_pool_lock(&mutex_pool, mutex_index);
	double tmp;
	mram_read((__mram_ptr void*)(update_address), &tmp,
            sizeof(double));
  tmp += val;
  mram_write(&tmp, (__mram_ptr void*)(update_address),
             sizeof(double));
  mutex_pool_unlock(&mutex_pool, mutex_index);
}
