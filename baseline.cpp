#include <iostream>
#include <cstring>
#include <thread>
#include <vector>
#include <tuple>

#include <immintrin.h>

// TODO: Template functions with different types for rows and values.

template <typename T>
void calculate_aggregates(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS) {
  const size_t agg_group_size = BUCKETS * BUCKETS * BUCKETS;
  for (size_t row = 0; row < ROW_NUM; row++) {
    size_t agg_group_idx = 0;
    uint8_t* curr_row = &rows[COL_NUM * row];
    const int val = vals[row];
    for (size_t i = 0; i < COL_NUM - 2; i++) {
      size_t i_idx = curr_row[i];
      for (size_t j = i + 1; j < COL_NUM - 1; j++) {
        size_t j_idx = curr_row[j];
        for (size_t k = j + 1; k < COL_NUM; k++) {
          aggs[agg_group_idx * agg_group_size + (i_idx * BUCKETS * BUCKETS) + j_idx * BUCKETS + curr_row[k]] += val;
          agg_group_idx++;
        }
      }
    }
  }
}

void calculate_aggregates_int(uint8_t* rows, int* vals, int* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS) {
  calculate_aggregates(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS);
}

void calculate_aggregates_float(uint8_t* rows, float* vals, float* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS) {
  calculate_aggregates(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS);
}

void calculate_aggregates_double(uint8_t* rows, double* vals, double* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS) {
  calculate_aggregates(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS);
}

extern "C"
void baseline(uint8_t* rows, void* vals, void* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int data_type) {
  if (data_type == 0) {
    calculate_aggregates_int(rows, (int*)vals, (int*)aggs, ROW_NUM, COL_NUM, BUCKETS);
  } else if (data_type == 1) {
    calculate_aggregates_float(rows, (float*)vals, (float*)aggs, ROW_NUM, COL_NUM, BUCKETS);
  } else if (data_type == 2) {
    calculate_aggregates_double(rows, (double*)vals, (double*)aggs, ROW_NUM, COL_NUM, BUCKETS);
  }
}

template <typename T>
void calculate_aggregate(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, std::tuple<size_t, size_t, size_t>* combs, const size_t num_combs) {
  for (size_t i = 0; i < num_combs; i++) {
    size_t COL0 = std::get<0>(combs[i]);
    size_t COL1 = std::get<1>(combs[i]);
    size_t COL2 = std::get<2>(combs[i]);
    T* curr_agg = &aggs[BUCKETS * BUCKETS * BUCKETS * i];
    for (size_t row = 0; row < ROW_NUM; row++) {
      const uint8_t* curr_row = &rows[COL_NUM * row];
      curr_agg[curr_row[COL0] * BUCKETS * BUCKETS + curr_row[COL1] * BUCKETS + curr_row[COL2]] += vals[row];
    }
  }
}

template <typename T>
void calculate_aggregates_threaded(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS) {
  std::vector<std::thread> thread_vec;
  const size_t agg_group_size = BUCKETS * BUCKETS * BUCKETS;

  std::vector<std::tuple<size_t, size_t, size_t>> combs;
  for (size_t i = 0; i < COL_NUM - 2; i++) {
    for (size_t j = i + 1; j < COL_NUM - 1; j++) {
      for (size_t k = j + 1; k < COL_NUM; k++) {
        combs.push_back({i, j, k});
      }
    }
  }

  size_t combs_per_thread = (combs.size() + NUM_THREADS  - 1) / NUM_THREADS;
  size_t processed_combs = 0;
  for (size_t i = 0; i < NUM_THREADS; i++) {
    if (processed_combs < combs.size()) {
      size_t combs_to_process = combs.size() - processed_combs < combs_per_thread ? combs.size() - processed_combs : combs_per_thread;
      thread_vec.emplace_back(calculate_aggregate<T>, rows, vals, &aggs[agg_group_size * processed_combs], ROW_NUM, COL_NUM, BUCKETS, &combs[processed_combs], combs_to_process);
      processed_combs += combs_to_process;
    }
  }

  for (auto& t : thread_vec) {
    t.join();
  }
}

void calculate_aggregates_threaded_int(uint8_t* rows, int* vals, int* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS) {
  calculate_aggregates_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
}

void calculate_aggregates_threaded_float(uint8_t* rows, float* vals, float* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS) {
  calculate_aggregates_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
}

void calculate_aggregates_threaded_double(uint8_t* rows, double* vals, double* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS) {
  calculate_aggregates_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
}

extern "C"
void baseline_threaded(uint8_t* rows, void* vals, void* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int data_type) {
  if (data_type == 0) {
    calculate_aggregates_threaded_int(rows, (int*)vals, (int*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
  } else if (data_type == 1) {
    calculate_aggregates_threaded_float(rows, (float*)vals, (float*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
  } else if (data_type == 2) {
    calculate_aggregates_threaded_double(rows, (double*)vals, (double*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS);
  }
}

template <typename T>
void calculate_aggregate_divide(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, std::tuple<size_t, size_t, size_t>* combs, const size_t num_combs, const size_t divide) {
  for (size_t i = 0; i < num_combs; i++) {
    size_t COL0 = std::get<0>(combs[i]);
    size_t COL1 = std::get<1>(combs[i]);
    size_t COL2 = std::get<2>(combs[i]);
    T* curr_agg = &aggs[BUCKETS * BUCKETS * BUCKETS * i];

    uint8_t* rows0 = &rows[ROW_NUM * COL0];
    uint8_t* rows1 = &rows[ROW_NUM * COL1];
    uint8_t* rows2 = &rows[ROW_NUM * COL2];

    const uint8_t divide_size = (BUCKETS + divide - 1) / divide;

    for (uint8_t d = 0; d < BUCKETS; d += divide_size) {
      const uint8_t divide_start = d;
      const uint8_t divide_end = d + divide_size < BUCKETS ? d + divide_size : BUCKETS;
      for (size_t row = 0; row < ROW_NUM; row++) {
        if (rows0[row] >= divide_start && rows0[row] < divide_end) {
          curr_agg[rows0[row] * BUCKETS * BUCKETS + rows1[row] * BUCKETS + rows2[row]] += vals[row];
        }
      }
    }
  }
}

template <typename T>
void calculate_aggregates_divide_threaded(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int divide) {
  std::vector<std::thread> thread_vec;
  const size_t agg_group_size = BUCKETS * BUCKETS * BUCKETS;

  std::vector<std::tuple<size_t, size_t, size_t>> combs;
  for (size_t i = 0; i < COL_NUM - 2; i++) {
    for (size_t j = i + 1; j < COL_NUM - 1; j++) {
      for (size_t k = j + 1; k < COL_NUM; k++) {
        combs.push_back({i, j, k});
      }
    }
  }

  size_t combs_per_thread = (combs.size() + NUM_THREADS  - 1) / NUM_THREADS;
  size_t processed_combs = 0;
  for (size_t i = 0; i < NUM_THREADS; i++) {
    if (processed_combs < combs.size()) {
      size_t combs_to_process = combs.size() - processed_combs < combs_per_thread ? combs.size() - processed_combs : combs_per_thread;
      thread_vec.emplace_back(calculate_aggregate_divide<T>, rows, vals, &aggs[agg_group_size * processed_combs], ROW_NUM, COL_NUM, BUCKETS, &combs[processed_combs], combs_to_process, divide);
      processed_combs += combs_to_process;
    }
  }

  for (auto& t : thread_vec) {
    t.join();
  }
}

void calculate_aggregates_divide_threaded_int(uint8_t* rows, int* vals, int* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int divide) {
  calculate_aggregates_divide_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
}

void calculate_aggregates_divide_threaded_float(uint8_t* rows, float* vals, float* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int divide) {
  calculate_aggregates_divide_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
}

void calculate_aggregates_divide_threaded_double(uint8_t* rows, double* vals, double* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int divide) {
  calculate_aggregates_divide_threaded(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
}

extern "C"
void baseline_divide_threaded(uint8_t* rows, void* vals, void* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int divide, const int data_type) {
  if (data_type == 0) {
    calculate_aggregates_divide_threaded_int(rows, (int*)vals, (int*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
  } else if (data_type == 1) {
    calculate_aggregates_divide_threaded_float(rows, (float*)vals, (float*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
  } else if (data_type == 2) {
    calculate_aggregates_divide_threaded_double(rows, (double*)vals, (double*)aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, divide);
  }
}