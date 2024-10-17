#include <dpu>
#include <iostream>

template <typename T>
struct task_info {
  dpu::DpuSet* dpus;
  dpu::DpuSetAsync* dpus_async;
  uint8_t* rows;
	T* vals;
	T* aggs;
  int ROW_NUM;
  int COL_NUM;
  int BUCKETS;
  int max_aggs_per_dpu;
  std::vector<std::pair<int, int>> col_range;
  std::vector<std::vector<std::vector<std::pair<int, int>>>> agg_dists;
  std::vector<int> horizontal_partitioning;
  std::vector<std::vector<uint8_t>> buffer;
};

template <typename T>
void pim_process(uint8_t* rows, T* vals, T* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int NUM_DPUS) {
	task_info<T> tinfo;
	tinfo.rows = rows;
	tinfo.vals = vals;
	tinfo.aggs = aggs;
	tinfo.ROW_NUM = ROW_NUM;
	tinfo.COL_NUM = COL_NUM;
	tinfo.BUCKETS = BUCKETS;

	std::cout << "HERE" << std::endl;
}

void pim_process_int(uint8_t* rows, int* vals, int* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int NUM_DPUS) {
	pim_process(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, NUM_DPUS);
}

extern "C"
void pim_process_c(uint8_t* rows, int* vals, int* aggs, const int ROW_NUM, const int COL_NUM, const int BUCKETS, const int NUM_THREADS, const int NUM_DPUS) {
	pim_process_int(rows, vals, aggs, ROW_NUM, COL_NUM, BUCKETS, NUM_THREADS, NUM_DPUS);
}