#include <iostream>
#include <cstring>

extern "C"
void bucketize_data(int data[], const int num, const int total_num, uint8_t* buckets, int bucket_num) {
  int msb = 0;
  int total_num_tmp = total_num;
  while (total_num_tmp >>= 1) {
    msb++;
  }
  msb++;

  // 2^20
  const int HISTO_SIZE = 1048576;

  int histogram[HISTO_SIZE];
  memset(histogram, 0, HISTO_SIZE * sizeof(int));

  // TODO: Check if num and total_num is way above HISTO_SIZE

  const int shift = msb - 20 > 0 ? msb - 20 : 0;

  for (int i = 0; i < num; i++) {
    histogram[data[i] >> shift]++;
  }

  const int num_per_bucket = (num + bucket_num - 1) / bucket_num;

  int curr_bucket_idx = 0;
  int curr_num_in_bucket = 0;
  for (int i = 0; i < HISTO_SIZE; i++) {
    curr_num_in_bucket += histogram[i];
    histogram[i] = curr_bucket_idx;
    if (curr_num_in_bucket >= num_per_bucket) {
      curr_bucket_idx++;
      curr_num_in_bucket = 0;
    }
  }

  for (int i = 0; i < num; i++) {
    buckets[i] = histogram[data[i] >> shift];
  }
}