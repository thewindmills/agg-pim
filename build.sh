#!/bin/sh
g++ -shared -fPIC -O3 -o bucketize_data.so bucketize_data.cpp
g++ -mavx -march=native -shared -pthread -fPIC -O3 -o baseline.so baseline.cpp
#g++ -mavx -march=native -shared -pthread -fPIC --std=c++11 -O3 app.cpp -o dpu_app.so `dpu-pkg-config --cflags --libs dpu`
dpu-upmem-dpurte-clang -O3 -DNR_TASKLETS=11 -I. -o task task.c
g++ -mavx -march=native -shared -pthread -fPIC -O3 -o buffer.so buffer.cpp