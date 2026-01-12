# Makefile for C examples

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -I../include
LDFLAGS = -L../build -lrpl -lm -fopenmp

# Targets
all: c_api_demo simple_nn

c_api_demo: c_api_demo.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

simple_nn: simple_nn.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run_demo: c_api_demo
	LD_LIBRARY_PATH=../build ./$<

run_nn: simple_nn
	LD_LIBRARY_PATH=../build ./$<

clean:
	rm -f c_api_demo simple_nn

.PHONY: all run_demo run_nn clean
