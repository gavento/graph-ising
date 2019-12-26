.PHONY: all clean

CFLAGS=-Wall -O3 -g3 -std=c99

all: _cising.so test1

clean:
	rm -f _cising.so test1 *.o

_cising.so: cising.c utils.c clustering.c
	gcc -shared -o $@ -fpic $(CFLAGS) $^ -lm 

test1: cising.c utils.c clustering.c test1.c
	gcc -o $@ $(CFLAGS) $^ -lm 

