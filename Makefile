.PHONY: all clean

CFLAGS=-Wall -O3 -g3 -std=c99

all: cising.so isingtest
clean:
	rm -f *.pyc cising.so isingtest

cising.so: cising.c
	gcc -shared -o $@ -fpic $(CFLAGS) $< -lm 

isingtest: cising.c
	gcc -o $@ $(CFLAGS) -Dmain_test=main $< -lm 

