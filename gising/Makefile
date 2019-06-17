.PHONY: all clean

CFLAGS=-Wall -O3 -g3 -std=c99

all: _cising.so isingtest

clean:
	rm -f *.pyc _cising.so isingtest

_cising.so: _cising.c
	gcc -shared -o $@ -fpic $(CFLAGS) $< -lm 

isingtest: _cising.c
	gcc -o $@ $(CFLAGS) -Dmain_test=main $< -lm 

