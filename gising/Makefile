.PHONY: all clean test lib

all: lib test

clean:
	rm -f *.pyc _cising.so
	cd cising && make clean

lib:
	cd cising && make _cising.so
	cp cising/_cising.so .

test:
	cd cising && make test1
	cd cising && ./test1

