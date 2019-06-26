.PHONY: all clean

all:
	cd gising; make all

clean:
	cd gising; make clean
	rm -rf .pytest_cache

test: clean all
	./gising/isingtest
	pytest

