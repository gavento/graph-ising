.PHONY: all clean test lib

all: lib

clean:
	cd cising && make clean
	rm -f netising/*.pyc netising/_cising.so
	rm -rf __pycache__

lib:
	cd cising && make _cising.so
	cp cising/_cising.so netising/

test: lib
	cd cising && make tests
	cd cising && make runtests
	pytest . -v

