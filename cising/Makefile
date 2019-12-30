.PHONY: all clean tests runtests

CFLAGS=-Wall -O3 -g3 -std=c99
SRCS=ising.c utils.c clustering.c game.c
TESTS=test_game test_ising

all: _cising.so $(TESTS)

clean:
	rm -f _cising.so $(TESTS) *.o

_cising.so: $(SRCS)
	gcc -shared -o $@ -fpic $(CFLAGS) $^ -lm 

tests: $(TESTS)

test_game: $(SRCS) test_game.c
	gcc -o $@ $(CFLAGS) $^ -lm 

test_ising: $(SRCS) test_ising.c
	gcc -o $@ $(CFLAGS) $^ -lm 

runtests: tests
	./test_game >/dev/null
	./test_ising >/dev/null
