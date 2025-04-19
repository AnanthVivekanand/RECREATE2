all: build
build:
	@cmake -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j
run: build
	@./build/create2_miner $(ARGS)
test: build
	@./build/tests
clean:
	@rm -rf build