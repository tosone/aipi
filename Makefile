deps        = $(shell jq --raw-output '.deps | join(" ")' deps.json | tr -d "")
export_deps = $(shell jq --raw-output '.export | join(" ")' deps.json | tr -d "")
version     = $(shell cat VERSION | tr -d "")

BUILD_OUTPUT = bin

.PHONY: build
build: debug

.PHONY: bin
bin:
	if [ ! -d $(BUILD_OUTPUT) ]; then mkdir -p $(BUILD_OUTPUT); fi

.PHONY: debug
debug: bin
	g++ -g -O0 -std=c++17 -DDEBUG -DSPIDER_VERSION="$(version)" \
		-I./pkgs/installed/arm64-osx/include \
		-L./pkgs/installed/arm64-osx/lib \
		run.cc -o $(BUILD_OUTPUT)/aipi \
		-lggml -lggml-base -lggml-cpu -lggml-opencl \
		-lOpenCL -framework Accelerate -framework OpenCL

.PHONY: release
release: bin
	g++ -O3 -DNDEBUG -std=c++17 -DSPIDER_VERSION="$(version)" \
		-I./pkgs/installed/arm64-osx/include \
		-L./pkgs/installed/arm64-osx/lib \
		run.cc -o $(BUILD_OUTPUT)/aipi \
		-lggml -lggml-base -lggml-cpu -lggml-opencl \
		-lOpenCL -framework Accelerate -framework OpenCL

.PHONY: deps
deps:
	vcpkg install $(deps) && vcpkg export --raw --output=pkgs --output-dir=. $(export_deps)

.PHONY: image
image:
	docker build --build-arg SPIDER_VERSION=$(version) -t ghcr.io/spider-all/spider-cplusplus:$(version)-$(shell date +'%Y%m%d%H%M') .

.PHONY: changelog
changelog:
	git-chglog --next-tag $(version) -o CHANGELOG.md

.PHONY: clean
clean:
	$(RM) -r $(BUILD_OUTPUT)
