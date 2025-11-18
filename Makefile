deps        = $(shell jq --raw-output '.deps | join(" ")' deps.json | tr -d "")
export_deps = $(shell jq --raw-output '.export | join(" ")' deps.json | tr -d "")
version     = $(shell cat VERSION | tr -d "")

BUILD_OUTPUT = bin

.PHONY: build
build: debug

.PHONY: bin
bin:
	if [ ! -d $(BUILD_OUTPUT) ]; then mkdir -p $(BUILD_OUTPUT); fi

.PHONY: release debug
release debug: bin
	cd $(BUILD_OUTPUT) && \
	cmake -DCMAKE_BUILD_TYPE=$@ -DSPIDER_VERSION=$(version) .. && \
	cmake --build . -j 8

.PHONY: deps
deps:
	vcpkg install $(deps) && vcpkg export --raw --output=pkgs --output-dir=. $(export_deps)

.PHONY: image
image:
	@docker buildx bake --file ./build/docker-bake.hcl --progress plain --set "*.args.USE_MIRROR=false" --provenance false --sbom false aipi-dev

.PHONY: changelog
changelog:
	git-chglog --next-tag $(version) -o CHANGELOG.md

.PHONY: clean
clean:
	$(RM) -r $(BUILD_OUTPUT)
