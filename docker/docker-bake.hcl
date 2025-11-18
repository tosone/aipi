group "all" {
  targets = ["aipi-dev", "aipi-release"]
}

target "base" {
  dockerfile = "./build/Dockerfile.base"
}

target "aipi-dev" {
  dockerfile = "./build/Dockerfile.dev"
  contexts = {
    base = "target:base"
  }
}

target "aipi-release" {
  dockerfile = "./build/Dockerfile"
  contexts = {
    base = "target:base"
  }
}
