cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Os")

add_definitions(-DAIPI_VERSION=${AIPI_VERSION})

project(aipi)

add_executable(aipi run2.cc)

if (CMAKE_BUILD_TYPE STREQUAL release)
  add_custom_command(TARGET aipi POST_BUILD COMMAND strip aipi)
endif ()

find_package(ggml CONFIG REQUIRED)
target_link_libraries(aipi PRIVATE ggml::all ggml::ggml)

find_package(llama CONFIG REQUIRED)
target_link_libraries(aipi PRIVATE llama)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(aipi PRIVATE fmt::fmt-header-only)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(aipi PRIVATE spdlog::spdlog_header_only)

target_include_directories(aipi PRIVATE include)

if (APPLE)
  target_link_libraries(aipi PRIVATE "-framework CoreFoundation")
endif ()

message("System info: ${CMAKE_SYSTEM}")
message("System name: ${CMAKE_SYSTEM_NAME}")
message("System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message("System host name: ${CMAKE_HOST_SYSTEM_NAME}")
message("CXX Standard: ${CMAKE_CXX_STANDARD}")
