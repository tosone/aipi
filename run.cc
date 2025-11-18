#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// 简单的模型结构
struct simple_gguf_model {
  // GGUF 相关
  struct gguf_context *ctx_gguf = nullptr;
  struct ggml_context *ctx_weights = nullptr;

  // 计算后端
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer_weights = nullptr;

  // 模型张量映射
  std::map<std::string, struct ggml_tensor *> tensors;

  // 模型参数 (示例用)
  struct ggml_tensor *input = nullptr;
  struct ggml_tensor *weight = nullptr;
  struct ggml_tensor *bias = nullptr;
  struct ggml_tensor *output = nullptr;
};

static void log_callback_impl(ggml_log_level level, const char *text, void *user_data) {
  (void)user_data;
  std::string msg(text);
  if (!msg.empty() && msg.back() == '\n') {
    msg.pop_back();
  }

  switch (level) {
  case GGML_LOG_LEVEL_ERROR:
    spdlog::error("[GGML] {}", msg);
    break;
  case GGML_LOG_LEVEL_WARN:
    spdlog::warn("[GGML] {}", msg);
    break;
  case GGML_LOG_LEVEL_INFO:
    spdlog::info("[GGML] {}", msg);
    break;
  default:
    spdlog::debug("[GGML] {}", msg);
    break;
  }
}

// 从 GGUF 文件加载模型权重到后端缓冲区
bool load_weights_from_gguf(const char *fname, struct gguf_context *ctx_gguf, struct ggml_context *ctx_weights) {
  FILE *f = fopen(fname, "rb");
  if (!f) {
    spdlog::error("Could not open file {}", fname);
    return false;
  }

  const size_t buf_size = 4 << 20;
  void *buf = malloc(buf_size);
  if (!buf) {
    fclose(f);
    return false;
  }

  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  spdlog::info("Loading {} tensors from GGUF file...", n_tensors);

  for (int i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_weights, name);

    if (!tensor) {
      spdlog::warn("Tensor {} not found in context", name);
      continue;
    }

    // 计算张量在文件中的偏移量
    const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

    if (fseek(f, offs, SEEK_SET) != 0) {
      spdlog::error("Failed to seek to tensor {}", name);
      free(buf);
      fclose(f);
      return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    spdlog::info("  Loading tensor: {} [{} bytes]", name, nbytes);

    // 分块读取大张量
    for (size_t pos = 0; pos < nbytes; pos += buf_size) {
      const size_t chunk_size = (buf_size < nbytes - pos) ? buf_size : (nbytes - pos);

      if (fread(buf, 1, chunk_size, f) != chunk_size) {
        spdlog::error("Failed to read tensor data for {}", name);
        free(buf);
        fclose(f);
        return false;
      }

      // 将数据设置到后端张量
      ggml_backend_tensor_set(tensor, buf, pos, chunk_size);
    }
  }

  free(buf);
  fclose(f);
  spdlog::info("Successfully loaded all tensors");
  return true;
}

// 初始化模型
bool init_model(simple_gguf_model &model, const char *model_path) {
  // 设置日志回调
  ggml_log_set(log_callback_impl, nullptr);

  // 加载所有后端
  ggml_backend_load_all();

  // 初始化最佳后端 (GPU 优先，然后是 CPU)
  model.backend = ggml_backend_init_best();
  if (!model.backend) {
    spdlog::error("Failed to initialize backend");
    return false;
  }

  spdlog::info("Using backend: {}", ggml_backend_name(model.backend));

  // 从 GGUF 文件加载模型
  struct gguf_init_params gguf_params = {
      /*.no_alloc =*/true, // 我们手动分配内存
      /*.ctx      =*/&model.ctx_weights,
  };

  model.ctx_gguf = gguf_init_from_file(model_path, gguf_params);
  if (!model.ctx_gguf) {
    spdlog::error("Failed to load GGUF file {}", model_path);
    return false;
  }

  spdlog::info("GGUF file loaded successfully");
  spdlog::info("  Version: {}", gguf_get_version(model.ctx_gguf));
  spdlog::info("  Tensors: {}", gguf_get_n_tensors(model.ctx_gguf));
  spdlog::info("  KV pairs: {}", gguf_get_n_kv(model.ctx_gguf));

  // 打印一些元数据信息
  for (int64_t i = 0; i < gguf_get_n_kv(model.ctx_gguf); i++) {
    const char *key = gguf_get_key(model.ctx_gguf, i);
    const enum gguf_type type = gguf_get_kv_type(model.ctx_gguf, i);

    spdlog::info("  KV[{}]: {} (type: {})", i, key, gguf_type_name(type));

    // 打印一些字符串类型的值作为示例
    if (type == GGUF_TYPE_STRING) {
      const char *value = gguf_get_val_str(model.ctx_gguf, i);
      spdlog::info("    Value: {}", value);
    }
  }

  // 为权重分配后端缓冲区
  model.buffer_weights = ggml_backend_alloc_ctx_tensors(model.ctx_weights, model.backend);
  if (!model.buffer_weights) {
    spdlog::error("Failed to allocate weights buffer");
    return false;
  }

  // 加载权重数据
  if (!load_weights_from_gguf(model_path, model.ctx_gguf, model.ctx_weights)) {
    spdlog::error("Failed to load weights");
    return false;
  }

  // 构建张量映射
  const int n_tensors = gguf_get_n_tensors(model.ctx_gguf);
  for (int i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(model.ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(model.ctx_weights, name);
    if (tensor) {
      model.tensors[std::string(name)] = tensor;
    }
  }

  spdlog::info("Model initialized successfully with {} tensors", (int)model.tensors.size());
  return true;
}

// 构建推理计算图
struct ggml_cgraph *build_inference_graph(simple_gguf_model &model,
                                          struct ggml_context *ctx_compute,
                                          const float *input_data, int input_size) {

  struct ggml_cgraph *graph = ggml_new_graph(ctx_compute);

  // 创建输入张量
  model.input = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, input_size);
  ggml_set_name(model.input, "input");
  ggml_set_input(model.input);

  // 这里是一个简单的线性层示例: output = input * weight + bias
  // 实际使用时需要根据您的模型结构来构建计算图

  // 假设我们有一个权重张量和偏置张量
  if (model.tensors.find("weight") != model.tensors.end()) {
    model.weight = model.tensors["weight"];
  }
  if (model.tensors.find("bias") != model.tensors.end()) {
    model.bias = model.tensors["bias"];
  }

  if (model.weight && model.bias) {
    // 矩阵乘法: input * weight
    struct ggml_tensor *mul_result = ggml_mul_mat(ctx_compute, model.weight, model.input);

    // 加偏置: result + bias
    model.output = ggml_add(ctx_compute, mul_result, model.bias);
    ggml_set_name(model.output, "output");
    ggml_set_output(model.output);

    // 构建前向传播图
    ggml_build_forward_expand(graph, model.output);
  } else {
    // 如果没有找到预期的张量，创建一个简单的恒等变换
    model.output = ggml_dup(ctx_compute, model.input);
    ggml_set_name(model.output, "output");
    ggml_set_output(model.output);
    ggml_build_forward_expand(graph, model.output);

    spdlog::warn("weight/bias tensors not found, using identity transformation");
  }

  return graph;
}

// 执行推理
bool run_inference(simple_gguf_model &model, const float *input_data,
                   int input_size, float *output_data, int output_size) {

  // 创建计算上下文
  size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
  struct ggml_init_params params = {
      /*.mem_size   =*/ctx_size,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };

  struct ggml_context *ctx_compute = ggml_init(params);
  if (!ctx_compute) {
    spdlog::error("Failed to create compute context");
    return false;
  }

  // 构建计算图
  struct ggml_cgraph *graph = build_inference_graph(model, ctx_compute, input_data, input_size);

  // 创建图分配器
  ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
  if (!allocr) {
    spdlog::error("Failed to create allocator");
    ggml_free(ctx_compute);
    return false;
  }

  // 分配计算图内存
  if (!ggml_gallocr_alloc_graph(allocr, graph)) {
    spdlog::error("Failed to allocate graph");
    ggml_gallocr_free(allocr);
    ggml_free(ctx_compute);
    return false;
  }

  // 设置输入数据
  ggml_backend_tensor_set(model.input, input_data, 0, input_size * sizeof(float));

  // 执行推理
  spdlog::info("Running inference...");
  if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS) {
    spdlog::error("Failed to compute graph");
    ggml_gallocr_free(allocr);
    ggml_free(ctx_compute);
    return false;
  }

  // 获取输出数据
  const size_t output_bytes = ggml_nelements(model.output) * sizeof(float);
  const size_t copy_size = std::min(output_bytes, (size_t)(output_size * sizeof(float)));
  ggml_backend_tensor_get(model.output, output_data, 0, copy_size);

  spdlog::info("Inference completed successfully");

  // 清理
  ggml_gallocr_free(allocr);
  ggml_free(ctx_compute);

  return true;
}

// 释放模型资源
void free_model(simple_gguf_model &model) {
  if (model.buffer_weights) {
    ggml_backend_buffer_free(model.buffer_weights);
    model.buffer_weights = nullptr;
  }

  if (model.backend) {
    ggml_backend_free(model.backend);
    model.backend = nullptr;
  }

  if (model.ctx_gguf) {
    gguf_free(model.ctx_gguf);
    model.ctx_gguf = nullptr;
  }

  // ctx_weights 由 gguf_free 自动释放
  model.ctx_weights = nullptr;
}

// 主函数示例
int main(int argc, char **argv) {
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%L] %v");
  spdlog::set_level(spdlog::level::debug);

  if (argc != 2) {
    spdlog::error("Usage: {} <model.gguf>", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];

  // 初始化时间
  ggml_time_init();

  simple_gguf_model model = {};

  // 加载模型
  spdlog::info("Loading model from {}...", model_path);
  if (!init_model(model, model_path)) {
    spdlog::error("Failed to initialize model");
    return 1;
  }

  // 准备示例输入数据 (这里使用随机数据作为示例)
  const int input_size = 128; // 根据您的模型调整
  const int output_size = 64; // 根据您的模型调整

  std::vector<float> input_data(input_size);
  std::vector<float> output_data(output_size);

  // 填充随机输入数据
  for (int i = 0; i < input_size; i++) {
    input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
  }

  spdlog::info("Input data: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}{}",
               input_data[0], input_data[1], input_data[2], input_data[3], input_data[4],
               input_data[5], input_data[6], input_data[7], input_data[8], input_data[9],
               input_size > 10 ? "..." : "");

  const int64_t t_start = ggml_time_us();

  if (!run_inference(model, input_data.data(), input_size,
                     output_data.data(), output_size)) {
    spdlog::error("Inference failed");
    free_model(model);
    return 1;
  }

  const int64_t t_end = ggml_time_us();

  spdlog::info("Inference time: {:.2f} ms", (t_end - t_start) / 1000.0);

  // 打印输出结果
  spdlog::info("Output data: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}{}",
               output_data[0], output_data[1], output_data[2], output_data[3], output_data[4],
               output_data[5], output_data[6], output_data[7], output_data[8], output_data[9],
               output_size > 10 ? "..." : "");

  // 清理资源
  free_model(model);

  spdlog::info("Done!");
  return 0;
}
