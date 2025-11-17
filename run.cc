#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

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

// 日志回调函数
static void log_callback_impl(ggml_log_level level, const char *text, void *user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

// 从 GGUF 文件加载模型权重到后端缓冲区
bool load_weights_from_gguf(const char *fname, struct gguf_context *ctx_gguf,
                            struct ggml_context *ctx_weights) {
  FILE *f = fopen(fname, "rb");
  if (!f) {
    fprintf(stderr, "Error: could not open file %s\n", fname);
    return false;
  }

  // 4MB 缓冲区用于批量读取
  const size_t buf_size = 4 * 1024 * 1024;
  void *buf = malloc(buf_size);
  if (!buf) {
    fclose(f);
    return false;
  }

  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  printf("Loading %d tensors from GGUF file...\n", n_tensors);

  for (int i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_weights, name);

    if (!tensor) {
      printf("Warning: tensor %s not found in context\n", name);
      continue;
    }

    // 计算张量在文件中的偏移量
    const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

    if (fseek(f, offs, SEEK_SET) != 0) {
      fprintf(stderr, "Error: failed to seek to tensor %s\n", name);
      free(buf);
      fclose(f);
      return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    printf("  Loading tensor: %s [%zu bytes]\n", name, nbytes);

    // 分块读取大张量
    for (size_t pos = 0; pos < nbytes; pos += buf_size) {
      const size_t chunk_size = (buf_size < nbytes - pos) ? buf_size : (nbytes - pos);

      if (fread(buf, 1, chunk_size, f) != chunk_size) {
        fprintf(stderr, "Error: failed to read tensor data for %s\n", name);
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
  printf("Successfully loaded all tensors\n");
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
    fprintf(stderr, "Error: failed to initialize backend\n");
    return false;
  }

  printf("Using backend: %s\n", ggml_backend_name(model.backend));

  // 从 GGUF 文件加载模型
  struct gguf_init_params gguf_params = {
      /*.no_alloc =*/true, // 我们手动分配内存
      /*.ctx      =*/&model.ctx_weights,
  };

  model.ctx_gguf = gguf_init_from_file(model_path, gguf_params);
  if (!model.ctx_gguf) {
    fprintf(stderr, "Error: failed to load GGUF file %s\n", model_path);
    return false;
  }

  printf("GGUF file loaded successfully\n");
  printf("  Version: %u\n", gguf_get_version(model.ctx_gguf));
  printf("  Tensors: %lld\n", gguf_get_n_tensors(model.ctx_gguf));
  printf("  KV pairs: %lld\n", gguf_get_n_kv(model.ctx_gguf));

  // 打印一些元数据信息
  for (int64_t i = 0; i < gguf_get_n_kv(model.ctx_gguf); i++) {
    const char *key = gguf_get_key(model.ctx_gguf, i);
    const enum gguf_type type = gguf_get_kv_type(model.ctx_gguf, i);

    printf("  KV[%lld]: %s (type: %s)\n", i, key, gguf_type_name(type));

    // 打印一些字符串类型的值作为示例
    if (type == GGUF_TYPE_STRING) {
      const char *value = gguf_get_val_str(model.ctx_gguf, i);
      printf("    Value: %s\n", value);
    }
  }

  // 为权重分配后端缓冲区
  model.buffer_weights = ggml_backend_alloc_ctx_tensors(model.ctx_weights, model.backend);
  if (!model.buffer_weights) {
    fprintf(stderr, "Error: failed to allocate weights buffer\n");
    return false;
  }

  // 加载权重数据
  if (!load_weights_from_gguf(model_path, model.ctx_gguf, model.ctx_weights)) {
    fprintf(stderr, "Error: failed to load weights\n");
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

  printf("Model initialized successfully with %d tensors\n", (int)model.tensors.size());
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

    printf("Warning: weight/bias tensors not found, using identity transformation\n");
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
    fprintf(stderr, "Error: failed to create compute context\n");
    return false;
  }

  // 构建计算图
  struct ggml_cgraph *graph = build_inference_graph(model, ctx_compute, input_data, input_size);

  // 创建图分配器
  ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
  if (!allocr) {
    fprintf(stderr, "Error: failed to create allocator\n");
    ggml_free(ctx_compute);
    return false;
  }

  // 分配计算图内存
  if (!ggml_gallocr_alloc_graph(allocr, graph)) {
    fprintf(stderr, "Error: failed to allocate graph\n");
    ggml_gallocr_free(allocr);
    ggml_free(ctx_compute);
    return false;
  }

  // 设置输入数据
  ggml_backend_tensor_set(model.input, input_data, 0, input_size * sizeof(float));

  // 执行推理
  printf("Running inference...\n");
  if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "Error: failed to compute graph\n");
    ggml_gallocr_free(allocr);
    ggml_free(ctx_compute);
    return false;
  }

  // 获取输出数据
  const size_t output_bytes = ggml_nelements(model.output) * sizeof(float);
  const size_t copy_size = std::min(output_bytes, (size_t)(output_size * sizeof(float)));
  ggml_backend_tensor_get(model.output, output_data, 0, copy_size);

  printf("Inference completed successfully\n");

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
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];

  // 初始化时间
  ggml_time_init();

  simple_gguf_model model = {};

  // 加载模型
  printf("Loading model from %s...\n", model_path);
  if (!init_model(model, model_path)) {
    fprintf(stderr, "Failed to initialize model\n");
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

  printf("Input data: ");
  for (int i = 0; i < std::min(10, input_size); i++) {
    printf("%.3f ", input_data[i]);
  }
  printf("%s\n", input_size > 10 ? "..." : "");

  // 执行推理
  const int64_t t_start = ggml_time_us();

  if (!run_inference(model, input_data.data(), input_size,
                     output_data.data(), output_size)) {
    fprintf(stderr, "Inference failed\n");
    free_model(model);
    return 1;
  }

  const int64_t t_end = ggml_time_us();

  printf("Inference time: %.2f ms\n", (t_end - t_start) / 1000.0);

  // 打印输出结果
  printf("Output data: ");
  for (int i = 0; i < std::min(10, output_size); i++) {
    printf("%.3f ", output_data[i]);
  }
  printf("%s\n", output_size > 10 ? "..." : "");

  // 清理资源
  free_model(model);

  printf("Done!\n");
  return 0;
}
