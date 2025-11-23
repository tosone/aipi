#include "llama.h"

#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

// 使用 llama.cpp 的模型结构
struct LlamaCppModel {
  struct llama_model *model = nullptr;
  struct llama_context *ctx = nullptr;
  const struct llama_vocab *vocab = nullptr;

  int32_t n_ctx = 0;
  int32_t n_vocab = 0;
  int32_t n_embd = 0;

  // 特殊token IDs
  llama_token bos_token = -1;
  llama_token eos_token = -1;
  llama_token nl_token = -1;

  // 多轮对话支持
  std::vector<llama_token> conversation_tokens; // 完整对话历史
  int32_t current_pos = 0;                      // 当前位置
};

// 初始化模型
bool init_llama_model(LlamaCppModel &model, const char *model_path) {
  // 初始化 llama 后端
  llama_backend_init();

  // 设置模型参数
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // 纯 CPU 推理
  model_params.use_mmap = true;
  model_params.use_mlock = false;
  model_params.vocab_only = false;

  spdlog::info("Loading model from: {}", model_path);

  // 加载模型
  model.model = llama_model_load_from_file(model_path, model_params);
  if (!model.model) {
    spdlog::error("Failed to load model from {}", model_path);
    return false;
  }

  spdlog::info("Model loaded successfully");

  // 获取模型信息
  model.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.model));
  model.n_embd = llama_model_n_embd(model.model);

  spdlog::info("Model info:");
  spdlog::info("  Vocabulary size: {}", model.n_vocab);
  spdlog::info("  Embedding size: {}", model.n_embd);
  spdlog::info("  Training context: {}", llama_model_n_ctx_train(model.model));
  spdlog::info("  Number of layers: {}", llama_model_n_layer(model.model));
  spdlog::info("  Number of heads: {}", llama_model_n_head(model.model));

  // 设置上下文参数
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;   // 上下文长度
  ctx_params.n_batch = 512;  // 批处理大小
  ctx_params.n_ubatch = 512; // 物理批大小
  ctx_params.n_threads = 4;  // 线程数
  ctx_params.n_threads_batch = 4;
  ctx_params.no_perf = false;
  ctx_params.embeddings = false;

  model.n_ctx = ctx_params.n_ctx;

  spdlog::info("Creating context with {} tokens", model.n_ctx);

  // 创建上下文
  model.ctx = llama_init_from_model(model.model, ctx_params);
  if (!model.ctx) {
    spdlog::error("Failed to create context");
    llama_model_free(model.model);
    return false;
  }

  // 获取词汇表
  model.vocab = llama_model_get_vocab(model.model);

  // 获取特殊token
  model.bos_token = llama_vocab_bos(model.vocab);
  model.eos_token = llama_vocab_eos(model.vocab);
  model.nl_token = llama_vocab_nl(model.vocab);

  spdlog::info("Special tokens:");
  spdlog::info("  BOS token: {}", model.bos_token);
  spdlog::info("  EOS token: {}", model.eos_token);
  spdlog::info("  NL token: {}", model.nl_token);

  spdlog::info("Model initialization completed successfully");
  return true;
}

// 编码文本为token序列
std::vector<llama_token> tokenize_text(const LlamaCppModel &model, const std::string &text, bool add_special = true) {
  // 计算需要的token数量
  const int n_tokens_max = text.length() + (add_special ? 8 : 0);
  std::vector<llama_token> tokens(n_tokens_max);

  // 进行tokenization
  const int n_tokens = llama_tokenize(
      model.vocab,
      text.c_str(),
      text.length(),
      tokens.data(),
      n_tokens_max,
      add_special, // add_special (BOS/EOS)
      false        // parse_special
  );

  if (n_tokens < 0) {
    spdlog::error("Tokenization failed, need {} tokens but only have space for {}", -n_tokens, n_tokens_max);
    return {};
  }

  tokens.resize(n_tokens);

  spdlog::info("Tokenized '{}' into {} tokens", text, n_tokens);

  // 打印一些token调试信息
  std::string debug_str = "Tokens: ";
  for (size_t i = 0; i < std::min((size_t)10, tokens.size()); i++) {
    debug_str += std::to_string(tokens[i]) + " ";
  }
  if (tokens.size() > 10) {
    debug_str += "...";
  }
  spdlog::debug(debug_str);

  return tokens;
}

// 解码token序列为文本
std::string detokenize_tokens(const LlamaCppModel &model, const std::vector<llama_token> &tokens) {
  if (tokens.empty()) {
    return "";
  }

  // 估算需要的缓冲区大小
  const int buffer_size = tokens.size() * 8;
  std::vector<char> buffer(buffer_size);

  // 进行detokenization
  const int n_chars = llama_detokenize(
      model.vocab,
      tokens.data(),
      tokens.size(),
      buffer.data(),
      buffer_size,
      true, // remove_special
      false // unparse_special
  );

  if (n_chars < 0) {
    spdlog::error("Detokenization failed, need {} chars but only have space for {}", -n_chars, buffer_size);
    return "";
  }

  std::string result(buffer.data(), n_chars);
  spdlog::debug("Detokenized {} tokens to: '{}'", tokens.size(), result);

  return result;
}

// 执行推理生成文本（支持多轮对话）
std::string generate_text(LlamaCppModel &model, const std::string &prompt, int max_new_tokens = 50) {
  spdlog::info("Starting text generation for prompt: '{}'", prompt);

  // 1. 对新的prompt进行tokenization
  std::vector<llama_token> prompt_tokens = tokenize_text(model, prompt, false); // 不添加特殊token
  if (prompt_tokens.empty()) {
    spdlog::error("Failed to tokenize prompt");
    return "";
  }

  // 2. 添加换行符分隔对话（如果不是第一次对话）
  if (!model.conversation_tokens.empty()) {
    if (model.nl_token != -1) {
      model.conversation_tokens.push_back(model.nl_token);
    }
  }

  // 3. 将新prompt添加到对话历史
  model.conversation_tokens.insert(model.conversation_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());

  // 4. 检查是否超出上下文长度，如果超出则截断早期对话
  if (model.conversation_tokens.size() > model.n_ctx - max_new_tokens - 10) {
    int tokens_to_remove = model.conversation_tokens.size() - (model.n_ctx - max_new_tokens - 10);
    model.conversation_tokens.erase(model.conversation_tokens.begin(), model.conversation_tokens.begin() + tokens_to_remove);
    spdlog::info("Truncated {} tokens from conversation history to fit context", tokens_to_remove);
    // 重新清理内存并重新处理整个对话历史
    llama_memory_clear(llama_get_memory(model.ctx), true);
    model.current_pos = 0;
  }

  // 5. 处理新tokens (prefill阶段)
  std::vector<llama_token> tokens_to_process;
  int start_pos = model.current_pos;

  // 确定需要处理的tokens
  if (model.current_pos < model.conversation_tokens.size()) {
    tokens_to_process.assign(
        model.conversation_tokens.begin() + model.current_pos,
        model.conversation_tokens.end());
  }

  if (!tokens_to_process.empty()) {
    spdlog::info("Processing {} new tokens...", tokens_to_process.size());

    // 创建batch来处理新tokens
    llama_batch batch = llama_batch_init(tokens_to_process.size(), 0, 1);

    // 填充batch数据
    for (size_t i = 0; i < tokens_to_process.size(); i++) {
      batch.token[i] = tokens_to_process[i];
      batch.pos[i] = start_pos + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;                                        // 直接使用已分配的内存
      batch.logits[i] = (i == tokens_to_process.size() - 1) ? 1 : 0; // 只需要最后一个token的logits
    }
    batch.n_tokens = tokens_to_process.size();

    // 执行处理
    int result = llama_decode(model.ctx, batch);
    if (result != 0) {
      spdlog::error("Failed to process tokens, error code: {}", result);
      llama_batch_free(batch);
      return "";
    }

    // 释放batch内存
    llama_batch_free(batch);

    // 更新当前位置
    model.current_pos = model.conversation_tokens.size();
  }

  spdlog::info("Starting generation...");
  
  // 开始流式输出显示
  std::cout << "助手: " << std::flush;

  // 6. 逐步生成新token (generation阶段)
  std::vector<llama_token> generated_tokens;
  
  for (int i = 0; i < max_new_tokens; i++) {
    // 等待计算完成
    llama_synchronize(model.ctx);

    // 获取logits
    const float *logits = llama_get_logits_ith(model.ctx, -1);
    if (!logits) {
      spdlog::error("Failed to get logits");
      break;
    }

    // 打印前10个logits用于调试
    if (i < 3) { // 只在前几次迭代中打印
      std::string logits_debug = "Top 10 logits: ";
      for (int k = 0; k < std::min(10, model.n_vocab); k++) {
        logits_debug += fmt::format("{}:{:.3f} ", k, logits[k]);
      }
      spdlog::debug(logits_debug);
    }

    // 改进的贪心采样 - 选择概率最高的token
    llama_token next_token = 0;
    float max_logit = -std::numeric_limits<float>::infinity(); // 使用负无穷大作为初始值
    
    for (int j = 0; j < model.n_vocab; j++) {
      if (logits[j] > max_logit) {
        max_logit = logits[j];
        next_token = j;
      }
    }
    
    spdlog::debug("Selected token: {} with logit: {:.6f}", next_token, max_logit);

    // 检查是否是结束token
    if (next_token == model.eos_token) {
      spdlog::debug("Generated EOS token, stopping generation");
      break;
    }

    // 添加到生成的tokens和对话历史
    generated_tokens.push_back(next_token);
    model.conversation_tokens.push_back(next_token);
    
    // 实时显示生成的token
    std::vector<llama_token> single_token = {next_token};
    std::string token_text = detokenize_tokens(model, single_token);
    
    // 调试信息：显示token ID和转换后的文本
    if (i < 5) { // 只在前几个中打印详细信息
      spdlog::debug("Token {}: ID={}, text='{}'", i+1, next_token, token_text);
    }
    
    std::cout << token_text << std::flush; // 立即刷新显示

    // 为下一次推理准备单个token的batch
    llama_batch single_batch = llama_batch_init(1, 0, 1);
    single_batch.token[0] = next_token;
    single_batch.pos[0] = model.current_pos + i;
    single_batch.n_seq_id[0] = 1;
    single_batch.seq_id[0][0] = 0; // 直接使用已分配的内存
    single_batch.logits[0] = 1;    // 需要logits用于下一次采样
    single_batch.n_tokens = 1;

    // 执行单token推理
    int result = llama_decode(model.ctx, single_batch);

    // 清理
    llama_batch_free(single_batch);

    if (result != 0) {
      spdlog::error("Failed to decode token at step {}, error code: {}", i, result);
      break;
    }

    // 打印生成进度（但不干扰输出显示）
    if ((i + 1) % 20 == 0) {
      spdlog::debug("Generated {} tokens so far", i + 1);
    }
  }
  
  // 生成结束后换行
  std::cout << std::endl;
  
  // 更新当前位置
  model.current_pos = model.conversation_tokens.size();

  spdlog::debug("Generation completed with {} new tokens", generated_tokens.size());
  spdlog::debug("Total conversation tokens: {}", model.conversation_tokens.size());
  
  // 返回生成的文本（用于日志记录）
  std::string response = detokenize_tokens(model, generated_tokens);
  return response;
}

// 重置对话历史
void reset_conversation(LlamaCppModel &model) {
  model.conversation_tokens.clear();
  model.current_pos = 0;
  llama_memory_clear(llama_get_memory(model.ctx), true);
  spdlog::info("Conversation history cleared");
}

// 释放模型资源
void free_llama_model(LlamaCppModel &model) {
  if (model.ctx) {
    llama_free(model.ctx);
    model.ctx = nullptr;
  }

  if (model.model) {
    llama_model_free(model.model);
    model.model = nullptr;
  }

  llama_backend_free();
}

// 主函数
int main(int argc, char **argv) {
  // 设置日志格式和级别
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%L] %v");
  spdlog::set_level(spdlog::level::debug); // 临时设置为debug级别来查看调试信息

  if (argc != 2) {
    spdlog::error("Usage: {} <model.gguf>", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];

  LlamaCppModel model = {};

  // 初始化模型
  spdlog::info("=== Initializing LLaMA model ===");
  if (!init_llama_model(model, model_path)) {
    spdlog::error("Failed to initialize model");
    return 1;
  }

  // 交互式对话循环
  spdlog::info("=== Starting interactive chat ===");
  std::cout << "\nLLaMA.cpp 模型已就绪！输入文本开始对话\n";
  std::cout << "命令:\n";
  std::cout << "  'quit' 或 'exit' - 退出程序\n";
  std::cout << "  'reset' - 清除对话历史\n";
  std::cout << "  'status' - 显示对话状态\n"
            << std::endl;

  std::string input;

  while (true) {
    std::cout << "用户: ";
    if (!std::getline(std::cin, input)) {
      break;
    }

    if (input == "quit" || input == "exit") {
      break;
    }

    if (input == "reset") {
      reset_conversation(model);
      std::cout << "对话历史已清除\n"
                << std::endl;
      continue;
    }

    if (input == "status") {
      std::cout << "对话状态:" << std::endl;
      std::cout << "  总tokens: " << model.conversation_tokens.size() << std::endl;
      std::cout << "  当前位置: " << model.current_pos << std::endl;
      std::cout << "  剩余上下文: " << (model.n_ctx - model.conversation_tokens.size()) << " tokens\n"
                << std::endl;
      continue;
    }

    if (input.empty()) {
      continue;
    }

    // 记录开始时间
    const int64_t t_start = llama_time_us();

    // 生成回复（流式显示）
    std::string response = generate_text(model, input, 30);

    // 计算耗时
    const int64_t t_end = llama_time_us();
    const double duration_ms = (t_end - t_start) / 1000.0;

    // 显示性能信息
    spdlog::info("Generation completed in {:.2f} ms", duration_ms);
    std::cout << std::endl;
  }

  // 清理资源
  spdlog::info("=== Cleaning up ===");
  free_llama_model(model);

  spdlog::info("Done!");
  return 0;
}
