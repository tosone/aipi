#include "llama.h"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// 对话消息结构
struct ChatMessage {
  std::string role; // "user" or "assistant"
  std::string content;
};

// 模型上下文结构
struct ModelContext {
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  const llama_vocab *vocab = nullptr;
  llama_sampler *sampler = nullptr;

  int32_t n_ctx = 0;
  int32_t n_vocab = 0;

  llama_token bos_token = -1;
  llama_token eos_token = -1;
  llama_token eot_token = -1; // end of turn token

  std::vector<llama_token> history;  // 对话历史 tokens
  std::vector<ChatMessage> messages; // 对话消息历史
  int32_t pos = 0;                   // 当前位置

  std::string chat_template; // 自定义 chat template
};

// 采样参数配置
struct SamplingConfig {
  float temperature = 0.8f;
  float top_p = 0.9f;
  int32_t top_k = 40;
  float min_p = 0.05f;
  int32_t penalty_last_n = 64;
  float penalty_repeat = 1.1f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  uint32_t seed = 12345;
};

// 初始化模型
bool load_model(ModelContext &mc, const char *path) {
  llama_backend_init();

  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;
  params.use_mmap = true;

  spdlog::info("Loading model: {}", path);
  mc.model = llama_model_load_from_file(path, params);
  if (!mc.model) {
    spdlog::error("Failed to load model");
    return false;
  }

  mc.vocab = llama_model_get_vocab(mc.model);
  mc.n_vocab = llama_vocab_n_tokens(mc.vocab);
  mc.bos_token = llama_vocab_bos(mc.vocab);
  mc.eos_token = llama_vocab_eos(mc.vocab);
  mc.eot_token = llama_vocab_eot(mc.vocab); // 获取 EOT token

  spdlog::info("Special tokens: BOS={}, EOS={}, EOT={}", mc.bos_token, mc.eos_token, mc.eot_token);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = 4;
  ctx_params.n_threads_batch = 4;

  mc.n_ctx = ctx_params.n_ctx;

  spdlog::info("Creating context (n_ctx={})", mc.n_ctx);
  mc.ctx = llama_init_from_model(mc.model, ctx_params);
  if (!mc.ctx) {
    spdlog::error("Failed to create context");
    llama_model_free(mc.model);
    return false;
  }

  spdlog::info("Model loaded successfully");
  spdlog::info("Vocabulary size: {}", mc.n_vocab);
  spdlog::info("Context size: {}", mc.n_ctx);

  return true;
}

// 创建采样器链
bool setup_sampler(ModelContext &mc, const SamplingConfig &cfg) {
  auto chain_params = llama_sampler_chain_default_params();
  chain_params.no_perf = false;

  mc.sampler = llama_sampler_chain_init(chain_params);

  // 添加采样器（顺序很重要）
  llama_sampler_chain_add(mc.sampler, llama_sampler_init_top_k(cfg.top_k));
  llama_sampler_chain_add(mc.sampler, llama_sampler_init_top_p(cfg.top_p, 1));
  llama_sampler_chain_add(mc.sampler, llama_sampler_init_min_p(cfg.min_p, 1));
  llama_sampler_chain_add(mc.sampler, llama_sampler_init_temp(cfg.temperature));

  // 添加重复惩罚
  llama_sampler_chain_add(mc.sampler,
                          llama_sampler_init_penalties(cfg.penalty_last_n, cfg.penalty_repeat, cfg.penalty_freq, cfg.penalty_present));

  // 最后添加分布采样器
  llama_sampler_chain_add(mc.sampler, llama_sampler_init_dist(cfg.seed));

  spdlog::info("Sampler configured:");
  spdlog::info("  temperature: {}", cfg.temperature);
  spdlog::info("  top_p: {}", cfg.top_p);
  spdlog::info("  top_k: {}", cfg.top_k);
  spdlog::info("  min_p: {}", cfg.min_p);
  spdlog::info("  penalty_repeat: {}", cfg.penalty_repeat);

  return true;
}

// 文本分词
std::vector<llama_token> tokenize(const ModelContext &mc, const std::string &text, bool add_special) {
  int max_tokens = text.length() + 16;
  std::vector<llama_token> tokens(max_tokens);

  int n = llama_tokenize(mc.vocab, text.c_str(), text.length(), tokens.data(), max_tokens, add_special, false);

  if (n < 0) {
    spdlog::error("Tokenization failed");
    return {};
  }

  tokens.resize(n);
  return tokens;
}

// Token 转文本
std::string detokenize(const ModelContext &mc, const std::vector<llama_token> &tokens) {
  if (tokens.empty())
    return "";

  int buf_size = tokens.size() * 8;
  std::vector<char> buffer(buf_size);

  int n = llama_detokenize(mc.vocab, tokens.data(), tokens.size(), buffer.data(), buf_size, true, false);

  if (n < 0)
    return "";

  return std::string(buffer.data(), n);
}

// 读取文件内容
std::string read_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    spdlog::error("Failed to open file: {}", path);
    return "";
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// 应用 chat template 格式化消息
std::string apply_chat_template(const ModelContext &mc, const std::vector<ChatMessage> &messages,
                                const char *custom_template = nullptr, bool add_generation_prompt = true) {
  // 构建 llama_chat_message 数组
  std::vector<llama_chat_message> chat_msgs;
  for (const auto &msg : messages) {
    chat_msgs.push_back({msg.role.c_str(), msg.content.c_str()});
  }

  // 第一次调用获取需要的缓冲区大小
  int32_t required_size = llama_chat_apply_template(custom_template, chat_msgs.data(), chat_msgs.size(), add_generation_prompt, nullptr, 0);

  if (required_size < 0) {
    spdlog::error("Failed to apply chat template (template={}, size={})",
                  custom_template ? "custom" : "builtin", required_size);
    return "";
  }

  // 分配缓冲区并实际格式化
  std::vector<char> buffer(required_size + 1);
  int32_t result = llama_chat_apply_template(custom_template, chat_msgs.data(), chat_msgs.size(), add_generation_prompt, buffer.data(), buffer.size());

  if (result < 0) {
    spdlog::error("Failed to apply chat template");
    return "";
  }

  return std::string(buffer.data(), result);
}

// 执行推理生成
std::string generate(ModelContext &mc, const std::string &user_input, int max_tokens = 50) {
  spdlog::info("User: {}", user_input);

  // 添加用户消息到历史
  mc.messages.push_back({"user", user_input});

  // 应用 chat template 生成格式化的 prompt
  const char *tmpl = mc.chat_template.empty() ? nullptr : mc.chat_template.c_str();
  std::string formatted_prompt = apply_chat_template(mc, mc.messages, tmpl, true);

  spdlog::trace("Formatted prompt: \"{}\"", formatted_prompt); // 分词
  auto tokens = tokenize(mc, formatted_prompt, false);
  if (tokens.empty()) {
    spdlog::error("Empty tokens");
    return "";
  }

  spdlog::trace("Tokenized into {} tokens", tokens.size());

  // 替换历史（使用完整的格式化 prompt）
  mc.history = tokens;

  // 检查上下文长度
  if (mc.history.size() > mc.n_ctx - max_tokens - 10) {
    spdlog::warn("Context overflow, clearing history");
    // 清空上下文，只保留最新的用户消息
    mc.messages.clear();
    mc.messages.push_back({"user", user_input});
    formatted_prompt = apply_chat_template(mc, mc.messages, tmpl, true);
    mc.history = tokenize(mc, formatted_prompt, false);
    llama_memory_clear(llama_get_memory(mc.ctx), true);
    mc.pos = 0;
  }

  // Prefill: 处理 prompt tokens (整个历史)
  llama_memory_clear(llama_get_memory(mc.ctx), true);
  mc.pos = 0;

  auto batch = llama_batch_init(mc.history.size(), 0, 1);

  for (size_t i = 0; i < mc.history.size(); i++) {
    batch.token[i] = mc.history[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == mc.history.size() - 1) ? 1 : 0; // 只计算最后一个 token 的 logits
  }
  batch.n_tokens = mc.history.size();

  if (llama_decode(mc.ctx, batch) != 0) {
    spdlog::error("Decode failed");
    llama_batch_free(batch);
    return "";
  }

  llama_batch_free(batch);
  mc.pos = mc.history.size();

  // Generation: 逐个生成 token
  std::vector<llama_token> generated;
  std::string output_buffer; // 累积待输出的文本
  bool in_think_block = false;
  bool think_header_printed = false;

  std::cout << "Assistant: " << std::flush;

  for (int i = 0; i < max_tokens; i++) {
    llama_synchronize(mc.ctx);

    // 使用采样器采样
    llama_token token = llama_sampler_sample(mc.sampler, mc.ctx, -1);

    // 调试：打印生成的 token 信息
    if (i < 5) {
      std::string piece = detokenize(mc, {token});
      std::string escaped;
      for (char c : piece) {
        if (c == '\n')
          escaped += "\\n";
        else if (c == '\r')
          escaped += "\\r";
        else if (c == '\t')
          escaped += "\\t";
        else if (c == ' ')
          escaped += "·";
        else
          escaped += c;
      }
      spdlog::trace("Generated token[{}] = {} -> \"{}\" (len={})", i, token, escaped, piece.length());
    }

    // 接受 token（更新采样器内部状态）
    llama_sampler_accept(mc.sampler, token);

    // 检查结束 (检查 EOS 和 EOT token)
    if (token == mc.eos_token || token == mc.eot_token) {
      spdlog::info("End token encountered at step {} (token={})", i, token);
      break;
    }

    // 添加到结果
    generated.push_back(token);
    mc.history.push_back(token);

    // 将当前 token 转换为文本并添加到缓冲区
    std::string piece = detokenize(mc, {token});
    output_buffer += piece;

    // 检测文本级结束标记（模型生成的普通文本）
    bool found_end_marker = false;
    if (output_buffer.find("<|im_end|>") != std::string::npos ||
        output_buffer.find("<|im_start|>") != std::string::npos) {
      // 找到结束标记，移除它并停止生成
      size_t end_pos = output_buffer.find("<|im_end|>");
      if (end_pos == std::string::npos) {
        end_pos = output_buffer.find("<|im_start|>");
      }
      if (end_pos != std::string::npos) {
        output_buffer = output_buffer.substr(0, end_pos);
      }
      std::cout << std::endl;
      spdlog::info("Text end marker detected at step {}", i);
      found_end_marker = true;
    }

    // 处理 <think> 和 </think> 标签
    while (true) {
      bool found_tag = false;

      // 检测 <think> 标签开始
      size_t think_start = output_buffer.find("<think>");
      if (!in_think_block && think_start != std::string::npos) {
        // 输出标签前的正常内容
        if (think_start > 0) {
          std::cout << output_buffer.substr(0, think_start) << std::flush;
        }
        // 进入思考模式
        in_think_block = true;
        if (!think_header_printed) {
          std::cout << "\n\033[90m[Thinking...]\033[0m\n"
                    << std::flush;
          think_header_printed = true;
        }
        // 移除已处理的内容（包括 <think> 标签）
        output_buffer = output_buffer.substr(think_start + 7);
        found_tag = true;
      }

      // 检测 </think> 标签结束
      size_t think_end = output_buffer.find("</think>");
      if (in_think_block && think_end != std::string::npos) {
        // 输出标签前的思考内容（深灰色）
        if (think_end > 0) {
          std::cout << "\033[90m" << output_buffer.substr(0, think_end) << "\033[0m" << std::flush;
        }
        // 退出思考模式
        in_think_block = false;
        std::cout << "\033[90m[/Thinking]\033[0m\n\n"
                  << std::flush;
        // 移除已处理的内容（包括 </think> 标签）
        output_buffer = output_buffer.substr(think_end + 8);
        found_tag = true;
      }

      // 如果没有找到标签，退出循环
      if (!found_tag) {
        break;
      }
    }

    // 输出缓冲区中的内容
    // 在思考模式下：立即输出所有内容（灰色）
    // 在正常模式下：保留可能的不完整标签
    if (in_think_block) {
      // 在思考块内，立即输出所有思考内容
      if (!output_buffer.empty()) {
        std::cout << "\033[90m" << output_buffer << "\033[0m" << std::flush;
        output_buffer.clear();
      }
    } else {
      // 正常模式：保留可能的不完整标签
      size_t keep_size = found_end_marker ? 0 : 10;
      if (output_buffer.length() > keep_size) {
        std::string to_output = output_buffer.substr(0, output_buffer.length() - keep_size);
        std::cout << to_output << std::flush;
        output_buffer = output_buffer.substr(output_buffer.length() - keep_size);
      }
    }

    // 如果遇到结束标记，停止生成
    if (found_end_marker) {
      break;
    }

    // 准备下一次推理
    auto single = llama_batch_init(1, 0, 1);
    single.token[0] = token;
    single.pos[0] = mc.pos + i;
    single.n_seq_id[0] = 1;
    single.seq_id[0][0] = 0;
    single.logits[0] = 1;
    single.n_tokens = 1;

    if (llama_decode(mc.ctx, single) != 0) {
      llama_batch_free(single);
      break;
    }

    llama_batch_free(single);
  }

  // 输出剩余缓冲区内容
  if (!output_buffer.empty()) {
    if (in_think_block) {
      std::cout << "\033[90m" << output_buffer << "\033[0m" << std::flush;
      std::cout << "\033[90m[/Thinking]\\033[0m\n\n"
                << std::flush;
    } else {
      std::cout << output_buffer << std::flush;
    }
  }

  std::cout << std::endl;

  // 将生成的回复添加到消息历史
  std::string response = detokenize(mc, generated);
  mc.messages.push_back({"assistant", response});

  mc.pos = mc.history.size();

  return response;
}

// 清空对话历史
void clear_history(ModelContext &mc) {
  mc.history.clear();
  mc.messages.clear();
  mc.pos = 0;
  llama_memory_clear(llama_get_memory(mc.ctx), true);
  llama_sampler_reset(mc.sampler);
  spdlog::info("Conversation history cleared");
}

// 释放资源
void cleanup(ModelContext &mc) {
  if (mc.sampler) {
    llama_sampler_free(mc.sampler);
    mc.sampler = nullptr;
  }
  if (mc.ctx) {
    llama_free(mc.ctx);
    mc.ctx = nullptr;
  }
  if (mc.model) {
    llama_model_free(mc.model);
    mc.model = nullptr;
  }
  llama_backend_free();
}

// 显示帮助
void show_help() {
  std::cout << "\n=== Commands ===\n";
  std::cout << "  \\clear  - Clear conversation history\n";
  std::cout << "  \\status - Show current status\n";
  std::cout << "  \\config - Show sampling configuration\n";
  std::cout << "  \\quit   - Exit program\n";
  std::cout << "  \\help   - Show this help\n";
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");
  spdlog::set_level(spdlog::level::info);

  if (argc < 2) {
    spdlog::error("Usage: {} <model.gguf> [template.jinja]", argv[0]);
    spdlog::info("  model.gguf      - Path to the GGUF model file");
    spdlog::info("  template.jinja  - (Optional) Path to custom Jinja2 chat template");
    return 1;
  }

  ModelContext mc = {};
  SamplingConfig sampling_cfg = {};

  // 加载模型
  if (!load_model(mc, argv[1])) {
    return 1;
  }

  // 加载自定义模板（如果提供）
  if (argc >= 3) {
    mc.chat_template = read_file(argv[2]);
    if (mc.chat_template.empty()) {
      spdlog::error("Failed to read template file: {}", argv[2]);
      cleanup(mc);
      return 1;
    }
    spdlog::info("Loaded custom chat template from: {}", argv[2]);
    spdlog::debug("Template content:\n{}", mc.chat_template);
  } else {
    spdlog::info("Using built-in chat template from model");
  }

  // 设置采样器
  if (!setup_sampler(mc, sampling_cfg)) {
    cleanup(mc);
    return 1;
  }

  // 交互循环
  std::cout << "\n=== LLaMA Chat (run2) ===\n";
  std::cout << "Model loaded successfully!\n";
  std::cout << "Type your message and press Enter.\n";
  std::cout << "Type \\help for available commands.\n\n";

  std::string input;

  while (true) {
    std::cout << "You: ";
    if (!std::getline(std::cin, input)) {
      break;
    }

    if (input.empty()) {
      continue;
    }

    // 处理命令
    if (input[0] == '\\') {
      std::string cmd = input.substr(1);

      if (cmd == "clear") {
        clear_history(mc);
        std::cout << "History cleared.\n\n";
        continue;
      } else if (cmd == "quit" || cmd == "exit") {
        break;
      } else if (cmd == "status") {
        std::cout << "\n=== Status ===\n";
        std::cout << "  Messages: " << mc.messages.size() << "\n";
        std::cout << "  History tokens: " << mc.history.size() << "\n";
        std::cout << "  Current position: " << mc.pos << "\n";
        std::cout << "  Remaining context: " << (mc.n_ctx - mc.history.size()) << " tokens\n\n";
        continue;
      } else if (cmd == "config") {
        std::cout << "\n=== Sampling Config ===\n";
        std::cout << "  temperature: " << sampling_cfg.temperature << "\n";
        std::cout << "  top_p: " << sampling_cfg.top_p << "\n";
        std::cout << "  top_k: " << sampling_cfg.top_k << "\n";
        std::cout << "  min_p: " << sampling_cfg.min_p << "\n";
        std::cout << "  penalty_repeat: " << sampling_cfg.penalty_repeat << "\n";
        std::cout << "  penalty_last_n: " << sampling_cfg.penalty_last_n << "\n\n";
        continue;
      } else if (cmd == "help") {
        show_help();
        continue;
      } else {
        std::cout << "Unknown command: " << cmd << "\n";
        std::cout << "Type \\help for available commands.\n\n";
        continue;
      }
    }

    // 生成回复
    auto start = llama_time_us();
    std::string response = generate(mc, input, 2048);
    auto end = llama_time_us();

    double duration = (end - start) / 1000.0;
    spdlog::info("Generated in {:.2f} ms", duration);
    std::cout << std::endl;
  }

  // 清理
  spdlog::info("Cleaning up...");
  cleanup(mc);
  spdlog::info("Done!");

  return 0;
}
