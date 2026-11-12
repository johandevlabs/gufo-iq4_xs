#include "src/models/qwen/probe.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "src/families/qwen/config/qwen_config.hpp"
#include "src/models/qwen/weight_formats.hpp"

namespace gufo::models::qwen {

namespace {

/// True unless a tensor the loader reads (embedding, LM head, trunk layers)
/// is quantized in a format the Qwen kernels cannot decode, such as the
/// IQ1/IQ2/IQ3_XXS or Q2_K experts of low-bit mixes. Dense F16/I32 tensors
/// are left to the loader, which names the tensor if it needs another type.
/// Appended MTP layers are not part of the trunk and are not checked.
bool WeightFormatsSupported(const core::GgufReader& reader,
                            const std::string& prefix) {
  const auto block_count = reader.GetMetadataUint32(prefix + "block_count");
  const auto nextn =
      reader.GetMetadataUint32(prefix + "nextn_predict_layers").value_or(0);
  const auto trunk_layers =
      block_count && nextn < *block_count ? *block_count - nextn : 0U;
  for (const auto& tensor : reader.GetTensors()) {
    const std::string_view name = tensor.name;
    if (name.starts_with("blk.")) {
      std::uint32_t layer = 0;
      const auto [end, error] =
          std::from_chars(name.data() + 4, name.data() + name.size(), layer);
      if (error != std::errc{} || (block_count && layer >= trunk_layers)) {
        continue;
      }
    } else if (name != "token_embd.weight" && name != "output.weight") {
      continue;
    }
    if (!IsQwenWeightFormat(tensor.type) &&
        tensor.type != core::GgmlType::kF16 &&
        tensor.type != core::GgmlType::kI32) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool ProbeQwenGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture == "deepseek4" || architecture == "qwen4exp") {
    return false;
  }
  // Qwen3.6-35B-A3B and Ornith-1.5-122B share the hybrid runner with routed
  // MoE layers (Strycast).
  if (architecture == "qwen35moe") {
    const auto width =
        reader.GetMetadataUint64("qwen35moe.embedding_length").value_or(0);
    return (width == 2048 || width == 3072) &&
           WeightFormatsSupported(reader, "qwen35moe.");
  }
  if (architecture != "qwen" && architecture != "qwen2" &&
      architecture != "qwen35" && architecture != "qwen38") {
    return false;
  }
  const std::string prefix = std::string(architecture) + ".";
  const auto embedding_length =
      reader.GetMetadataUint64("qwen35.embedding_length")
          .value_or(reader.GetMetadataUint64(prefix + "embedding_length")
                        .value_or(0));
  return embedding_length == 5120 && WeightFormatsSupported(reader, prefix);
}

std::optional<sampling::TextSamplingDefaults> QwenSamplingDefaults(
    const core::ModelConfig& config) {
  return sampling::PresetDefaults(sampling::TextPreset(config));
}

std::optional<sampling::TextSamplingDefaults> QwenSamplingDefaults(
    const core::GgufReader& reader) {
  const auto config = ExtractQwenConfig(reader);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return QwenSamplingDefaults(*config);
}

bool ProbeQwen(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader)
        return false;
      return ProbeQwenGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen
