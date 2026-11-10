#include "src/models/qwen38_flash_next/probe.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen4exp") {
    return false;
  }
  const auto block_count = reader.GetMetadataUint32("qwen4exp.block_count");
  const auto nextn_layers =
      reader.GetMetadataUint32("qwen4exp.nextn_predict_layers").value_or(0);
  if (!block_count || nextn_layers >= *block_count) {
    return false;
  }
  const auto trunk_layers = *block_count - nextn_layers;
  // Check only canonical weights bound by ModelWeights::Bind. Appended MTP
  // layers and unrelated tensors are not loaded as part of the trunk.
  static constexpr std::array<core::GgmlType, 12> kExpertFormats = {
      core::GgmlType::kQ4_K,  core::GgmlType::kQ5_K,   core::GgmlType::kQ6_K,
      core::GgmlType::kQ5_1,  core::GgmlType::kQ5_0,   core::GgmlType::kQ8_0,
      core::GgmlType::kQ2_0,  core::GgmlType::kIQ2_S,  core::GgmlType::kIQ3_XXS,
      core::GgmlType::kIQ3_S, core::GgmlType::kIQ4_XS, core::GgmlType::kIQ4_NL};
  // Gate, up and down formats of the layers present, for the IQ layout rule.
  // Nothing is sized from block_count, which a corrupt file can inflate.
  std::map<std::uint32_t, std::array<std::optional<core::GgmlType>, 3>>
      layer_formats;
  for (const auto& tensor : reader.GetTensors()) {
    auto name = tensor.name;
    if (!name.starts_with("blk.")) {
      continue;
    }
    name.remove_prefix(4);
    std::uint32_t layer = 0;
    const auto [end, error] =
        std::from_chars(name.data(), name.data() + name.size(), layer);
    if (error != std::errc{} || layer >= trunk_layers ||
        (name.starts_with('0') && end != name.data() + 1)) {
      continue;
    }
    name.remove_prefix(end - name.data());
    const std::size_t role = name == ".ffn_gate_exps.weight"   ? 0
                             : name == ".ffn_up_exps.weight"   ? 1
                             : name == ".ffn_down_exps.weight" ? 2
                                                               : 3;
    if (role == 3) {
      continue;
    }
    if (std::find(kExpertFormats.begin(), kExpertFormats.end(), tensor.type) ==
        kExpertFormats.end()) {
      return false;
    }
    layer_formats[layer][role] = tensor.type;
  }
  // The loader diagnoses missing experts; only complete layers are checked.
  return std::all_of(
      layer_formats.begin(), layer_formats.end(), [](const auto& entry) {
        const auto& formats = entry.second;
        return !formats[0] || !formats[1] || !formats[2] ||
               IqExpertFormatsSupported(*formats[0], *formats[1], *formats[2]);
      });
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults() {
  return sampling::PresetDefaults(sampling::TextModelPreset::kQwen38);
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults(
    const core::GgufReader&) {
  return FlashNextSamplingDefaults();
}

bool ProbeFlashNext(std::string_view model_root) {
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
      return ProbeFlashNextGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next
