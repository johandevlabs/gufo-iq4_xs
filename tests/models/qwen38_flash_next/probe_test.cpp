#include "src/models/qwen38_flash_next/probe.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gufo::core::GgmlType;
using Tensor = std::pair<std::string, GgmlType>;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Independent GGUF fixture: each tensor has one 256-element row and enough
// payload for every tested storage format. The probe never reads its values.
bool Probe(const std::vector<Tensor>& tensors,
           std::optional<std::uint32_t> blocks = 49,
           std::optional<std::uint32_t> nextn = 1,
           const std::string& architecture = "qwen4exp") {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{1} + blocks.has_value() + nextn.has_value());
  text("general.architecture");
  pod(std::uint32_t{8});
  text(architecture);
  for (const auto& [key, value] :
       {std::pair{"qwen4exp.block_count", blocks},
        std::pair{"qwen4exp.nextn_predict_layers", nextn}}) {
    if (value) {
      text(key);
      pod(std::uint32_t{4});
      pod(*value);
    }
  }
  std::uint64_t offset = 0;
  for (const auto& [name, type] : tensors) {
    text(name);
    pod(std::uint32_t{1});
    pod(std::uint64_t{256});
    pod(static_cast<std::uint32_t>(type));
    pod(offset);
    offset += 2048;
  }
  bytes.resize((bytes.size() + 31) / 32 * 32 + offset);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(reader != nullptr, error.c_str());
  return gufo::models::qwen38_flash_next::ProbeFlashNextGguf(*reader);
}

void CheckExpertFormats() {
  for (const auto* role : {"gate", "up", "down"}) {
    const auto name = std::string("blk.47.ffn_") + role + "_exps.weight";
    for (const auto type :
         {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1,
          GgmlType::kQ5_0, GgmlType::kQ8_0, GgmlType::kQ2_0, GgmlType::kIQ2_S,
          GgmlType::kIQ3_XXS, GgmlType::kIQ3_S, GgmlType::kIQ4_XS,
          GgmlType::kIQ4_NL}) {
      Require(Probe({{name, type}}), "supported expert format declined");
    }
    for (const auto type : {GgmlType::kIQ2_XXS, GgmlType::kQ4_0, GgmlType::kF16,
                            GgmlType::kQ2_K}) {
      Require(!Probe({{name, type}}), "unsupported trunk expert claimed");
    }
  }
  Require(!Probe({{"blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K},
                  {"blk.47.ffn_down_exps.weight", GgmlType::kIQ2_XXS}}),
          "unsupported later expert missed after a supported expert");
}

// IQ experts have no tiled MMQ route, so a complete layer with an IQ expert
// is claimed only in a routed F16 (WMMA) layout, as the executor requires.
void CheckIqExpertLayouts() {
  const auto layer = [](GgmlType gate, GgmlType up, GgmlType down) {
    return std::vector<Tensor>{{"blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K},
                               {"blk.0.ffn_up_exps.weight", GgmlType::kQ4_K},
                               {"blk.0.ffn_down_exps.weight", GgmlType::kQ5_1},
                               {"blk.47.ffn_gate_exps.weight", gate},
                               {"blk.47.ffn_up_exps.weight", up},
                               {"blk.47.ffn_down_exps.weight", down}};
  };
  using enum GgmlType;
  for (const auto& [gate, down] :
       {std::pair{kIQ3_S, kIQ4_NL}, std::pair{kIQ3_S, kQ8_0},
        std::pair{kIQ4_XS, kIQ4_NL}, std::pair{kIQ4_XS, kQ5_1},
        std::pair{kQ4_K, kIQ4_NL}, std::pair{kQ5_K, kIQ4_NL}}) {
    Require(Probe(layer(gate, gate, down)), "supported IQ layout declined");
  }
  Require(Probe(layer(kQ4_K, kQ4_K, kQ4_K)),
          "layout without IQ experts declined");
  Require(Probe(layer(kQ6_K, kQ8_0, kQ5_1)),
          "mixed layout without IQ experts declined");
  Require(!Probe(layer(kIQ3_S, kIQ4_XS, kIQ4_NL)), "mismatched IQ gate/up");
  Require(!Probe(layer(kIQ3_S, kIQ3_S, kQ4_K)), "IQ gate/up with Q4_K down");
  Require(!Probe(layer(kIQ4_NL, kIQ4_NL, kIQ4_NL)), "IQ4_NL gate/up");
  Require(!Probe(layer(kQ6_K, kQ6_K, kIQ4_NL)), "Q6_K gate/up, IQ4_NL down");
  Require(!Probe(layer(kQ8_0, kQ8_0, kIQ4_XS)), "IQ4_XS down");
  Require(!Probe(layer(kQ4_K, kIQ3_S, kIQ4_NL)), "IQ up with Q4_K gate");
  // An incomplete layer is left to the loader, which names the missing tensor.
  Require(Probe({{"blk.47.ffn_gate_exps.weight", kIQ3_S},
                 {"blk.47.ffn_down_exps.weight", kQ4_K}}),
          "incomplete layer declined instead of left to the loader");
}

void CheckIgnoredTensors() {
  const std::vector<Tensor> trunk = {
      {"blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K},
      {"blk.47.ffn_down_exps.weight", GgmlType::kQ8_0}};
  for (const auto* name :
       {"blk.48.ffn_gate_exps.weight",  // Appended MTP layer.
        "blk.49.ffn_up_exps.weight",    // Beyond the model's layer count.
        "blk.4294967296.ffn_gate_exps.weight",
        "blk.00.ffn_gate_exps.weight",  // Noncanonical layer spelling.
        "blk.-1.ffn_gate_exps.weight", "blk.+0.ffn_gate_exps.weight",
        "blk..ffn_gate_exps.weight", "blk.", "other.blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_gate_exps.weight.extra", "blk.0.extra.ffn_gate_exps.weight",
        "per_layer_token_embd.weight"}) {
    auto tensors = trunk;
    tensors.emplace_back(name, GgmlType::kIQ4_NL);
    Require(Probe(tensors), "unused tensor incorrectly declined the trunk");
  }
  Require(Probe({{"blk.47.ffn_up_exps.weight", GgmlType::kIQ2_XXS}}, 48, 1),
          "appended layer was not excluded from block_count");
  Require(!Probe({{"blk.47.ffn_up_exps.weight", GgmlType::kIQ2_XXS}}, 48,
                 std::nullopt),
          "absent nextn metadata incorrectly excluded the final trunk layer");
}

void CheckMetadata() {
  // A corrupt block_count must not size the probe's work: only the layers
  // present are checked, including one near the 32-bit limit.
  constexpr std::uint32_t kHugeBlocks = 0xffffffff;
  Require(Probe({{"blk.4294967293.ffn_gate_exps.weight", GgmlType::kIQ3_S},
                 {"blk.4294967293.ffn_up_exps.weight", GgmlType::kIQ3_S},
                 {"blk.4294967293.ffn_down_exps.weight", GgmlType::kIQ4_NL}},
                kHugeBlocks),
          "huge block_count declined a supported layer");
  Require(!Probe({{"blk.4294967293.ffn_gate_exps.weight", GgmlType::kIQ3_S},
                  {"blk.4294967293.ffn_up_exps.weight", GgmlType::kIQ4_XS},
                  {"blk.4294967293.ffn_down_exps.weight", GgmlType::kIQ4_NL}},
                 kHugeBlocks),
          "huge block_count claimed an unsupported layer");
  Require(!Probe({}, 49, 1, "qwen3"), "unrelated architecture claimed");
  Require(!Probe({}, std::nullopt), "missing block_count claimed");
  Require(!Probe({}, 0, 0), "zero block_count claimed");
  Require(!Probe({}, 48, 48), "empty trunk claimed");
  Require(!Probe({}, 48, 49), "nextn count underflowed");
  // The probe chooses a provider; the loader still diagnoses missing tensors.
  Require(Probe({}), "probe unexpectedly required complete weights");
}

}  // namespace

int main() {
  CheckExpertFormats();
  CheckIqExpertLayouts();
  CheckIgnoredTensors();
  CheckMetadata();
  std::cout << "PASS: Flash-Next provider probe\n";
}
