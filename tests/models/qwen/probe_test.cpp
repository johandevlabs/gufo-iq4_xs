#include "src/models/qwen/probe.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gufo::core::GgmlType;
using Tensor = std::pair<std::string, GgmlType>;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Shape {
  std::string architecture;
  std::uint64_t width;
  std::uint32_t blocks;
  std::uint32_t nextn;
};

const Shape k35b{"qwen35moe", 2048, 41, 1};
const Shape kOrnith{"qwen35moe", 3072, 49, 1};
const Shape k27b{"qwen35", 5120, 65, 1};

// Independent GGUF fixture: each tensor has one 256-element row and enough
// payload for every tested storage format. The probe never reads its values.
bool Probe(const Shape& shape, const std::vector<Tensor>& tensors) {
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
  const std::string prefix = shape.architecture + ".";
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{4});
  text("general.architecture");
  pod(std::uint32_t{8});
  text(shape.architecture);
  text(prefix + "embedding_length");
  pod(std::uint32_t{10});
  pod(shape.width);
  text(prefix + "block_count");
  pod(std::uint32_t{4});
  pod(shape.blocks);
  text(prefix + "nextn_predict_layers");
  pod(std::uint32_t{4});
  pod(shape.nextn);
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
  Require(reader != nullptr, error);
  return gufo::models::qwen::ProbeQwenGguf(*reader);
}

// A trunk as the loader reads it, with every expert in `expert`.
std::vector<Tensor> Trunk(GgmlType expert) {
  return {{"token_embd.weight", GgmlType::kQ6_K},
          {"output.weight", GgmlType::kQ6_K},
          {"output_norm.weight", GgmlType::kF32},
          {"blk.0.attn_norm.weight", GgmlType::kF32},
          {"blk.0.attn_qkv.weight", GgmlType::kQ8_0},
          {"blk.0.ffn_gate_inp.weight", GgmlType::kF32},
          {"blk.0.ffn_gate_exps.weight", expert},
          {"blk.0.ffn_up_exps.weight", expert},
          {"blk.0.ffn_down_exps.weight", expert}};
}

void CheckSupportedFormats() {
  for (const auto& shape : {k35b, kOrnith, k27b}) {
    for (const auto type :
         {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ8_K,
          GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K, GgmlType::kQ3_K,
          GgmlType::kIQ4_NL, GgmlType::kIQ4_XS, GgmlType::kIQ3_S}) {
      Require(Probe(shape, Trunk(type)),
              shape.architecture + ": supported format declined: " +
                  std::string(gufo::core::ToString(type)));
    }
    // Dense F16 is left to the loader, which names the tensor.
    Require(Probe(shape, Trunk(GgmlType::kF16)),
            shape.architecture + ": F16 trunk declined at probe time");
  }
}

void CheckUnsupportedFormats() {
  for (const auto& shape : {k35b, kOrnith, k27b}) {
    for (const auto type :
         {GgmlType::kIQ1_S, GgmlType::kIQ1_M, GgmlType::kIQ2_XXS,
          GgmlType::kIQ2_XS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS,
          GgmlType::kQ2_K, GgmlType::kQ4_0, GgmlType::kQ4_1, GgmlType::kQ5_0,
          GgmlType::kQ5_1}) {
      const std::string what =
          shape.architecture + " " + std::string(gufo::core::ToString(type));
      Require(!Probe(shape, Trunk(type)), what + ": trunk experts claimed");
      auto tensors = Trunk(GgmlType::kQ4_K);
      tensors.front().second = type;
      Require(!Probe(shape, tensors), what + ": embedding claimed");
      tensors = Trunk(GgmlType::kQ4_K);
      tensors[1].second = type;
      Require(!Probe(shape, tensors), what + ": LM head claimed");
    }
  }
}

void CheckIgnoredTensors() {
  for (const auto& shape : {k35b, kOrnith, k27b}) {
    auto tensors = Trunk(GgmlType::kIQ4_XS);
    // Appended MTP layers and tensors the loader never reads.
    const auto mtp = "blk." + std::to_string(shape.blocks - 1) + ".";
    tensors.emplace_back(mtp + "ffn_gate_exps.weight", GgmlType::kIQ2_XXS);
    tensors.emplace_back(mtp + "nextn.eh_proj.weight", GgmlType::kQ2_K);
    tensors.emplace_back("rope_freqs.weight", GgmlType::kIQ1_S);
    Require(Probe(shape, tensors),
            shape.architecture + ": unread tensor declined the trunk");
    // The last trunk layer is still checked.
    const auto last = "blk." + std::to_string(shape.blocks - 2) + ".";
    tensors.emplace_back(last + "ffn_down_exps.weight", GgmlType::kIQ2_S);
    Require(!Probe(shape, tensors),
            shape.architecture + ": last trunk layer not checked");
  }
}

void CheckShapes() {
  Require(!Probe({"qwen35moe", 4096, 41, 1}, Trunk(GgmlType::kQ4_K)),
          "unknown qwen35moe width claimed");
  Require(!Probe({"qwen35", 4096, 65, 1}, Trunk(GgmlType::kQ4_K)),
          "unknown dense width claimed");
}

}  // namespace

int main() {
  try {
    CheckSupportedFormats();
    CheckUnsupportedFormats();
    CheckIgnoredTensors();
    CheckShapes();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "PASS: Qwen provider probe\n";
  return 0;
}
