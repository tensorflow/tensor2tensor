#ifndef TENSOR2TESNOR_DATA_GENERATORS_OPS_SUBWORD_TEXT_ENCODER_H_
#define TENSOR2TESNOR_DATA_GENERATORS_OPS_SUBWORD_TEXT_ENCODER_H_

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/icu/include/unicode/uchar.h"
#include "third_party/tensorflow/core/framework/tensor.h"

namespace tensor2tensor {

// A subword text encoder with built in tokenizer.
//
// Equivalent to tensor2tensor's subword text
// https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py,
// This code (or a suitable replacement) should eventually move into tfds
//   and should be deleted from tensor2tensor.

class SubwordTextEncoder {
 public:
  explicit SubwordTextEncoder(const string& vocab_filename);
  virtual ~SubwordTextEncoder() {}

  // Breaks up input text into subtokens.
  void Encode(absl::string_view text, std::vector<int>* ids);

 private:
  // Given a full token as input, breaks the token up into subtokens and appends
  // corresponding IDs to the ids vector.
  void EncodeSubtokens(absl::string_view token, std::vector<int>* ids);

  // Escapes a token so unencodable characters are replaced by escape sequences.
  string EscapeToken(absl::string_view token);

  // Maps subword tokens to IDs.
  absl::flat_hash_map<string, int64> vocab_;
  // A set containing all valid unicode code points that can be encoded without
  // being escaped.
  absl::flat_hash_set<UChar32> alphabet_;
};

}  // namespace tensor2tensor

#endif  // TENSOR2TESNOR_DATA_GENERATORS_OPS_SUBWORD_TEXT_ENCODER_H_
