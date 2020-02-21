#include "third_party/py/tensor2tensor/data_generators/ops/subword_text_encoder.h"

#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/icu/include/unicode/uchar.h"
#include "third_party/icu/include/unicode/utf8.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/platform/env.h"

namespace tensor2tensor {
namespace {

using ::tensorflow::Env;

// End of Sequence token ID to insert at end of encoded text.
constexpr int64 kEosTokenId = 1;

}  // namespace

SubwordTextEncoder::SubwordTextEncoder(const std::string& vocab_filename) {
  // TODO(ormandi): Add a unified vocabulary reader function.
  std::string vocab_contents;
  TF_CHECK_OK(
      ReadFileToString(Env::Default(), vocab_filename, &vocab_contents));
  std::vector<absl::string_view> vocab_list =
      absl::StrSplit(vocab_contents, '\n');
  // Strip trailing newline by skipping last element, then strip the first and
  // last chars to remove enclosing quotes.
  auto vocab_size = vocab_list.size() - vocab_list.back().empty();
  for (auto i = 0; i < vocab_size; ++i) {
    absl::string_view token =
        vocab_list[i].substr(1, vocab_list[i].length() - 2);
    int char_index = 0;
    do {
      // Note throughout that these strings are unicode so we iterate over utf-8
      // code points, which may be between 8-32 bits long, using U8_NEXT. It is
      // important never to iterate directly over ascii characters or models
      // will fail to handle non-ascii alphabets properly.
      UChar32 c;
      U8_NEXT(token, char_index, token.length(), c);
      CHECK_GE(c, 0);
      alphabet_.insert(c);
    } while (char_index < token.length());
    vocab_.insert({std::string(token), i});
  }
}

void SubwordTextEncoder::Encode(absl::string_view text, std::vector<int>* ids) {
  ids->clear();
  int token_start = 0;
  int token_end = 0;
  UChar32 c;
  UChar32 next_c;
  U8_NEXT(text, token_end, text.length(), c);
  CHECK_GE(c, 0);
  while (token_end <= text.length()) {
    int next_end = token_end;
    U8_NEXT(text, next_end, text.length(), next_c);
    CHECK_GE(next_c, 0);
    // Subtoken break when switching from non-alphanum to alphanum, or when
    // reaching the end of the original token.
    if (u_isalnum(next_c) != u_isalnum(c) || token_end >= text.length()) {
      absl::string_view next_token =
          text.substr(token_start, token_end - token_start);
      if (next_token != " ") {
        EncodeSubtokens(next_token, ids);
      }
      token_start = token_end;
    }
    token_end = next_end;
    c = next_c;
  }
  ids->push_back(kEosTokenId);
}

void SubwordTextEncoder::EncodeSubtokens(
    absl::string_view token, std::vector<int> *ids) {
  std::string token_s = EscapeToken(token);
  token = token_s;
  int subtoken_start = 0;
  // TODO(noam): this algorithm is quadratic in the length of the token.
  //   We should instead start with a length equal to the maximum subtoken
  //   length in the vocabulary.
  int subtoken_end = token.length();
  while (subtoken_start < token.length()) {
    absl::string_view subtoken =
        token.substr(subtoken_start, subtoken_end - subtoken_start);
    auto iter = vocab_.find(subtoken);
    if (iter != vocab_.end()) {
      ids->push_back(iter->second);
      subtoken_start = subtoken_end;
      // TODO(noam): again, set subtoken_end forward only enough to catch
      // the longest subtoken in the vocabulary.
      subtoken_end = token.length();
    } else {
      U8_BACK_1((const uint8_t*)token_s.data(), 0, subtoken_end);
      if (subtoken_end <= subtoken_start) {
        LOG(FATAL) << "Unencodable tokens found.";
      }
    }
  }
}

std::string SubwordTextEncoder::EscapeToken(absl::string_view token) {
  std::string token_s;
  int i = 0;
  do {
    int prev = i;
    UChar32 c;
    U8_NEXT(token, i, token.length(), c);
    CHECK_GE(c, 0);
    if (c == '_') {
      absl::StrAppend(&token_s, "\\u");
    } else if (c == '\\') {
      absl::StrAppend(&token_s, "\\\\");
    } else if (c == '\n' || alphabet_.find(c) == alphabet_.end()) {
      absl::StrAppend(&token_s, "\\", c, ";");
    } else {
      absl::StrAppend(&token_s, token.substr(prev, i - prev));
    }
  } while (i < token.length());
  absl::StrAppend(&token_s, "_");
  return token_s;
}

}  // namespace tensor2tensor
