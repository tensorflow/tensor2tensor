#include "third_party/py/tensor2tensor/data_generators/ops/subword_text_encoder.h"

#include "testing/base/public/gunit.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_testutil.h"

namespace tensor2tensor {
namespace {

TEST(SubwordTextEncoderTest, EncodesSubTokens) {
  SubwordTextEncoder encoder("third_party/py/tensor2tensor/"
                             "data_generators/ops/testdata/subwords");
  std::vector<int> t;
  encoder.Encode("the quick brown fox jumps over the lazy dog", &t);
  EXPECT_EQ(t, std::vector<int>({2, 3, 4, 5, 6, 7, 8, 9, 2, 11, 12, 1}));
}

TEST(SubwordTextEncoderTest, EncodesUnicodeSubTokens) {
  SubwordTextEncoder encoder("third_party/py/tensor2tensor/"
                             "data_generators/ops/testdata/subwords");
  std::vector<int> t;
  encoder.Encode("ɧęĻĽÒ", &t);
  EXPECT_EQ(t, std::vector<int>({13, 14, 1}));
}

TEST(SubwordTextEncoderTest, EncodesUnicodeCodePoints) {
  SubwordTextEncoder encoder("third_party/py/tensor2tensor/"
                             "data_generators/ops/testdata/subwords");
  std::vector<int> t;
  encoder.Encode("⻦ ⻭", &t);
  EXPECT_EQ(t, std::vector<int>({15, 18, 16, 17, 1}));
}

TEST(SubwordTextEncoderTest, EncodesCharactersNotInAlphabet) {
  SubwordTextEncoder encoder("third_party/py/tensor2tensor/"
                             "data_generators/ops/testdata/subwords");
  std::vector<int> t;
  encoder.Encode("!", &t);
  // Subtokens: '\', '3', '3', ';', '_', '<eos>', '<pad>'.
  EXPECT_EQ(t, std::vector<int>({19, 23, 23, 30, 17, 1}));
}

}  // namespace
}  // namespace tensor2tensor
