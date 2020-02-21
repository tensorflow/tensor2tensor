#include "third_party/py/tensor2tensor/data_generators/ops/subword_text_encoder.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/types.h"

namespace tensor2tensor {
namespace {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("SubwordTextEncoderEncode")
    .Input("s: string")
    .Output("encoded: int64")
    .Attr("vocab_filename: string")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->Vector(ctx->UnknownDim()));
      return Status::OK();
    });

class SubwordTextEncoderEncodeOp : public OpKernel {
 public:
  explicit SubwordTextEncoderEncodeOp(
      OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::string vocab_filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_filename", &vocab_filename));
    encoder_ = absl::make_unique<SubwordTextEncoder>(vocab_filename);
  }

  void Compute(OpKernelContext* ctx) override {
    // Get input string and deserialize into ArticleExample proto.
    absl::string_view s = ctx->input(0).scalar<tstring>()();

    // Construct encoded output tensors.
    std::vector<int> encoded_ids;
    encoder_->Encode(s, &encoded_ids);
    Tensor* encoded;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_output(0, TensorShape(
            {static_cast<int64>(encoded_ids.size())}), &encoded));
    auto encoded_vec = encoded->vec<int64>();
    // TODO(noam): find someone who remembers c++ eigen and ask the proper way
    // to copy a std::Vector to an Eigen whatever-this-is
    for (int i = 0; i < encoded_ids.size(); i++) {
      encoded_vec(i) = encoded_ids[i];
    }
  }

 private:
  std::unique_ptr<SubwordTextEncoder> encoder_;
};

REGISTER_KERNEL_BUILDER(Name("SubwordTextEncoderEncode").Device(DEVICE_CPU),
                        SubwordTextEncoderEncodeOp);

}  // namespace
}  // namespace tensor2tensor
