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
using ::tensorflow::shape_inference::InferenceContext;

// TODO(noam): this op packs a dataset of pairs of sequences (inputs, targets)
// Generalize later to an arbitrary number of sequences.
REGISTER_OP("PackSequences2")
    .Input("inputs: int64")
    .Input("targets: int64")
    .Input("inputs_max_length: int32")
    .Input("targets_max_length: int32")
    .Output("inputs_packed: int64")
    .Output("inputs_segmentation: int32")
    .Output("inputs_position: int32")
    .Output("targets_packed: int64")
    .Output("targets_segmentation: int32")
    .Output("targets_position: int32")
    .SetShapeFn([](InferenceContext* ctx) {
                  for (int i=0; i < ctx->num_outputs(); i++) {
                    ctx->set_output(i, ctx->Matrix(ctx->UnknownDim(),
                                                   ctx->UnknownDim()));
                  }
                  return Status::OK();
                });

class PackSequences2Op : public OpKernel {
 public:
  explicit PackSequences2Op(
      OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    auto inputs = ctx->input(0).matrix<int64>();
    auto targets = ctx->input(1).matrix<int64>();
    int inputs_max_length = ctx->input(2).scalar<int32>()();
    int targets_max_length = ctx->input(3).scalar<int32>()();
    int n = inputs.dimension(0);
    std::vector<int> inputs_lengths(n);
    std::vector<int> targets_lengths(n);
    int padded_inputs_length =
        std::min(static_cast<int>(inputs.dimension(1)), inputs_max_length);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < padded_inputs_length; j++) {
          if (inputs(i, j) != 0)
            inputs_lengths[i]++;
      }
    }
    int padded_targets_length =
        std::min(static_cast<int>(targets.dimension(1)), targets_max_length);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < padded_targets_length; j++) {
          if (targets(i, j) != 0)
            targets_lengths[i]++;
      }
    }
    int num_combined = 0;
    std::vector<int> combined_inputs_length;
    std::vector<int> combined_targets_length;
    std::vector<std::vector<int> > combined_sequence_ids;
    for (int seq_id = 0; seq_id < n; seq_id++) {
      int inputs_length = inputs_lengths[seq_id];
      int targets_length = targets_lengths[seq_id];
      for (int combined_id = std::max(0, num_combined - 1000); true;
           combined_id++) {
        if (combined_id == num_combined) {
          combined_inputs_length.push_back(inputs_length);
          combined_targets_length.push_back(targets_length);
          combined_sequence_ids.push_back(std::vector<int>(1, seq_id));
          num_combined++;
          break;
        } else if (
            (combined_inputs_length[combined_id] + inputs_length
             <= inputs_max_length) &&
            (combined_targets_length[combined_id] + targets_length
             <= targets_max_length)) {
          combined_inputs_length[combined_id] += inputs_length;
          combined_targets_length[combined_id] += targets_length;
          combined_sequence_ids[combined_id].push_back(seq_id);
          break;
        }
      }
    }

    auto output_shape_inputs = TensorShape(
        {static_cast<int64>(num_combined),
         static_cast<int64>(inputs_max_length)});
    auto output_shape_targets = TensorShape(
        {static_cast<int64>(num_combined),
         static_cast<int64>(targets_max_length)});

    Tensor* inputs_packed;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, output_shape_inputs, &inputs_packed));
    auto inputs_packed_m = inputs_packed->matrix<int64>();
    inputs_packed_m.setZero();

    Tensor* inputs_segmentation;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            1, output_shape_inputs, &inputs_segmentation));
    auto inputs_segmentation_m = inputs_segmentation->matrix<int32>();
    inputs_segmentation_m.setZero();

    Tensor* inputs_position;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, output_shape_inputs, &inputs_position));
    auto inputs_position_m = inputs_position->matrix<int32>();
    inputs_position_m.setZero();

    Tensor* targets_packed;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        3, output_shape_targets, &targets_packed));
    auto targets_packed_m = targets_packed->matrix<int64>();
    targets_packed_m.setZero();

    Tensor* targets_segmentation;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            4, output_shape_targets, &targets_segmentation));
    auto targets_segmentation_m = targets_segmentation->matrix<int32>();
    targets_segmentation_m.setZero();

    Tensor* targets_position;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(5, output_shape_targets, &targets_position));
    auto targets_position_m = targets_position->matrix<int32>();
    targets_position_m.setZero();

    for (int combined_id = 0; combined_id < num_combined; combined_id++) {
      int inputs_pos = 0;
      int targets_pos = 0;
      for (int i=0; i < combined_sequence_ids[combined_id].size(); i++) {
        int seq_id = combined_sequence_ids[combined_id][i];
        for (int j=0; j < inputs_lengths[seq_id]; j++) {
          inputs_packed_m(combined_id, inputs_pos) = inputs(seq_id, j);
          inputs_segmentation_m(combined_id, inputs_pos) = i + 1;
          inputs_position_m(combined_id, inputs_pos) = j;
          inputs_pos++;
        }
        for (int j=0; j < targets_lengths[seq_id]; j++) {
          targets_packed_m(combined_id, targets_pos) = targets(seq_id, j);
          targets_segmentation_m(combined_id, targets_pos) = i + 1;
          targets_position_m(combined_id, targets_pos) = j;
          targets_pos++;
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PackSequences2").Device(DEVICE_CPU),
                        PackSequences2Op);

}  // namespace
}  // namespace tensor2tensor
