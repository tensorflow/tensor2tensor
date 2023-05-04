#include "base/integral_types.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/types.h"
#include "third_party/tensorflow/core/framework/types.proto.h"
#include "third_party/tensorflow/core/platform/errors.h"

namespace tensor2tensor {
namespace {

using ::tensorflow::bfloat16;
using ::tensorflow::DataTypeVector;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpInputList;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::OpOutputList;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TTypes;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

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
                  return tensorflow::Status();
                });

// Given a collection of examples, each of which consists of two sequences
// ('inputs' and 'targets') this op packs them into as few packed/combined
// examples as possible, to try to minimize padding.
class PackSequences2Op : public OpKernel {
 public:
  explicit PackSequences2Op(
      OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    auto inputs = ctx->input(0).matrix<int64_t>();
    auto targets = ctx->input(1).matrix<int64_t>();
    int inputs_max_length = ctx->input(2).scalar<int32_t>()();
    int targets_max_length = ctx->input(3).scalar<int32_t>()();
    int n = inputs.dimension(0);  // Number of examples in the input.
    std::vector<int> inputs_lengths(n);
    std::vector<int> targets_lengths(n);
    // Calculate, in 'inputs_lengths', the actual length of each input sequence
    // in "inputs", ignoring padding:
    int padded_inputs_length =
        std::min(static_cast<int>(inputs.dimension(1)), inputs_max_length);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < padded_inputs_length; j++) {
          if (inputs(i, j) != 0)
            inputs_lengths[i]++;
      }
    }
    // Calculate, in 'targets_lengths', the actual length of each target
    // sequence in "targets", ignoring padding:
    int padded_targets_length =
        std::min(static_cast<int>(targets.dimension(1)), targets_max_length);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < padded_targets_length; j++) {
          if (targets(i, j) != 0)
            targets_lengths[i]++;
      }
    }
    int num_combined = 0;  // Number of combined examples currently generated.
    std::vector<int> combined_inputs_length;
    std::vector<int> combined_targets_length;
    std::vector<std::vector<int> > combined_sequence_ids;
    for (int seq_id = 0; seq_id < n; seq_id++) {
      int inputs_length = inputs_lengths[seq_id];
      int targets_length = targets_lengths[seq_id];
      // Try to fit the current example, 'seq_id', into one of the existing
      // packed examples. The code checks to see if the current example fits in
      // any of the last 1000 packed examples already generated. If it fits in
      // any, then the example if packed there. Otherwise, a new packed example
      // is generated with the new example, and 'num_combined' is increased to
      // reflect this:
      for (int combined_id = std::max(0, num_combined - 1000); true;
           combined_id++) {
        if (combined_id == num_combined) {
          // The current example, 'seq_id', did not fit in any of the current
          // packed examples, so, we generate a new packed example:
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
          // The current example, 'seq_id', fits in one of the current packed
          // examples, 'combined_id', so, we just add it there,
          combined_inputs_length[combined_id] += inputs_length;
          combined_targets_length[combined_id] += targets_length;
          combined_sequence_ids[combined_id].push_back(seq_id);
          break;
        }
      }
    }

    auto output_shape_inputs =
        TensorShape({static_cast<int64_t>(num_combined),
                     static_cast<int64_t>(inputs_max_length)});
    auto output_shape_targets =
        TensorShape({static_cast<int64_t>(num_combined),
                     static_cast<int64_t>(targets_max_length)});

    Tensor* inputs_packed;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, output_shape_inputs, &inputs_packed));
    auto inputs_packed_m = inputs_packed->matrix<int64_t>();
    inputs_packed_m.setZero();

    Tensor* inputs_segmentation;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            1, output_shape_inputs, &inputs_segmentation));
    auto inputs_segmentation_m = inputs_segmentation->matrix<int32_t>();
    inputs_segmentation_m.setZero();

    Tensor* inputs_position;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, output_shape_inputs, &inputs_position));
    auto inputs_position_m = inputs_position->matrix<int32_t>();
    inputs_position_m.setZero();

    Tensor* targets_packed;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        3, output_shape_targets, &targets_packed));
    auto targets_packed_m = targets_packed->matrix<int64_t>();
    targets_packed_m.setZero();

    Tensor* targets_segmentation;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            4, output_shape_targets, &targets_segmentation));
    auto targets_segmentation_m = targets_segmentation->matrix<int32_t>();
    targets_segmentation_m.setZero();

    Tensor* targets_position;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(5, output_shape_targets, &targets_position));
    auto targets_position_m = targets_position->matrix<int32_t>();
    targets_position_m.setZero();

    // Copy the actual sequences from 'inputs' and 'targets' into the
    // packed/combined examples:
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

REGISTER_OP("PackSequencesK")
    .Input("inputs: Tinput_types")
    .Input("max_lengths: Tinput_count * int32")
    .Attr("Tinput_types: list(type)")
    .Attr("Tinput_count: int")
    .Output("outputs_packed: Tinput_types")
    .Output("outputs_segmentation: Tinput_count * int32")
    .Output("outputs_position: Tinput_count * int32")
    .SetShapeFn([](InferenceContext* ctx) {
      DataTypeVector input_types;
      int input_count;
      TF_RETURN_IF_ERROR(ctx->GetAttr("Tinput_types", &input_types));
      TF_RETURN_IF_ERROR(ctx->GetAttr("Tinput_count", &input_count));
      if (input_types.size() != input_count) {
        return InvalidArgument(
            "`inputs` and `max_lengths` had different numbers of elements");
      }
      std::vector<ShapeHandle> input_shapes;
      TF_RETURN_IF_ERROR(ctx->input("inputs", &input_shapes));
      std::vector<ShapeHandle> output_shapes;
      std::vector<ShapeHandle> segmentation_shapes;
      std::vector<ShapeHandle> position_shapes;
      for (int i = 0; i < input_shapes.size(); i++) {
        const auto& input_shape = input_shapes.at(i);
        int rank = ctx->Rank(input_shape);
        segmentation_shapes.push_back(
            ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
        position_shapes.push_back(
            ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
        if (rank == 2) {
          output_shapes.push_back(
              ctx->MakeShape({ctx->UnknownDim(), ctx->UnknownDim()}));
        } else if (rank == 3) {
          output_shapes.push_back(
              ctx->MakeShape({ctx->UnknownDim(), ctx->UnknownDim(),
                              ctx->Value(ctx->Dim(input_shape, 2))}));
        } else {
          return InvalidArgument(
              "Only rank 2 and rank 3 inputs are supported");
        }
      }
      TF_RETURN_IF_ERROR(ctx->set_output("outputs_packed", output_shapes));
      TF_RETURN_IF_ERROR(
          ctx->set_output("outputs_segmentation", segmentation_shapes));
      TF_RETURN_IF_ERROR(ctx->set_output("outputs_position", position_shapes));
      return tensorflow::Status();
    });

typedef int InputIndex;
typedef int BatchIndex;
typedef int SeqIndex;

struct PackingSpec {
  SeqIndex seq_id;
  BatchIndex batch_pos;
  int seq_length;
  int offset;
  int segment_id;
};

// This op generalizes PackSequences2Op to examples that contain an arbitrary
// number of sequences (rather than assuming there are just inputs and targets).
// The packing logic is the same.
class PackSequencesKOp : public OpKernel {
 public:
  explicit PackSequencesKOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tinput_types", &input_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tinput_count", &input_count_));
    OP_REQUIRES(
        ctx, input_types_.size() == input_count_,
        InvalidArgument(
            "`inputs` and `max_lengths` had different numbers of elements"));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    OpInputList max_lengths_list;

    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input_list("max_lengths", &max_lengths_list));
    OP_REQUIRES(
        ctx, inputs.size() == max_lengths_list.size(),
        InvalidArgument(
            "`inputs` and `max_lengths` had different numbers of elements"));

    std::map<InputIndex, int> max_lengths;
    for (InputIndex i = 0; i < max_lengths_list.size(); i++) {
      max_lengths[i] = max_lengths_list[i].scalar<int32_t>()();
    }

    int n = inputs.begin()->dim_size(0);
    for (const auto& input : inputs) {
      OP_REQUIRES(ctx, input.dim_size(0) == n,
                  InvalidArgument("`inputs` had different batch sizes"));
    }

    std::map<InputIndex, int> padded_inputs_lengths;
    for (InputIndex i = 0; i < inputs.size(); i++) {
      padded_inputs_lengths[i] =
          std::min(static_cast<int>(inputs[i].dim_size(1)), max_lengths[i]);
    }

    std::map<InputIndex, std::vector<int>> inputs_lengths;
    for (InputIndex i = 0; i < inputs.size(); i++) {
      inputs_lengths[i] =
          GetInputLengths(ctx, inputs[i], padded_inputs_lengths[i]);
    }

    int num_combined = 0;
    std::map<InputIndex, std::map<BatchIndex, int>> combined_inputs_lengths;
    std::map<InputIndex, std::map<SeqIndex, PackingSpec>> packing_specs;
    std::map<BatchIndex, int> segment_counter;

    for (SeqIndex seq_id = 0; seq_id < n; seq_id++) {
      for (BatchIndex b = std::max(0, num_combined - 1000); b < n; b++) {
        bool enough_room = true;
        for (InputIndex i = 0; i < inputs.size(); i++) {
          int cur_seq_len = combined_inputs_lengths[i][b];
          if (cur_seq_len + inputs_lengths[i][seq_id] > max_lengths[i]) {
            enough_room = false;
            break;
          }
        }
        if (enough_room) {
          num_combined = std::max(num_combined, b + 1);
          for (InputIndex i = 0; i < inputs.size(); i++) {
            packing_specs[i][seq_id] = {
              .seq_id = seq_id,
              .batch_pos = b,
              .seq_length = inputs_lengths[i][seq_id],
              .offset = combined_inputs_lengths[i][b],
              .segment_id = (segment_counter[b] + 1)  // Add 1 because zero=pad
            };
            combined_inputs_lengths[i][b] += inputs_lengths[i][seq_id];
          }
          segment_counter[b]++;
          break;
        }
      }
      for (InputIndex i = 0; i < inputs.size(); i++) {
        if (packing_specs[i].find(seq_id) == packing_specs[i].end()) {
          ctx->CtxFailure(InvalidArgument(tensorflow::strings::StrCat(
              "failed to pack example=", seq_id, " into input=", i)));
        }
      }
    }

    OpOutputList outputs_packed;
    OpOutputList outputs_segmentation;
    OpOutputList outputs_position;

    OP_REQUIRES_OK(
        ctx, ctx->output_list("outputs_packed", &outputs_packed));
    OP_REQUIRES_OK(
        ctx, ctx->output_list("outputs_segmentation", &outputs_segmentation));
    OP_REQUIRES_OK(
        ctx, ctx->output_list("outputs_position", &outputs_position));

    for (InputIndex i = 0; i < inputs.size(); i++) {
      TensorShape output_shape_2d = {static_cast<int64_t>(num_combined),
                                     static_cast<int64_t>(max_lengths[i])};

      TensorShape output_shape = output_shape_2d;
      if (inputs[i].dims() == 3) {
        output_shape.AddDim(inputs[i].dim_size(2));
      } else if (inputs[i].dims() != 2) {
        ctx->CtxFailure(InvalidArgument("invalid rank"));
      }

      Tensor* packed;
      Tensor* segmentation;
      Tensor* position;

      OP_REQUIRES_OK(ctx, outputs_packed.allocate(i, output_shape, &packed));
      OP_REQUIRES_OK(ctx, outputs_segmentation.allocate(i, output_shape_2d,
                                                        &segmentation));
      OP_REQUIRES_OK(ctx,
                     outputs_position.allocate(i, output_shape_2d, &position));

      auto segmentation_eigen = segmentation->matrix<int32_t>();
      auto position_eigen = position->matrix<int32_t>();

      SetZero(ctx, packed);
      segmentation_eigen.setZero();
      position_eigen.setZero();

      for (const auto& pair : packing_specs.at(i)) {
        PackSequence(ctx, inputs[i], packed, segmentation_eigen,
                     position_eigen, pair.second);
      }
    }
  }

 private:
  std::vector<int> GetInputLengths(
      OpKernelContext* ctx,
      const Tensor& input,
      const int padded_input_length) {
    switch (input.dtype()) {
      case tensorflow::DT_BFLOAT16:
        return GetInputLengths<bfloat16>(ctx, input, padded_input_length);
      case tensorflow::DT_FLOAT:
        return GetInputLengths<float>(ctx, input, padded_input_length);
      case tensorflow::DT_INT32:
        return GetInputLengths<int32_t>(ctx, input, padded_input_length);
      case tensorflow::DT_INT64:
        return GetInputLengths<int64_t>(ctx, input, padded_input_length);
      default:
        ctx->CtxFailure(
            tensorflow::errors::InvalidArgument("unsupported input dtype"));
        return {};
    }
  }

  template <typename T>
  std::vector<int> GetInputLengths(
      OpKernelContext* ctx,
      const Tensor& input,
      const int padded_input_length) {
    if (input.dims() == 2) {
      return GetInputLengths<const T>(
          input.tensor<T, 2>(), padded_input_length);
    } else if (input.dims() == 3) {
      return GetInputLengths<const T>(
          input.tensor<T, 3>(), padded_input_length);
    } else {
      ctx->CtxFailure(
          tensorflow::errors::InvalidArgument("unsupported input rank"));
      return {};
    }
  }

  template <typename T>
  std::vector<int> GetInputLengths(
      const typename TTypes<T, 2>::Tensor& input,
      const int padded_input_length) {
    std::vector<int> input_lengths;
    for (int i = 0; i < input.dimension(0); i++) {
      int input_length = 0;
      for (int j = 0; j < padded_input_length; j++) {
        if (input(i, j) != 0) {
          input_length++;
        }
      }
      input_lengths.push_back(input_length);
    }
    return input_lengths;
  }

  template <typename T>
  std::vector<int> GetInputLengths(
      const typename TTypes<T, 3>::Tensor& input,
      const int padded_input_length) {
    std::vector<int> input_lengths;
    for (int i = 0; i < input.dimension(0); i++) {
      int input_length = 0;
      for (int j = 0; j < padded_input_length; j++) {
        for (int k = 0; k < input.dimension(2); k++) {
          if (input(i, j, k) != 0) {
            input_length++;
            break;
          }
        }
      }
      input_lengths.push_back(input_length);
    }
    return input_lengths;
  }

  void SetZero(OpKernelContext* ctx, Tensor* inputs) {
    switch (inputs->dtype()) {
      case tensorflow::DT_BFLOAT16:
        SetZero<bfloat16>(ctx, inputs);
        break;
      case tensorflow::DT_FLOAT:
        SetZero<float>(ctx, inputs);
        break;
      case tensorflow::DT_INT32:
        SetZero<int32_t>(ctx, inputs);
        break;
      case tensorflow::DT_INT64:
        SetZero<int64_t>(ctx, inputs);
        break;
      default:
        ctx->CtxFailure(
            tensorflow::errors::InvalidArgument("unsupported input dtype"));
    }
  }

  template <typename T>
  void SetZero(OpKernelContext* ctx, Tensor* inputs) {
    switch (inputs->dims()) {
      case 2:
        inputs->tensor<T, 2>().setZero();
        break;
      case 3:
        inputs->tensor<T, 3>().setZero();
        break;
      default:
        ctx->CtxFailure(
            tensorflow::errors::InvalidArgument("unsupported input rank"));
    }
  }

  void PackSequence(OpKernelContext* ctx, const Tensor& inputs, Tensor* packed,
                    TTypes<int32_t, 2>::Tensor segmentation,
                    TTypes<int32_t, 2>::Tensor position,
                    const PackingSpec& spec) {
    switch (inputs.dtype()) {
      case tensorflow::DT_FLOAT:
        PackSequence<float>(
            ctx, inputs, packed, segmentation, position, spec);
        break;
      case tensorflow::DT_BFLOAT16:
        PackSequence<bfloat16>(
            ctx, inputs, packed, segmentation, position, spec);
        break;
      case tensorflow::DT_INT32:
        PackSequence<int32_t>(ctx, inputs, packed, segmentation, position,
                              spec);
        break;
      case tensorflow::DT_INT64:
        PackSequence<int64_t>(ctx, inputs, packed, segmentation, position,
                              spec);
        break;
      default:
        ctx->CtxFailure(
            tensorflow::errors::InvalidArgument("unsupported input dtype"));
    }
  }

  template <typename T>
  void PackSequence(OpKernelContext* ctx, const Tensor& inputs, Tensor* packed,
                    TTypes<int32_t, 2>::Tensor segmentation,
                    TTypes<int32_t, 2>::Tensor position,
                    const PackingSpec& spec) {
    switch (inputs.dims()) {
      case 2:
        PackSequence<T>(
            ctx,
            inputs.tensor<T, 2>(),
            packed->tensor<T, 2>(),  // TensorMap is pass-by-ref.
            segmentation,
            position,
            spec);
        break;
      case 3:
        PackSequence<T>(
            ctx,
            inputs.tensor<T, 3>(),
            packed->tensor<T, 3>(),  // TensorMap is pass-by-ref.
            segmentation,
            position,
            spec);
        break;
      default:
        ctx->CtxFailure(
            tensorflow::errors::InvalidArgument("unsupported input rank"));
    }
  }

  template <typename T>
  void PackSequence(OpKernelContext* ctx,
                    const typename TTypes<const T, 2>::Tensor& inputs,
                    typename TTypes<T, 2>::Tensor packed,
                    TTypes<int32_t, 2>::Tensor segmentation,
                    TTypes<int32_t, 2>::Tensor position,
                    const PackingSpec& spec) {
    for (int i = 0; i < spec.seq_length; i++) {
      packed(spec.batch_pos, spec.offset + i) = inputs(spec.seq_id, i);
      segmentation(spec.batch_pos, spec.offset + i) = spec.segment_id;
      position(spec.batch_pos, spec.offset + i) = i;
    }
  }

  template <typename T>
  void PackSequence(OpKernelContext* ctx,
                    const typename TTypes<const T, 3>::Tensor& inputs,
                    typename TTypes<T, 3>::Tensor packed,
                    TTypes<int32_t, 2>::Tensor segmentation,
                    TTypes<int32_t, 2>::Tensor position,
                    const PackingSpec& spec) {
    for (int i = 0; i < spec.seq_length; i++) {
      for (int k = 0; k < inputs.dimension(2); k++) {
        packed(spec.batch_pos, spec.offset + i, k) = inputs(spec.seq_id, i, k);
      }
      segmentation(spec.batch_pos, spec.offset + i) = spec.segment_id;
      position(spec.batch_pos, spec.offset + i) = i;
    }
  }

  DataTypeVector input_types_;
  int input_count_;
};

REGISTER_KERNEL_BUILDER(Name("PackSequences2").Device(DEVICE_CPU),
                        PackSequences2Op);

REGISTER_KERNEL_BUILDER(Name("PackSequencesK").Device(DEVICE_CPU),
                        PackSequencesKOp);

}  // namespace
}  // namespace tensor2tensor
