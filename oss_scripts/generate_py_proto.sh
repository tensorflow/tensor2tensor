#!/bin/bash

# This script use the protoc compiler to generate the python code of the
# all of our proto files.


# Function to prepend a pylint directive to skip the generated python file.
function pylint_skip_file() {
  local file_name=$1
  printf "%s\n%s" "# pylint: skip-file" "$(cat ${file_name})" > ${file_name}
}


# Setup tmp directories
TMP_DIR=$(mktemp -d)
TMP_TF_DIR=${TMP_DIR}/tensorflow
TMP_T2T_DIR="$PWD"

echo "Temporary directory created: "
echo ${TMP_DIR}


TMP_T2T_PROTO_DIR="${TMP_T2T_DIR}/tensor2tensor/envs"
ENV_SERVICE_PROTO="${TMP_T2T_PROTO_DIR}/env_service.proto"
if [ ! -f ${ENV_SERVICE_PROTO} ]; then
    echo "${ENV_SERVICE_PROTO} not found."
    echo "Please run this script from the appropriate root directory."
fi

# Clone tensorflow repository.
git clone https://github.com/tensorflow/tensorflow.git ${TMP_TF_DIR}

# Install gRPC tools.
pip install grpcio-tools

# Invoke the grpc protoc compiler on env_service.proto
python -m grpc_tools.protoc \
  --proto_path=${TMP_TF_DIR}/ \
  --proto_path=${TMP_T2T_DIR}/ \
  --python_out=${TMP_T2T_DIR}/ \
  --grpc_python_out=${TMP_T2T_DIR}/ \
  ${ENV_SERVICE_PROTO}

# Add pylint ignore and name the file as generated.
GENERATED_ENV_SERVICE_PY="${TMP_T2T_PROTO_DIR}/env_service_generated_pb2.py"
GENERATED_ENV_SERVICE_GRPC_PY="${TMP_T2T_PROTO_DIR}/env_service_generated_pb2_grpc.py"
mv ${TMP_T2T_PROTO_DIR}/env_service_pb2.py ${GENERATED_ENV_SERVICE_PY}
mv ${TMP_T2T_PROTO_DIR}/env_service_pb2_grpc.py ${GENERATED_ENV_SERVICE_GRPC_PY}
pylint_skip_file "${GENERATED_ENV_SERVICE_PY}"
pylint_skip_file "${GENERATED_ENV_SERVICE_GRPC_PY}"


LICENSING_TEXT=$(cat <<-END
# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
END
)

function add_licensing_text() {
  local file_name=$1
  printf "%s\n%s" "${LICENSING_TEXT}" "$(cat ${file_name})" > ${file_name}
}

add_licensing_text "${GENERATED_ENV_SERVICE_PY}"
add_licensing_text "${GENERATED_ENV_SERVICE_GRPC_PY}"

