/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_IFRT_IR_IFRT_CAPI_H_
#define XLA_PYTHON_IFRT_IR_IFRT_CAPI_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ifrt, ifrt);

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

/** Returns `true` if the given type is an IFRT ArrayType. */
MLIR_CAPI_EXPORTED bool ifrtTypeIsAArrayType(MlirType type);

/** Returns `true` if the given type is an IFRT ControlType. */
MLIR_CAPI_EXPORTED bool ifrtTypeIsAControlType(MlirType type);

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/** Creates an IFRT ArrayType. */
MLIR_CAPI_EXPORTED MlirType ifrtArrayTypeGet(
    MlirType shape,
    MlirAttribute sharding_attr,
    MlirAttribute devices_attr,
    MlirAttribute memory_kind_attr,
    MlirAttribute layout_attr);

/** Returns the shape (RankedTensorType) of an IFRT ArrayType. */
MLIR_CAPI_EXPORTED MlirType ifrtArrayTypeGetShape(MlirType type);

/** Returns the sharding attribute of an IFRT ArrayType. */
MLIR_CAPI_EXPORTED MlirAttribute ifrtArrayTypeGetShardingAttr(MlirType type);

/** Returns the devices attribute of an IFRT ArrayType. */
MLIR_CAPI_EXPORTED MlirAttribute ifrtArrayTypeGetDevicesAttr(MlirType type);

/** Returns the memory kind attribute of an IFRT ArrayType (may be null). */
MLIR_CAPI_EXPORTED MlirAttribute ifrtArrayTypeGetMemoryKindAttr(MlirType type);

/** Returns the layout attribute of an IFRT ArrayType (may be null). */
MLIR_CAPI_EXPORTED MlirAttribute ifrtArrayTypeGetLayoutAttr(MlirType type);

//===----------------------------------------------------------------------===//
// ControlType
//===----------------------------------------------------------------------===//

/** Creates an IFRT ControlType. */
MLIR_CAPI_EXPORTED MlirType ifrtControlTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Attribute predicates
//===----------------------------------------------------------------------===//

/** Returns `true` if the given attribute is an IFRT DevicesAttr. */
MLIR_CAPI_EXPORTED bool ifrtAttributeIsADevicesAttr(MlirAttribute attr);

/** Returns `true` if the given attribute is an IFRT UnspecifiedShardingAttr. */
MLIR_CAPI_EXPORTED bool ifrtAttributeIsAUnspecifiedShardingAttr(MlirAttribute attr);

/** Returns `true` if the given attribute is an IFRT ShardingParamAttr. */
MLIR_CAPI_EXPORTED bool ifrtAttributeIsAShardingParamAttr(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DevicesAttr
//===----------------------------------------------------------------------===//

/** Creates an IFRT DevicesAttr from a list of device IDs. */
MLIR_CAPI_EXPORTED MlirAttribute ifrtDevicesAttrGet(
    MlirContext ctx,
    intptr_t num_devices,
    const int64_t* devices);

/** Returns the number of devices in an IFRT DevicesAttr. */
MLIR_CAPI_EXPORTED intptr_t ifrtDevicesAttrGetIdsSize(MlirAttribute attr);

/** Returns the device ID at the given index in an IFRT DevicesAttr. */
MLIR_CAPI_EXPORTED int64_t ifrtDevicesAttrGetIdsElem(
    MlirAttribute attr,
    intptr_t pos);

//===----------------------------------------------------------------------===//
// UnspecifiedShardingAttr
//===----------------------------------------------------------------------===//

/** Creates an IFRT UnspecifiedShardingAttr. */
MLIR_CAPI_EXPORTED MlirAttribute ifrtUnspecifiedShardingAttrGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// ShardingParamAttr
//===----------------------------------------------------------------------===//

/** Creates an IFRT ShardingParamAttr from dim_shards, permutation, and axis_sizes. */
MLIR_CAPI_EXPORTED MlirAttribute ifrtShardingParamAttrGet(
    MlirContext ctx,
    intptr_t num_dim_shards,
    const int64_t* dim_shards,
    intptr_t num_permutation,
    const int* permutation,
    intptr_t num_axis_sizes,
    const int* axis_sizes);

/** Returns the number of dim_shards in a ShardingParamAttr. */
MLIR_CAPI_EXPORTED intptr_t ifrtShardingParamAttrGetDimShardsSize(MlirAttribute attr);

/** Returns a dim_shard value at the given index. */
MLIR_CAPI_EXPORTED int64_t ifrtShardingParamAttrGetDimShardsElem(
    MlirAttribute attr,
    intptr_t pos);

/** Returns the number of elements in the permutation. */
MLIR_CAPI_EXPORTED intptr_t ifrtShardingParamAttrGetPermutationSize(MlirAttribute attr);

/** Returns a permutation value at the given index. */
MLIR_CAPI_EXPORTED int ifrtShardingParamAttrGetPermutationElem(
    MlirAttribute attr,
    intptr_t pos);

/** Returns the number of axis sizes. */
MLIR_CAPI_EXPORTED intptr_t ifrtShardingParamAttrGetAxisSizesSize(MlirAttribute attr);

/** Returns an axis size value at the given index. */
MLIR_CAPI_EXPORTED int ifrtShardingParamAttrGetAxisSizesElem(
    MlirAttribute attr,
    intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PYTHON_IFRT_IR_IFRT_CAPI_H_

