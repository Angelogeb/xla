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

#include "xla/python/ifrt/ir/ifrt_capi.h"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// Include IFRT dialect - this brings in all the generated types
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/sharding_param.h"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ifrt, ifrt,
                                       xla::ifrt::IfrtDialect)

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

bool ifrtTypeIsAArrayType(MlirType type) {
  return llvm::isa<xla::ifrt::IfrtArrayType>(unwrap(type));
}

bool ifrtTypeIsAControlType(MlirType type) {
  return llvm::isa<xla::ifrt::IfrtControlType>(unwrap(type));
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

MlirType ifrtArrayTypeGet(
    MlirType shape,
    MlirAttribute sharding_attr,
    MlirAttribute devices_attr,
    MlirAttribute memory_kind_attr,
    MlirAttribute layout_attr) {
  auto ranked_tensor_type = llvm::cast<mlir::RankedTensorType>(unwrap(shape));
  auto sharding = llvm::cast<xla::ifrt::IfrtShardingAttrInterface>(unwrap(sharding_attr));
  auto devices = llvm::cast<xla::ifrt::IfrtDevicesAttr>(unwrap(devices_attr));
  
  mlir::StringAttr memory_kind = {};
  if (!mlirAttributeIsNull(memory_kind_attr)) {
    memory_kind = llvm::cast<mlir::StringAttr>(unwrap(memory_kind_attr));
  }
  
  mlir::StringAttr layout = {};
  if (!mlirAttributeIsNull(layout_attr)) {
    layout = llvm::cast<mlir::StringAttr>(unwrap(layout_attr));
  }
  
  return wrap(xla::ifrt::IfrtArrayType::get(
      ranked_tensor_type.getContext(), ranked_tensor_type, sharding, devices, memory_kind, layout));
}

MlirType ifrtArrayTypeGetShape(MlirType type) {
  auto array_type = llvm::cast<xla::ifrt::IfrtArrayType>(unwrap(type));
  return wrap(array_type.getShape());
}

MlirAttribute ifrtArrayTypeGetShardingAttr(MlirType type) {
  auto array_type = llvm::cast<xla::ifrt::IfrtArrayType>(unwrap(type));
  return wrap(array_type.getShardingAttr());
}

MlirAttribute ifrtArrayTypeGetDevicesAttr(MlirType type) {
  auto array_type = llvm::cast<xla::ifrt::IfrtArrayType>(unwrap(type));
  return wrap(array_type.getDevicesAttr());
}

MlirAttribute ifrtArrayTypeGetMemoryKindAttr(MlirType type) {
  auto array_type = llvm::cast<xla::ifrt::IfrtArrayType>(unwrap(type));
  auto memory_kind = array_type.getMemoryKindAttr();
  if (memory_kind) {
    return wrap(static_cast<mlir::Attribute>(memory_kind));
  }
  return {nullptr};
}

MlirAttribute ifrtArrayTypeGetLayoutAttr(MlirType type) {
  auto array_type = llvm::cast<xla::ifrt::IfrtArrayType>(unwrap(type));
  auto layout = array_type.getLayoutAttr();
  if (layout) {
    return wrap(static_cast<mlir::Attribute>(layout));
  }
  return {nullptr};
}

//===----------------------------------------------------------------------===//
// ControlType
//===----------------------------------------------------------------------===//

MlirType ifrtControlTypeGet(MlirContext ctx) {
  return wrap(xla::ifrt::IfrtControlType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Attribute predicates
//===----------------------------------------------------------------------===//

bool ifrtAttributeIsADevicesAttr(MlirAttribute attr) {
  return llvm::isa<xla::ifrt::IfrtDevicesAttr>(unwrap(attr));
}

bool ifrtAttributeIsAUnspecifiedShardingAttr(MlirAttribute attr) {
  return llvm::isa<xla::ifrt::IfrtUnspecifiedShardingAttr>(unwrap(attr));
}

bool ifrtAttributeIsAShardingParamAttr(MlirAttribute attr) {
  return llvm::isa<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// DevicesAttr
//===----------------------------------------------------------------------===//

MlirAttribute ifrtDevicesAttrGet(
    MlirContext ctx,
    intptr_t num_devices,
    const int64_t* devices) {
  llvm::SmallVector<int> device_ids;
  device_ids.reserve(num_devices);
  for (intptr_t i = 0; i < num_devices; ++i) {
    device_ids.push_back(static_cast<int>(devices[i]));
  }
  return wrap(xla::ifrt::IfrtDevicesAttr::get(unwrap(ctx), device_ids));
}

intptr_t ifrtDevicesAttrGetIdsSize(MlirAttribute attr) {
  auto devices_attr = llvm::cast<xla::ifrt::IfrtDevicesAttr>(unwrap(attr));
  return devices_attr.getIds().size();
}

int64_t ifrtDevicesAttrGetIdsElem(MlirAttribute attr, intptr_t pos) {
  auto devices_attr = llvm::cast<xla::ifrt::IfrtDevicesAttr>(unwrap(attr));
  return devices_attr.getIds()[pos];
}

//===----------------------------------------------------------------------===//
// UnspecifiedShardingAttr
//===----------------------------------------------------------------------===//

MlirAttribute ifrtUnspecifiedShardingAttrGet(MlirContext ctx) {
  return wrap(xla::ifrt::IfrtUnspecifiedShardingAttr::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// ShardingParamAttr
//===----------------------------------------------------------------------===//

MlirAttribute ifrtShardingParamAttrGet(
    MlirContext ctx,
    intptr_t num_dim_shards,
    const int64_t* dim_shards,
    intptr_t num_permutation,
    const int* permutation,
    intptr_t num_axis_sizes,
    const int* axis_sizes) {
  std::vector<int64_t> dim_shards_vec(dim_shards, dim_shards + num_dim_shards);
  
  xla::ifrt::ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation.assign(permutation, permutation + num_permutation);
  minor_to_major.axis_sizes.assign(axis_sizes, axis_sizes + num_axis_sizes);
  
  xla::ifrt::ShardingParam sharding_param(std::move(dim_shards_vec), std::move(minor_to_major));
  
  return wrap(xla::ifrt::IfrtShardingParamAttr::get(unwrap(ctx), sharding_param));
}

intptr_t ifrtShardingParamAttrGetDimShardsSize(MlirAttribute attr) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().dim_shards().size();
}

int64_t ifrtShardingParamAttrGetDimShardsElem(MlirAttribute attr, intptr_t pos) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().dim_shards()[pos];
}

intptr_t ifrtShardingParamAttrGetPermutationSize(MlirAttribute attr) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().minor_to_major().permutation.size();
}

int ifrtShardingParamAttrGetPermutationElem(MlirAttribute attr, intptr_t pos) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().minor_to_major().permutation[pos];
}

intptr_t ifrtShardingParamAttrGetAxisSizesSize(MlirAttribute attr) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().minor_to_major().axis_sizes.size();
}

int ifrtShardingParamAttrGetAxisSizesElem(MlirAttribute attr, intptr_t pos) {
  auto sharding_param_attr = llvm::cast<xla::ifrt::IfrtShardingParamAttr>(unwrap(attr));
  return sharding_param_attr.getSharding().minor_to_major().axis_sizes[pos];
}

