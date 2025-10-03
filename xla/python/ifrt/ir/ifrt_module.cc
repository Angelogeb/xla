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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"    // IWYU pragma: keep
#include "nanobind/stl/vector.h"    // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/ir/ifrt_capi.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/support/sharding_conversions.h"

namespace xla {
namespace ifrt {

namespace {

namespace nb = nanobind;

// Returns a vector containing elements with type T extracted from an attribute
// using the two provided callbacks.
template <typename T>
std::vector<T> propertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<T(MlirAttribute, intptr_t)> getFn) {
  std::vector<T> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

NB_MODULE(_ifrt, m) {
  m.doc() = "IFRT IR Python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__ifrt__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Types.
  //

  mlir::python::nanobind_adaptors::mlir_type_subclass(
      m, "ArrayType", ifrtTypeIsAArrayType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType shape, MlirAttribute sharding_attr,
             MlirAttribute devices_attr,
             std::optional<MlirAttribute> memory_kind_attr,
             std::optional<MlirAttribute> layout_attr) {
            MlirAttribute memory_kind = memory_kind_attr.has_value()
                                            ? *memory_kind_attr
                                            : MlirAttribute{nullptr};
            MlirAttribute layout =
                layout_attr.has_value() ? *layout_attr : MlirAttribute{nullptr};
            return cls(ifrtArrayTypeGet(shape, sharding_attr, devices_attr,
                                        memory_kind, layout));
          },
          nb::arg("cls"), nb::arg("shape"), nb::arg("sharding_attr"),
          nb::arg("devices_attr"),
          nb::arg("memory_kind_attr").none() = nb::none(),
          nb::arg("layout_attr").none() = nb::none(),
          "Creates an IFRT ArrayType with the given shape, sharding, devices, "
          "and optional memory kind and layout.")
      .def_property_readonly("shape",
                             [](MlirType self) {
                               return ifrtArrayTypeGetShape(self);
                             })
      .def_property_readonly("sharding_attr",
                             [](MlirType self) {
                               return ifrtArrayTypeGetShardingAttr(self);
                             })
      .def_property_readonly("devices_attr",
                             [](MlirType self) {
                               return ifrtArrayTypeGetDevicesAttr(self);
                             })
      .def_property_readonly("memory_kind_attr",
                             [](MlirType self) {
                               MlirAttribute attr =
                                   ifrtArrayTypeGetMemoryKindAttr(self);
                               return attr.ptr == nullptr
                                          ? std::nullopt
                                          : std::optional(attr);
                             })
      .def_property_readonly("layout_attr", [](MlirType self) {
        MlirAttribute attr = ifrtArrayTypeGetLayoutAttr(self);
        return attr.ptr == nullptr ? std::nullopt : std::optional(attr);
      });

  mlir::python::nanobind_adaptors::mlir_type_subclass(
      m, "ControlType", ifrtTypeIsAControlType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx) {
            return cls(ifrtControlTypeGet(ctx));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          "Creates an IFRT ControlType.");

  //
  // Attributes.
  //

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "DevicesAttr", ifrtAttributeIsADevicesAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t>& devices,
             MlirContext ctx) {
            return cls(
                ifrtDevicesAttrGet(ctx, devices.size(), devices.data()));
          },
          nb::arg("cls"), nb::arg("devices"),
          nb::arg("context").none() = nb::none(),
          "Creates an IFRT DevicesAttr with the given device IDs.")
      .def_property_readonly("ids", [](MlirAttribute self) {
        return propertyVector<int64_t>(self, ifrtDevicesAttrGetIdsSize,
                                       ifrtDevicesAttrGetIdsElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "UnspecifiedShardingAttr", ifrtAttributeIsAUnspecifiedShardingAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx) {
            return cls(ifrtUnspecifiedShardingAttrGet(ctx));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          "Creates an IFRT UnspecifiedShardingAttr.");

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ShardingParamAttr", ifrtAttributeIsAShardingParamAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t>& dim_shards,
             const std::vector<int>& permutation,
             const std::vector<int>& axis_sizes, MlirContext ctx) {
            return cls(ifrtShardingParamAttrGet(
                ctx, dim_shards.size(), dim_shards.data(),
                permutation.size(), permutation.data(), axis_sizes.size(),
                axis_sizes.data()));
          },
          nb::arg("cls"), nb::arg("dim_shards"), nb::arg("permutation"),
          nb::arg("axis_sizes"), nb::arg("context").none() = nb::none(),
          "Creates an IFRT ShardingParamAttr from dim_shards, permutation, and "
          "axis_sizes.")
      .def_property_readonly("dim_shards",
                             [](MlirAttribute self) {
                               return propertyVector<int64_t>(
                                   self, ifrtShardingParamAttrGetDimShardsSize,
                                   ifrtShardingParamAttrGetDimShardsElem);
                             })
      .def_property_readonly("permutation",
                             [](MlirAttribute self) {
                               return propertyVector<int>(
                                   self, ifrtShardingParamAttrGetPermutationSize,
                                   ifrtShardingParamAttrGetPermutationElem);
                             })
      .def_property_readonly("axis_sizes", [](MlirAttribute self) {
        return propertyVector<int>(
            self, ifrtShardingParamAttrGetAxisSizesSize,
            ifrtShardingParamAttrGetAxisSizesElem);
      });

  //
  // ShardingParam C++ class bindings
  //

  nb::class_<ShardingParam::MinorToMajor>(m, "MinorToMajor")
      .def(nb::init<>())
      .def_prop_rw(
          "permutation",
          [](const ShardingParam::MinorToMajor& mtm) {
            return std::vector<int>(mtm.permutation.begin(),
                                    mtm.permutation.end());
          },
          [](ShardingParam::MinorToMajor& mtm, const std::vector<int>& perm) {
            mtm.permutation.clear();
            mtm.permutation.append(perm.begin(), perm.end());
          },
          "A permutation of range [0...n]")
      .def_prop_rw(
          "axis_sizes",
          [](const ShardingParam::MinorToMajor& mtm) {
            return std::vector<int>(mtm.axis_sizes.begin(),
                                    mtm.axis_sizes.end());
          },
          [](ShardingParam::MinorToMajor& mtm, const std::vector<int>& sizes) {
            mtm.axis_sizes.clear();
            mtm.axis_sizes.append(sizes.begin(), sizes.end());
          },
          "The size of mesh dimensions before the permutation")
      .def("__eq__", &ShardingParam::MinorToMajor::operator==)
      .def("__repr__", [](const ShardingParam::MinorToMajor& mtm) {
        std::string result = "MinorToMajor(permutation=[";
        for (size_t i = 0; i < mtm.permutation.size(); ++i) {
          if (i > 0) result += ", ";
          result += std::to_string(mtm.permutation[i]);
        }
        result += "], axis_sizes=[";
        for (size_t i = 0; i < mtm.axis_sizes.size(); ++i) {
          if (i > 0) result += ", ";
          result += std::to_string(mtm.axis_sizes[i]);
        }
        result += "])";
        return result;
      });

  nb::class_<ShardingParam>(m, "ShardingParam")
      .def(nb::init<std::vector<int64_t>, ShardingParam::MinorToMajor>(),
           nb::arg("dim_shards"), nb::arg("minor_to_major"),
           "Constructs a ShardingParam from dim_shards and minor_to_major")
      .def_prop_ro(
          "dim_shards",
          [](const ShardingParam& sp) {
            return std::vector<int64_t>(sp.dim_shards().begin(),
                                        sp.dim_shards().end());
          },
          "Sharding dimensions")
      .def_prop_ro("minor_to_major", &ShardingParam::minor_to_major,
                   "Minor to major device ordering")
      .def("num_devices", &ShardingParam::NumDevices,
           "Returns the number of devices the array is sharded over")
      .def("debug_string", &ShardingParam::DebugString,
           "Returns a debug string representation")
      .def("__eq__", &ShardingParam::operator==)
      .def("__ne__", &ShardingParam::operator!=)
      .def("__repr__",
           [](const ShardingParam& sp) { return sp.DebugString(); });

  // ToShardingParam function binding
  m.def(
      "to_sharding_param",
      [](const xla::HloSharding& hlo_sharding, int rank,
         int num_devices) -> ShardingParam {
        return xla::ValueOrThrow(
            support::ToShardingParam(hlo_sharding, rank, num_devices));
      },
      nb::arg("hlo_sharding"), nb::arg("rank"), nb::arg("num_devices"),
      "Converts HloSharding to ShardingParam");

  // ToHloSharding function binding
  m.def(
      "to_hlo_sharding",
      [](const ShardingParam& sharding_param) -> xla::HloSharding {
        return xla::ValueOrThrow(support::ToHloSharding(sharding_param));
      },
      nb::arg("sharding_param"), "Converts ShardingParam to HloSharding");
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

