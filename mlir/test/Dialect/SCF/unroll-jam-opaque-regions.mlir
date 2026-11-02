// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// A loop inside an opaque operation is cloned with that operation. Its
// carried arguments must not be widened or pre-mapped as jammed arguments.
// CHECK-LABEL: func.func @scf_conditional_reduction
// CHECK: scf.for %[[I:.*]] =
// CHECK: arith.cmpi ult, %[[I]],
// CHECK: %[[FIRST:.*]] = scf.if
// CHECK: %[[FIRST_SUM:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK: scf.yield %[[FIRST_SUM]] : i32
// CHECK: memref.store %[[FIRST]],
// CHECK: %[[NEXT:.*]] = arith.addi %[[I]],
// CHECK: arith.cmpi ult, %[[NEXT]],
// CHECK: %[[SECOND:.*]] = scf.if
// CHECK: %[[SECOND_SUM:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK: scf.yield %[[SECOND_SUM]] : i32
// CHECK: memref.store %[[SECOND]],
func.func @scf_conditional_reduction(%input: memref<4x4xi32>, %output: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  scf.for %i = %c0 to %c4 step %c1 {
    %condition = arith.cmpi ult, %i, %c1 : index
    %result = scf.if %condition -> i32 {
      %sum = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> i32 {
        %value = memref.load %input[%i, %j] : memref<4x4xi32>
        %next = arith.addi %acc, %value : i32
        scf.yield %next : i32
      }
      scf.yield %sum : i32
    } else {
      scf.yield %zero : i32
    }
    memref.store %result, %output[%i] : memref<4xi32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %store = transform.structured.match ops{["memref.store"]} in %root : (!transform.any_op) -> !transform.any_op
    %outer = transform.get_parent_op %store {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %outer {factor = 2} : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @affine_conditional_reduction
// CHECK: affine.for %[[I:.*]] = 0 to 4 step 2
// CHECK: arith.cmpi ult, %[[I]],
// CHECK: %[[FIRST:.*]] = scf.if
// CHECK: %[[FIRST_SUM:.*]] = affine.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK: scf.yield %[[FIRST_SUM]] : i32
// CHECK: memref.store %[[FIRST]],
// CHECK: %[[NEXT:.*]] = affine.apply {{.*}}(%[[I]])
// CHECK: arith.cmpi ult, %[[NEXT]],
// CHECK: %[[SECOND:.*]] = scf.if
// CHECK: %[[SECOND_SUM:.*]] = affine.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK: scf.yield %[[SECOND_SUM]] : i32
// CHECK: memref.store %[[SECOND]],
func.func @affine_conditional_reduction(%input: memref<4x4xi32>, %output: memref<4xi32>) {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    %condition = arith.cmpi ult, %i, %c1 : index
    %result = scf.if %condition -> i32 {
      %sum = affine.for %j = 0 to 4 iter_args(%acc = %zero) -> i32 {
        %value = memref.load %input[%i, %j] : memref<4x4xi32>
        %next = arith.addi %acc, %value : i32
        affine.yield %next : i32
      }
      scf.yield %sum : i32
    } else {
      scf.yield %zero : i32
    }
    memref.store %result, %output[%i] : memref<4xi32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %store = transform.structured.match ops{["memref.store"]} in %root : (!transform.any_op) -> !transform.any_op
    %outer = transform.get_parent_op %store {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    transform.loop.unroll_and_jam %outer {factor = 2} : !transform.op<"affine.for">
    transform.yield
  }
}
