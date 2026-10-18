; RUN: mlir-translate -import-llvm %s | FileCheck %s

declare i32 @callee(i32)
declare i32 @personality(...)

; CHECK-LABEL: llvm.func @direct(
; CHECK-SAME: %[[VALUE:.*]]: i32, %[[ADDRESS:.*]]: !llvm.ptr)
; CHECK: llvm.call @callee(%[[VALUE]]) ["values"(%[[VALUE]], %[[ADDRESS]] : i32, !llvm.ptr), "empty"()] : (i32) -> i32
define i32 @direct(i32 %value, ptr %address) {
  %result = call i32 @callee(i32 %value) ["values"(i32 %value, ptr %address), "empty"()]
  ret i32 %result
}

; CHECK-LABEL: llvm.func @indirect(
; CHECK-SAME: %[[CALLEE:.*]]: !llvm.ptr, %[[VALUE:.*]]: i32)
; CHECK: llvm.call %[[CALLEE]](%[[VALUE]]) ["value"(%[[VALUE]] : i32)] : !llvm.ptr, (i32) -> i32
define i32 @indirect(ptr %callee, i32 %value) {
  %result = call i32 %callee(i32 %value) ["value"(i32 %value)]
  ret i32 %result
}

; CHECK-LABEL: llvm.func @unwind(
; CHECK-SAME: %[[VALUE:.*]]: i32, %[[ADDRESS:.*]]: !llvm.ptr)
; CHECK: llvm.invoke @callee(%[[VALUE]]) to ^{{.*}} unwind ^{{.*}} ["empty"(), "values"(%[[ADDRESS]], %[[VALUE]] : !llvm.ptr, i32)] : (i32) -> i32
define i32 @unwind(i32 %value, ptr %address) personality ptr @personality {
  %result = invoke i32 @callee(i32 %value) ["empty"(), "values"(ptr %address, i32 %value)]
      to label %normal unwind label %exception
normal:
  ret i32 %result
exception:
  %landing = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %landing
}
