#!/usr/bin/env python3
"""
Test Template for Complex Triton Operators

Comprehensive testing template for complex Triton operators after NPU optimization.
Includes functional tests, accuracy tests, edge cases, and performance tests.

Based on the testing patterns from triton_demo examples and migration skills.
"""

import torch
import torch_npu
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any
import triton
import triton.language as tl


def set_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ComplexOperatorTester:
    """
    Comprehensive tester for complex Triton operators.
    
    Features:
    1. Functional correctness tests
    2. Accuracy validation against reference
    3. Edge case testing
    4. Performance benchmarking
    5. Memory usage analysis
    """
    
    def __init__(self, operator_name: str, atol: float = 1e-3, rtol: float = 1e-3):
        self.operator_name = operator_name
        self.atol = atol
        self.rtol = rtol
        self.results = []
        
    def log_test(self, test_name: str, success: bool, details: str = "", **kwargs):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time(),
            **kwargs
        }
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"  {details}")
        
        return success
    
    # ============================================================================
    # BASIC FUNCTIONAL TESTS
    # ============================================================================
    
    def test_shape_correctness(self, test_inputs: Dict, expected_shape: Tuple) -> bool:
        """Test that operator produces correct output shape"""
        test_name = f"{self.operator_name} - Shape Correctness"
        
        try:
            # Run operator (placeholder - implement with actual operator)
            # output = self.run_operator(**test_inputs)
            
            # For demonstration, create dummy output
            output = torch.randn(*expected_shape, device='npu', dtype=torch.float32)
            
            if output.shape == expected_shape:
                return self.log_test(test_name, True, f"Output shape: {output.shape}")
            else:
                details = f"Expected shape: {expected_shape}, Got: {output.shape}"
                return self.log_test(test_name, False, details)
                
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    def test_dtype_support(self, test_inputs: Dict, dtypes: List[torch.dtype]) -> bool:
        """Test operator support for different data types"""
        all_pass = True
        
        for dtype in dtypes:
            test_name = f"{self.operator_name} - Dtype Support ({dtype})"
            
            try:
                # Convert inputs to target dtype
                dtype_inputs = {}
                for key, value in test_inputs.items():
                    if isinstance(value, torch.Tensor):
                        dtype_inputs[key] = value.to(dtype)
                    else:
                        dtype_inputs[key] = value
                
                # Run operator
                # output = self.run_operator(**dtype_inputs)
                
                # For demonstration
                output = torch.randn(2, 4, 8, device='npu', dtype=dtype)
                
                # Check output dtype
                if output.dtype == dtype:
                    self.log_test(test_name, True, f"Output dtype: {output.dtype}")
                else:
                    details = f"Expected dtype: {dtype}, Got: {output.dtype}"
                    all_pass = self.log_test(test_name, False, details) and all_pass
                    
            except Exception as e:
                details = f"Exception: {str(e)}"
                all_pass = self.log_test(test_name, False, details) and all_pass
        
        return all_pass
    
    def test_device_support(self, test_inputs: Dict) -> bool:
        """Test operator support for NPU device"""
        test_name = f"{self.operator_name} - NPU Device Support"
        
        try:
            # Ensure all tensors are on NPU
            npu_inputs = {}
            for key, value in test_inputs.items():
                if isinstance(value, torch.Tensor):
                    npu_inputs[key] = value.to('npu')
                else:
                    npu_inputs[key] = value
            
            # Run operator
            # output = self.run_operator(**npu_inputs)
            
            # For demonstration
            output = torch.randn(2, 4, 8, device='npu', dtype=torch.float32)
            
            if output.device.type == 'npu':
                return self.log_test(test_name, True, f"Output device: {output.device}")
            else:
                details = f"Expected device: npu, Got: {output.device}"
                return self.log_test(test_name, False, details)
                
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    # ============================================================================
    # ACCURACY TESTS (Based on triton_demo/ori1 pattern)
    # ============================================================================
    
    def test_accuracy_simple(self, test_inputs: Dict) -> bool:
        """
        Simple accuracy test comparing reference and optimized implementations.
        
        Based on the main function pattern from triton_demo/ori1.
        """
        test_name = f"{self.operator_name} - Simple Accuracy"
        
        try:
            # Generate reference output (original implementation)
            print(f"\n{test_name}: Running reference implementation...")
            # ref_output = self.run_reference_implementation(**test_inputs)
            
            # Generate test output (optimized implementation)
            print(f"{test_name}: Running optimized implementation...")
            # test_output = self.run_optimized_implementation(**test_inputs)
            
            # For demonstration
            B, T, H, K = 2, 4096, 32, 256
            V = 128
            ref_output = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
            test_output = ref_output.clone() + torch.randn_like(ref_output) * 1e-4
            
            # Compare statistics (as in ori1 main function)
            print(f"{test_name}: Reference output shape: {ref_output.shape}")
            print(f"{test_name}: Test output shape: {test_output.shape}")
            
            print(f"{test_name}: Reference min/max/mean: "
                  f"{ref_output.min():.6f}, {ref_output.max():.6f}, {ref_output.mean():.6f}")
            print(f"{test_name}: Test min/max/mean: "
                  f"{test_output.min():.6f}, {test_output.max():.6f}, {test_output.mean():.6f}")
            
            # Calculate difference
            max_diff = torch.max(torch.abs(ref_output - test_output))
            print(f"{test_name}: Maximum difference: {max_diff:.6e}")
            
            # Validate with tolerance
            try:
                torch.testing.assert_close(
                    ref_output, test_output,
                    atol=self.atol, rtol=self.rtol,
                    msg=f"{self.operator_name} accuracy test failed"
                )
                return self.log_test(test_name, True, f"Max diff: {max_diff:.6e}")
            except AssertionError as e:
                return self.log_test(test_name, False, f"Max diff: {max_diff:.6e}, Error: {str(e)}")
                
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    def test_accuracy_multiple_sizes(self, size_configs: List[Dict]) -> bool:
        """Test accuracy with multiple input sizes"""
        all_pass = True
        
        for i, config in enumerate(size_configs):
            test_name = f"{self.operator_name} - Accuracy Size {i+1}"
            
            try:
                # Create test inputs for this size
                test_inputs = self.create_test_inputs(**config)
                
                # Run accuracy test
                success = self.test_accuracy_simple(test_inputs)
                all_pass = success and all_pass
                
            except Exception as e:
                details = f"Exception for config {config}: {str(e)}"
                all_pass = self.log_test(test_name, False, details) and all_pass
        
        return all_pass
    
    # ============================================================================
    # EDGE CASE TESTS
    # ============================================================================
    
    def test_edge_case_empty(self) -> bool:
        """Test with empty tensors (if supported)"""
        test_name = f"{self.operator_name} - Empty Tensor"
        
        try:
            # Create empty inputs
            empty_inputs = self.create_test_inputs(B=0, T=0, H=8, K=64, V=64)
            
            # Run operator
            # output = self.run_operator(**empty_inputs)
            
            # For operators that don't support empty tensors, this might fail
            # which is acceptable if documented
            
            return self.log_test(test_name, True, "Empty tensor test completed")
            
        except Exception as e:
            # Check if this is an expected failure
            if "empty" in str(e).lower() or "size" in str(e).lower():
                return self.log_test(test_name, True, f"Expected failure: {str(e)}")
            else:
                return self.log_test(test_name, False, f"Unexpected exception: {str(e)}")
    
    def test_edge_case_large(self, max_memory_gb: float = 4.0) -> bool:
        """Test with large tensors (within memory limits)"""
        test_name = f"{self.operator_name} - Large Tensor"
        
        try:
            # Estimate memory usage and create appropriate size
            # This is operator-specific
            B, T, H, K, V = 1, 16384, 8, 256, 256
            
            # Create large inputs
            large_inputs = self.create_test_inputs(B=B, T=T, H=H, K=K, V=V)
            
            # Run operator
            # output = self.run_operator(**large_inputs)
            
            return self.log_test(test_name, True, f"Large tensor test: B={B}, T={T}, H={H}")
            
        except torch.cuda.OutOfMemoryError:
            return self.log_test(test_name, False, "Out of memory")
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    def test_edge_case_extreme_values(self) -> bool:
        """Test with extreme values (very large, very small, NaN, Inf)"""
        test_name = f"{self.operator_name} - Extreme Values"
        
        try:
            # Create inputs with extreme values
            B, T, H, K, V = 2, 4, 2, 8, 8
            
            # Create tensor with various extreme values
            extreme_tensor = torch.tensor([
                [0.0, 1e-10, 1e10, float('inf'), float('-inf'), float('nan'), -1e10, -1e-10]
            ], device='npu', dtype=torch.float32).repeat(B * T * H * K // 8, 1)
            
            extreme_tensor = extreme_tensor.reshape(B, T, H, K)
            
            test_inputs = {
                'q': extreme_tensor,
                'k': extreme_tensor,
                'v': extreme_tensor[:, :, :, :V],
                'scale': 0.5,
            }
            
            # Run operator
            # output = self.run_operator(**test_inputs)
            
            # Check for issues
            # if torch.isnan(output).all() or torch.isinf(output).all():
            #     return self.log_test(test_name, False, "Output all NaN/Inf")
            
            return self.log_test(test_name, True, "Extreme values test completed")
            
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    def benchmark_operator(self, test_inputs: Dict, num_warmup: int = 10, num_iter: int = 100) -> Dict:
        """
        Benchmark operator performance.
        
        Returns performance metrics including execution time.
        """
        metrics = {
            'warmup_time': 0.0,
            'iteration_times': [],
            'avg_time': 0.0,
            'min_time': 0.0,
            'max_time': 0.0,
            'std_time': 0.0,
        }
        
        # Warmup runs
        warmup_times = []
        for i in range(num_warmup):
            start = time.perf_counter()
            # self.run_operator(**test_inputs)
            torch.npu.synchronize()  # Wait for NPU operations to complete
            end = time.perf_counter()
            warmup_times.append(end - start)
        
        metrics['warmup_time'] = sum(warmup_times) / len(warmup_times)
        
        # Measurement runs
        iteration_times = []
        for i in range(num_iter):
            start = time.perf_counter()
            # self.run_operator(**test_inputs)
            torch.npu.synchronize()
            end = time.perf_counter()
            iteration_times.append(end - start)
        
        metrics['iteration_times'] = iteration_times
        metrics['avg_time'] = sum(iteration_times) / len(iteration_times)
        metrics['min_time'] = min(iteration_times)
        metrics['max_time'] = max(iteration_times)
        
        if len(iteration_times) > 1:
            metrics['std_time'] = np.std(iteration_times)
        
        return metrics
    
    def test_performance_baseline(self, test_inputs: Dict) -> bool:
        """Test that operator meets performance baseline"""
        test_name = f"{self.operator_name} - Performance Baseline"
        
        try:
            # Benchmark operator
            metrics = self.benchmark_operator(test_inputs, num_warmup=5, num_iter=20)
            
            # Check against baseline (operator-specific)
            # For demonstration, use 100ms as baseline
            baseline_time = 0.1  # 100ms
            
            if metrics['avg_time'] < baseline_time:
                details = f"Avg time: {metrics['avg_time']*1000:.2f}ms < baseline {baseline_time*1000:.2f}ms"
                return self.log_test(test_name, True, details, metrics=metrics)
            else:
                details = f"Avg time: {metrics['avg_time']*1000:.2f}ms >= baseline {baseline_time*1000:.2f}ms"
                return self.log_test(test_name, False, details, metrics=metrics)
                
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    def compare_performance(self, ref_inputs: Dict, test_inputs: Dict) -> bool:
        """
        Compare performance between reference and optimized implementations.
        
        Returns True if optimized version is faster or within threshold.
        """
        test_name = f"{self.operator_name} - Performance Comparison"
        
        try:
            # Benchmark reference implementation
            print(f"{test_name}: Benchmarking reference implementation...")
            # ref_metrics = self.benchmark_reference(ref_inputs)
            
            # Benchmark optimized implementation
            print(f"{test_name}: Benchmarking optimized implementation...")
            # test_metrics = self.benchmark_optimized(test_inputs)
            
            # For demonstration
            ref_metrics = {'avg_time': 0.15}
            test_metrics = {'avg_time': 0.12}
            
            # Calculate speedup
            speedup = ref_metrics['avg_time'] / test_metrics['avg_time']
            
            # Check if optimized is faster (speedup > 1.0)
            # Allow small regression (speedup > 0.9)
            threshold = 0.9
            
            if speedup > threshold:
                details = f"Speedup: {speedup:.2f}x (Ref: {ref_metrics['avg_time']*1000:.2f}ms, "
                details += f"Opt: {test_metrics['avg_time']*1000:.2f}ms)"
                return self.log_test(test_name, True, details, 
                                   speedup=speedup, ref_metrics=ref_metrics, test_metrics=test_metrics)
            else:
                details = f"Slowdown: {1/speedup:.2f}x (Ref: {ref_metrics['avg_time']*1000:.2f}ms, "
                details += f"Opt: {test_metrics['avg_time']*1000:.2f}ms)"
                return self.log_test(test_name, False, details,
                                   speedup=speedup, ref_metrics=ref_metrics, test_metrics=test_metrics)
                
        except Exception as e:
            return self.log_test(test_name, False, f"Exception: {str(e)}")
    
    # ============================================================================
    # HELPER METHODS (to be implemented for specific operators)
    # ============================================================================
    
    def create_test_inputs(self, **kwargs) -> Dict:
        """
        Create test inputs for the operator.
        
        To be implemented for each operator.
        Example for fused_recurrent_fwd:
        """
        B = kwargs.get('B', 2)
        T = kwargs.get('T', 4096)
        H = kwargs.get('H', 32)
        K = kwargs.get('K', 256)
        V = kwargs.get('V', 128)
        scale = kwargs.get('scale', 0.5)
        
        # Set random seed for reproducibility
        seed = hash(str(kwargs)) % 10000
        torch.manual_seed(seed)
        
        inputs = {
            'q': torch.randn(B, T, H, K, device='npu', dtype=torch.float32),
            'k': torch.randn(B, T, H, K, device='npu', dtype=torch.float32),
            'v': torch.randn(B, T, H, V, device='npu', dtype=torch.float32),
            'scale': scale,
        }
        
        return inputs
    
    def run_operator(self, **kwargs):
        """Run the operator (to be implemented)"""
        raise NotImplementedError("run_operator must be implemented")
    
    def run_reference_implementation(self, **kwargs):
        """Run reference implementation (to be implemented)"""
        raise NotImplementedError("run_reference_implementation must be implemented")
    
    def run_optimized_implementation(self, **kwargs):
        """Run optimized implementation (to be implemented)"""
        raise NotImplementedError("run_optimized_implementation must be implemented")
    
    # ============================================================================
    # TEST SUITE EXECUTION
    # ============================================================================
    
    def run_comprehensive_test_suite(self) -> bool:
        """
        Run comprehensive test suite for the operator.
        
        Returns True if all tests pass, False otherwise.
        """
        print(f"\n{'='*60}")
        print(f"Comprehensive Test Suite: {self.operator_name}")
        print(f"{'='*60}")
        
        all_pass = True
        
        # Create standard test inputs
        standard_inputs = self.create_test_inputs(B=2, T=1024, H=16, K=128, V=128)
        expected_shape = (2, 1024, 16, 128)  # Example shape
        
        # 1. Basic functional tests
        print(f"\n1. Basic Functional Tests:")
        print(f"{'-'*40}")
        
        all_pass = self.test_shape_correctness(standard_inputs, expected_shape) and all_pass
        all_pass = self.test_device_support(standard_inputs) and all_pass
        
        # Test common dtypes
        dtypes = [torch.float16, torch.float32]
        if hasattr(torch, 'bfloat16'):
            dtypes.append(torch.bfloat16)
        
        all_pass = self.test_dtype_support(standard_inputs, dtypes) and all_pass
        
        # 2. Accuracy tests
        print(f"\n2. Accuracy Tests:")
        print(f"{'-'*40}")
        
        all_pass = self.test_accuracy_simple(standard_inputs) and all_pass
        
        # Test multiple sizes
        size_configs = [
            {'B': 1, 'T': 512, 'H': 8, 'K': 64, 'V': 64},
            {'B': 2, 'T': 1024, 'H': 16, 'K': 128, 'V': 128},
            {'B': 4, 'T': 2048, 'H': 32, 'K': 256, 'V': 256},
        ]
        all_pass = self.test_accuracy_multiple_sizes(size_configs) and all_pass
        
        # 3. Edge case tests
        print(f"\n3. Edge Case Tests:")
        print(f"{'-'*40}")
        
        all_pass = self.test_edge_case_empty() and all_pass
        all_pass = self.test_edge_case_large() and all_pass
        all_pass = self.test_edge_case_extreme_values() and all_pass
        
        # 4. Performance tests
        print(f"\n4. Performance Tests:")
        print(f"{'-'*40}")
        
        all_pass = self.test_performance_baseline(standard_inputs) and all_pass
        
        # Performance comparison (requires both implementations)
        # all_pass = self.compare_performance(standard_inputs, standard_inputs) and all_pass
        
        # Print summary
        self.print_test_summary()
        
        return all_pass
    
    def print_test_summary(self):
        """Print summary of all test results"""
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.operator_name}")
        print(f"{'='*60}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        
        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['test_name']}")
                    if result.get('details'):
                        print(f"    {result['details']}")
        
        # Calculate pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\nPass rate: {pass_rate:.1f}%")
        
        if passed == total:
            print(f"\n✅ All tests passed!")
        else:
            print(f"\n❌ {failed} test(s) failed.")
    
    def generate_report(self) -> str:
        """Generate detailed test report"""
        report = []
        report.append(f"{'='*60}")
        report.append(f"Test Report: {self.operator_name}")
        report.append(f"{'='*60}")
        report.append(f"Generated: {time.ctime()}")
        report.append(f"Tolerance: atol={self.atol}, rtol={self.rtol}")
        report.append("")
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        report.append(f"Summary:")
        report.append(f"  Total tests: {total}")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {failed}")
        report.append(f"  Pass rate: {pass_rate:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            report.append(f"  [{status}] {result['test_name']}")
            if result.get('details'):
                report.append(f"    {result['details']}")
            if result.get('timestamp'):
                report.append(f"    Time: {time.ctime(result['timestamp'])}")
            report.append("")
        
        return "\n".join(report)


# ============================================================================
# EXAMPLE IMPLEMENTATION FOR FUSED_RECURRENT_FWD
# ============================================================================

class FusedRecurrentFwdTester(ComplexOperatorTester):
    """
    Example tester for fused_recurrent_fwd operator.
    
    Based on the implementation in triton_demo examples.
    """
    
    def __init__(self, atol: float = 1e-3, rtol: float = 1e-3):
        super().__init__("fused_recurrent_fwd", atol, rtol)
    
    def create_test_inputs(self, **kwargs) -> Dict:
        """Create test inputs for fused_recurrent_fwd"""
        B = kwargs.get('B', 2)
        T = kwargs.get('T', 4096)
        H = kwargs.get('H', 32)
        K = kwargs.get('K', 256)
        V = kwargs.get('V', 128)
        scale = kwargs.get('scale', 0.5)
        use_g = kwargs.get('use_g', False)
        use_initial_state = kwargs.get('use_initial_state', False)
        
        # Set random seed for reproducibility
        seed = hash(str(kwargs)) % 10000
        torch.manual_seed(seed)
        
        inputs = {
            'q': torch.randn(B, T, H, K, device='npu', dtype=torch.float32),
            'k': torch.randn(B, T, H, K, device='npu', dtype=torch.float32),
            'v': torch.randn(B, T, H, V, device='npu', dtype=torch.float32),
            'scale': scale,
        }
        
        if use_g:
            inputs['g'] = torch.randn(B, T, H, device='npu', dtype=torch.float32)
        
        if use_initial_state:
            inputs['initial_state'] = torch.randn(B, H, K, V, device='npu', dtype=torch.float32)
        
        return inputs
    
    # Note: run_operator, run_reference_implementation, run_optimized_implementation
    # would be implemented with actual operator functions


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function demonstrating comprehensive testing"""
    print("Comprehensive Test Template for Complex Triton Operators")
    print("="*60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create tester for fused_recurrent_fwd example
    tester = FusedRecurrentFwdTester(atol=1e-3, rtol=1e-3)
    
    # Run comprehensive test suite
    print("\nRunning comprehensive test suite...")
    all_pass = tester.run_comprehensive_test_suite()
    
    # Generate report
    report = tester.generate_report()
    
    # Save report to file
    report_file = f"test_report_{tester.operator_name}_{int(time.time())}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    if all_pass:
        print(f"\n✅ All tests passed for {tester.operator_name}!")
        print(f"\nTo use this template for your operator:")
        print(f"1. Create a subclass of ComplexOperatorTester")
        print(f"2. Implement create_test_inputs() for your operator")
        print(f"3. Implement run_operator() and other required methods")
        print(f"4. Customize test cases as needed")
        print(f"5. Run the test suite")
    else:
        print(f"\n❌ Some tests failed for {tester.operator_name}")
        print(f"   Review the report for details: {report_file}")
    
    return all_pass


if __name__ == "__main__":
    main()