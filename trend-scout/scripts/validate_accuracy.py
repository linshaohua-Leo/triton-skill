#!/usr/bin/env python3
"""
Accuracy Validation for Complex Triton Operators

This template provides comprehensive accuracy validation for complex
Triton operators after NPU optimization, based on the pattern from
triton_demo/ori1 main function.

Key features:
1. Multiple test cases with different sizes
2. Statistical analysis of outputs
3. Detailed error reporting
4. Tolerance-based validation
5. NaN/Inf detection
"""

import torch
import torch_npu
import numpy as np
import random
from typing import Tuple, Optional, List


def set_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def analyze_tensor(tensor: torch.Tensor, name: str) -> dict:
    """
    Analyze tensor statistics.
    
    Returns:
        Dictionary containing tensor statistics
    """
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'nan_count': torch.isnan(tensor).sum().item() if torch.isnan(tensor).any() else 0,
        'inf_count': torch.isinf(tensor).sum().item() if torch.isinf(tensor).any() else 0,
    }
    return stats


def print_tensor_stats(stats: dict, indent: str = "  "):
    """Print tensor statistics in readable format"""
    print(f"{indent}{stats['name']}:")
    print(f"{indent}  Shape: {stats['shape']}")
    print(f"{indent}  Dtype: {stats['dtype']}")
    print(f"{indent}  Device: {stats['device']}")
    print(f"{indent}  Min: {stats['min']:.6f}")
    print(f"{indent}  Max: {stats['max']:.6f}")
    print(f"{indent}  Mean: {stats['mean']:.6f}")
    print(f"{indent}  Std: {stats['std']:.6f}")
    
    if stats['has_nan']:
        print(f"{indent}  ⚠️  Contains NaN: {stats['nan_count']}")
    
    if stats['has_inf']:
        print(f"{indent}  ⚠️  Contains Inf: {stats['inf_count']}")


def compare_tensors(
    ref_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
    test_name: str = "",
    atol: float = 1e-3,
    rtol: float = 1e-3,
    verbose: bool = True
) -> Tuple[bool, dict]:
    """
    Compare two tensors and return detailed analysis.
    
    Args:
        ref_tensor: Reference tensor (original implementation)
        test_tensor: Test tensor (optimized implementation)
        test_name: Name of the test for reporting
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (success: bool, analysis: dict)
    """
    analysis = {
        'test_name': test_name,
        'shapes_match': ref_tensor.shape == test_tensor.shape,
        'max_abs_error': 0.0,
        'mean_abs_error': 0.0,
        'max_rel_error': 0.0,
        'passes_absolute': False,
        'passes_relative': False,
        'has_nan': torch.isnan(test_tensor).any().item(),
        'has_inf': torch.isinf(test_tensor).any().item(),
        'max_error_position': None,
        'ref_value_at_max_error': 0.0,
        'test_value_at_max_error': 0.0,
    }
    
    if not analysis['shapes_match']:
        if verbose:
            print(f"\n❌ {test_name}: Shape mismatch")
            print(f"   Reference shape: {ref_tensor.shape}")
            print(f"   Test shape: {test_tensor.shape}")
        return False, analysis
    
    # Check for NaN/Inf in test tensor
    if analysis['has_nan']:
        nan_count = torch.isnan(test_tensor).sum().item()
        analysis['nan_count'] = nan_count
        if verbose:
            print(f"\n❌ {test_name}: Test tensor contains {nan_count} NaN values")
        return False, analysis
    
    if analysis['has_inf']:
        inf_count = torch.isinf(test_tensor).sum().item()
        analysis['inf_count'] = inf_count
        if verbose:
            print(f"\n❌ {test_name}: Test tensor contains {inf_count} Inf values")
        return False, analysis
    
    # Calculate absolute differences
    abs_diff = torch.abs(ref_tensor - test_tensor)
    analysis['max_abs_error'] = abs_diff.max().item()
    analysis['mean_abs_error'] = abs_diff.mean().item()
    
    # Calculate relative differences (avoid division by zero)
    ref_abs = torch.abs(ref_tensor)
    rel_diff = abs_diff / (ref_abs + 1e-8)
    analysis['max_rel_error'] = rel_diff.max().item()
    
    # Find position of maximum error
    if analysis['max_abs_error'] > 0:
        max_idx = torch.argmax(abs_diff.flatten())
        analysis['max_error_position'] = tuple(torch.unravel_index(max_idx, abs_diff.shape))
        analysis['ref_value_at_max_error'] = ref_tensor.flatten()[max_idx].item()
        analysis['test_value_at_max_error'] = test_tensor.flatten()[max_idx].item()
    
    # Check tolerances
    analysis['passes_absolute'] = analysis['max_abs_error'] < atol
    analysis['passes_relative'] = analysis['max_rel_error'] < rtol
    
    success = analysis['passes_absolute'] and analysis['passes_relative']
    
    if verbose:
        if success:
            print(f"\n✅ {test_name}: PASS")
        else:
            print(f"\n❌ {test_name}: FAIL")
        
        print(f"   Max absolute error: {analysis['max_abs_error']:.6e}")
        print(f"   Mean absolute error: {analysis['mean_abs_error']:.6e}")
        print(f"   Max relative error: {analysis['max_rel_error']:.6e}")
        
        if analysis['max_abs_error'] > atol:
            print(f"   ✗ Absolute error exceeds tolerance: {analysis['max_abs_error']:.6e} > {atol}")
        
        if analysis['max_rel_error'] > rtol:
            print(f"   ✗ Relative error exceeds tolerance: {analysis['max_rel_error']:.6e} > {rtol}")
        
        if analysis['max_error_position']:
            print(f"   Max error at position: {analysis['max_error_position']}")
            print(f"   Reference value: {analysis['ref_value_at_max_error']:.6f}")
            print(f"   Test value: {analysis['test_value_at_max_error']:.6f}")
    
    return success, analysis


class ComplexOperatorValidator:
    """
    Validator for complex Triton operators with multiple test cases.
    
    Based on the pattern from triton_demo/ori1 main function.
    """
    
    def __init__(self, atol: float = 1e-3, rtol: float = 1e-3):
        self.atol = atol
        self.rtol = rtol
        self.test_results = []
        
    def create_test_case(self, case_name: str, **kwargs) -> dict:
        """
        Create a test case for the operator.
        
        Common parameters for fused_recurrent_fwd example:
        - B: batch size
        - T: sequence length
        - H: number of heads
        - K: head dimension for Q/K
        - V: head dimension for V
        - scale: scaling factor
        - use_g: whether to use g tensor
        - use_initial_state: whether to use initial state
        """
        test_case = {
            'name': case_name,
            'parameters': kwargs,
            'inputs': {},
            'ref_output': None,
            'test_output': None,
            'success': None,
            'analysis': None,
        }
        return test_case
    
    def generate_inputs(self, test_case: dict, device: str = 'npu'):
        """
        Generate input tensors for the test case.
        
        This should be customized for each operator.
        """
        params = test_case['parameters']
        
        # Example for fused_recurrent_fwd
        B = params.get('B', 2)
        T = params.get('T', 4096)
        H = params.get('H', 32)
        K = params.get('K', 256)
        V = params.get('V', 128)
        scale = params.get('scale', 0.5)
        use_g = params.get('use_g', False)
        use_initial_state = params.get('use_initial_state', False)
        
        # Set random seed for reproducibility
        torch.manual_seed(hash(test_case['name']) % 10000)
        
        # Generate tensors
        inputs = {
            'q': torch.randn(B, T, H, K, device=device, dtype=torch.float32),
            'k': torch.randn(B, T, H, K, device=device, dtype=torch.float32),
            'v': torch.randn(B, T, H, V, device=device, dtype=torch.float32),
            'scale': scale,
        }
        
        if use_g:
            inputs['g'] = torch.randn(B, T, H, device=device, dtype=torch.float32)
        
        if use_initial_state:
            inputs['initial_state'] = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
        
        test_case['inputs'] = inputs
        return inputs
    
    def run_reference_implementation(self, test_case: dict):
        """
        Run the reference implementation (original GPU-style).
        
        This should be implemented for each operator.
        """
        # Placeholder - implement with actual reference function
        # Example:
        # from original_implementation import fused_recurrent_fwd
        # inputs = test_case['inputs']
        # output_ref, final_state_ref = fused_recurrent_fwd(**inputs)
        # test_case['ref_output'] = output_ref
        
        # For demonstration, create dummy output
        inputs = test_case['inputs']
        B, T, H, K = inputs['q'].shape
        V = inputs['v'].shape[-1]
        test_case['ref_output'] = torch.randn(B, T, H, V, device=inputs['q'].device, dtype=torch.float32)
        
        return test_case['ref_output']
    
    def run_test_implementation(self, test_case: dict):
        """
        Run the test implementation (optimized NPU-style).
        
        This should be implemented for each operator.
        """
        # Placeholder - implement with actual test function
        # Example:
        # from optimized_implementation import fused_recurrent_fwd_new
        # inputs = test_case['inputs']
        # output_test, final_state_test = fused_recurrent_fwd_new(**inputs)
        # test_case['test_output'] = output_test
        
        # For demonstration, create output similar to reference with small noise
        ref_output = test_case['ref_output']
        noise = torch.randn_like(ref_output) * 1e-4
        test_case['test_output'] = ref_output + noise
        
        return test_case['test_output']
    
    def validate_test_case(self, test_case: dict, verbose: bool = True) -> bool:
        """
        Validate a single test case.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Test Case: {test_case['name']}")
            print(f"{'='*60}")
        
        # Generate inputs
        self.generate_inputs(test_case)
        
        # Print input statistics
        if verbose:
            print("\nInput Statistics:")
            for name, tensor in test_case['inputs'].items():
                if isinstance(tensor, torch.Tensor):
                    stats = analyze_tensor(tensor, name)
                    print_tensor_stats(stats)
        
        # Run implementations
        if verbose:
            print("\nRunning reference implementation...")
        self.run_reference_implementation(test_case)
        
        if verbose:
            print("Running test implementation...")
        self.run_test_implementation(test_case)
        
        # Compare outputs
        success, analysis = compare_tensors(
            test_case['ref_output'],
            test_case['test_output'],
            test_case['name'],
            self.atol,
            self.rtol,
            verbose
        )
        
        test_case['success'] = success
        test_case['analysis'] = analysis
        
        # Store result
        self.test_results.append(test_case)
        
        return success
    
    def run_test_suite(self, test_cases: List[dict], verbose: bool = True) -> bool:
        """
        Run a suite of test cases.
        
        Returns:
            True if all test cases pass, False otherwise
        """
        print("\n" + "="*60)
        print("Complex Operator Validation Suite")
        print("="*60)
        
        all_pass = True
        
        for test_case in test_cases:
            success = self.validate_test_case(test_case, verbose)
            if not success:
                all_pass = False
        
        # Print summary
        self.print_summary()
        
        return all_pass
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("Validation Summary")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        failed = total - passed
        
        print(f"\nTotal test cases: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed test cases:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['name']}")
                    if result['analysis']:
                        print(f"    Max abs error: {result['analysis']['max_abs_error']:.6e}")
        
        if passed == total:
            print("\n✅ All test cases passed!")
        else:
            print(f"\n❌ {failed} test case(s) failed.")
    
    def get_detailed_report(self) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("="*60)
        report.append("Detailed Validation Report")
        report.append("="*60)
        
        for result in self.test_results:
            report.append(f"\nTest Case: {result['name']}")
            report.append(f"  Status: {'PASS' if result['success'] else 'FAIL'}")
            
            if result['analysis']:
                analysis = result['analysis']
                report.append(f"  Max absolute error: {analysis['max_abs_error']:.6e}")
                report.append(f"  Mean absolute error: {analysis['mean_abs_error']:.6e}")
                report.append(f"  Max relative error: {analysis['max_rel_error']:.6e}")
                
                if analysis['max_error_position']:
                    report.append(f"  Max error position: {analysis['max_error_position']}")
                    report.append(f"  Reference value: {analysis['ref_value_at_max_error']:.6f}")
                    report.append(f"  Test value: {analysis['test_value_at_max_error']:.6f}")
        
        return "\n".join(report)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_validation():
    """
    Example validation for fused_recurrent_fwd operator.
    
    Based on the pattern from triton_demo/ori1 main function.
    """
    validator = ComplexOperatorValidator(atol=1e-3, rtol=1e-3)
    
    # Define test cases
    test_cases = [
        validator.create_test_case(
            "Small batch, short sequence",
            B=2, T=512, H=8, K=64, V=64, scale=0.5
        ),
        validator.create_test_case(
            "Medium batch, medium sequence",
            B=4, T=1024, H=16, K=128, V=128, scale=0.5
        ),
        validator.create_test_case(
            "Large batch, long sequence",
            B=2, T=4096, H=32, K=256, V=128, scale=0.5
        ),
        validator.create_test_case(
            "With g tensor",
            B=2, T=1024, H=16, K=128, V=128, scale=0.5, use_g=True
        ),
        validator.create_test_case(
            "With initial state",
            B=2, T=1024, H=16, K=128, V=128, scale=0.5, use_initial_state=True
        ),
    ]
    
    # Run validation suite
    all_pass = validator.run_test_suite(test_cases, verbose=True)
    
    # Generate detailed report
    print("\n" + validator.get_detailed_report())
    
    return all_pass


def main():
    """Main function demonstrating validation workflow"""
    print("Complex Triton Operator Accuracy Validation")
    print("="*60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Run example validation
    print("\nRunning example validation...")
    success = example_validation()
    
    if success:
        print("\n✅ Validation workflow demonstrated successfully!")
        print("\nTo use this for your operator:")
        print("1. Implement generate_inputs() for your operator")
        print("2. Implement run_reference_implementation()")
        print("3. Implement run_test_implementation()")
        print("4. Define appropriate test cases")
        print("5. Run the validation suite")
    else:
        print("\n❌ Example validation failed (expected for demonstration)")
    
    return success


if __name__ == "__main__":
    main()