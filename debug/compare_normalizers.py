"""
Compare two normalizer files to check if they have different parameters.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.normalizer import LinearNormalizer


def compare_normalizers(path1, path2):
    print("=" * 80)
    print("Normalizer Comparison")
    print("=" * 80)
    
    print(f"\nFile 1: {path1}")
    print(f"File 2: {path2}")
    
    # Check if files exist
    if not os.path.exists(path1):
        print(f"ERROR: {path1} does not exist!")
        return
    if not os.path.exists(path2):
        print(f"ERROR: {path2} does not exist!")
        return
    
    # Load normalizers
    print("\n[1] Loading normalizers...")
    norm1 = LinearNormalizer.load(path1)
    norm2 = LinearNormalizer.load(path2)
    
    # Compare keys
    print("\n[2] Comparing keys...")
    keys1 = set(norm1.params_dict.keys())
    keys2 = set(norm2.params_dict.keys())
    
    print(f"  Normalizer 1 keys: {keys1}")
    print(f"  Normalizer 2 keys: {keys2}")
    
    if keys1 != keys2:
        print(f"  WARNING: Keys are different!")
        print(f"    Only in norm1: {keys1 - keys2}")
        print(f"    Only in norm2: {keys2 - keys1}")
    else:
        print(f"  Keys match: {keys1}")
    
    # Compare parameters for each common key
    common_keys = keys1 & keys2
    
    print("\n[3] Comparing parameters for each key...")
    
    all_match = True
    for key in sorted(common_keys):
        print(f"\n  [{key}]")
        params1 = norm1.params_dict[key]
        params2 = norm2.params_dict[key]
        
        # Compare scale
        scale1 = params1['scale']
        scale2 = params2['scale']
        
        print(f"    scale1 shape: {scale1.shape}, scale2 shape: {scale2.shape}")
        
        if scale1.shape != scale2.shape:
            print(f"    ERROR: Scale shapes don't match!")
            all_match = False
        else:
            scale_diff = torch.abs(scale1 - scale2)
            scale_match = torch.allclose(scale1, scale2, rtol=1e-5, atol=1e-8)
            print(f"    scale match: {scale_match}")
            if not scale_match:
                all_match = False
                print(f"    scale max diff: {scale_diff.max().item():.10f}")
                print(f"    scale mean diff: {scale_diff.mean().item():.10f}")
                print(f"\n    scale1 values: {scale1.detach().numpy().flatten()}")
                print(f"    scale2 values: {scale2.detach().numpy().flatten()}")
        
        # Compare offset
        offset1 = params1['offset']
        offset2 = params2['offset']
        
        print(f"\n    offset1 shape: {offset1.shape}, offset2 shape: {offset2.shape}")
        
        if offset1.shape != offset2.shape:
            print(f"    ERROR: Offset shapes don't match!")
            all_match = False
        else:
            offset_diff = torch.abs(offset1 - offset2)
            offset_match = torch.allclose(offset1, offset2, rtol=1e-5, atol=1e-8)
            print(f"    offset match: {offset_match}")
            if not offset_match:
                all_match = False
                print(f"    offset max diff: {offset_diff.max().item():.10f}")
                print(f"    offset mean diff: {offset_diff.mean().item():.10f}")
                print(f"\n    offset1 values: {offset1.detach().numpy().flatten()}")
                print(f"    offset2 values: {offset2.detach().numpy().flatten()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if all_match:
        print("✓ All parameters MATCH - normalizers are identical")
    else:
        print("✗ Parameters DIFFER - normalizers are NOT identical")
        print("\nThis could cause scale/offset issues in inference!")
    
    return all_match


if __name__ == "__main__":
    # Default paths
    path1 = "rdt2_pika_shards/pika_normalizer.pt"
    path2 = "normalizer.pt"
    
    # Allow command line override
    if len(sys.argv) >= 3:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
    
    compare_normalizers(path1, path2)
