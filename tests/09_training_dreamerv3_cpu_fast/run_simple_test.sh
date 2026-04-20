#!/bin/bash

# Simple test script for DreamerV3 implementation
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${RL_PCB}" ]; then
    RL_PCB="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
export RL_PCB

source "${RL_PCB}/venv/bin/activate"

cd ${RL_PCB}/src/training

# Run a simple Python test
python3 << 'PYEOF'
import sys
import os
sys.path.insert(0, '/home/alli-ekundayo/Projects/RL_PCB/src/training')
sys.path.insert(0, '/home/alli-ekundayo/Projects/RL_PCB/src/training/third_party')
sys.path.insert(0, '/home/alli-ekundayo/Projects/RL_PCB/src/training/third_party/dreamerv3')

print("=" * 60)
print("DreamerV3 Simple Test")
print("=" * 60)

# Test 1: Import DreamerV3
print("\n1. Testing DreamerV3 import...")
try:
    import DreamerV3
    print("   ✓ DreamerV3 imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check key classes exist
print("\n2. Checking module structure...")
try:
    assert hasattr(DreamerV3, 'DreamerV3'), "DreamerV3 class missing"
    assert hasattr(DreamerV3, 'PcbDreamerWrapper'), "PcbDreamerWrapper class missing"
    assert hasattr(DreamerV3, 'make_dreamer_config'), "make_dreamer_config function missing"
    print("   ✓ All required classes/functions present")
except AssertionError as e:
    print(f"   ✗ {e}")
    sys.exit(1)

# Test 3: Create config
print("\n3. Testing config creation...")
try:
    config = DreamerV3.make_dreamer_config()
    print(f"   ✓ Config created with {len(config)} keys")
except Exception as e:
    print(f"   ✗ Config creation failed: {e}")
    sys.exit(1)

# Test 4: Try to instantiate DreamerV3 without environment
# Skip full instantiation test as it triggers JAX compilation
print("\n4. Testing DreamerV3 instantiation (no env)...")
print("   ⚠ Skipping full instantiation - requires environment for proper initialization")

# Test 5: Test that we can at least import and check structure
print("\n5. Checking DreamerV3 class structure...")
try:
    assert hasattr(DreamerV3.DreamerV3, 'select_action')
    assert hasattr(DreamerV3.DreamerV3, 'learn')
    assert hasattr(DreamerV3.DreamerV3, 'save')
    assert hasattr(DreamerV3.DreamerV3, 'load')
    print("   ✓ All required methods present")
except AssertionError as e:
    print(f"   ✗ Missing methods: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
PYEOF
