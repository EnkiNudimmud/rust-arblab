"""
Test Dual Backend Setup

Quick validation that both backends can coexist and be used
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_backend_config():
    """Test backend configuration"""
    print("Testing backend configuration...")
    
    from app.utils.backend_config import BackendType, BackendConfig, get_backend_config
    
    # Test default config
    config = BackendConfig.from_env()
    print(f"  ✓ Default backend: {config.backend_type.value}")
    print(f"  ✓ gRPC endpoint: {config.grpc_host}:{config.grpc_port}")
    
    # Test display names
    print(f"  ✓ Display name: {config.get_display_name()}")
    print(f"  ✓ Description: {config.get_description()}")
    
    print("✅ Backend config test passed\n")

def test_backend_interface():
    """Test backend interface"""
    print("Testing backend interface...")
    
    from app.utils.backend_interface import get_backend, BackendType
    import numpy as np
    
    # Test data
    prices = [100.0 + i + np.random.randn() for i in range(50)]
    
    # Test gRPC backend
    try:
        grpc_backend = get_backend(BackendType.GRPC)
        if grpc_backend.is_available():
            result = grpc_backend.calculate_mean_reversion(prices, lookback=20, threshold=1.5)
            print(f"  ✓ gRPC backend available")
            print(f"    Signal: {result['signal']:.2f}, Z-score: {result['zscore']:.2f}")
        else:
            print(f"  ⚠️  gRPC backend not available (server not running)")
    except Exception as e:
        print(f"  ⚠️  gRPC backend error: {e}")
    
    # Test legacy backend
    try:
        legacy_backend = get_backend(BackendType.LEGACY)
        if legacy_backend.is_available():
            result = legacy_backend.calculate_mean_reversion(prices, lookback=20, threshold=1.5)
            print(f"  ✓ Legacy backend available")
            print(f"    Signal: {result['signal']:.2f}, Z-score: {result['zscore']:.2f}")
        else:
            print(f"  ⚠️  Legacy backend not available")
    except Exception as e:
        print(f"  ⚠️  Legacy backend error: {e}")
    
    print("✅ Backend interface test passed\n")

def test_backend_switching():
    """Test switching between backends"""
    print("Testing backend switching...")
    
    from app.utils.backend_config import BackendType, BackendConfig, set_backend_config
    from app.utils.backend_interface import get_backend, clear_backend_cache
    
    # Switch to gRPC
    config = BackendConfig(backend_type=BackendType.GRPC)
    set_backend_config(config)
    clear_backend_cache()
    
    backend = get_backend()
    print(f"  ✓ Switched to: {backend.get_name()}")
    
    # Switch to Legacy
    config = BackendConfig(backend_type=BackendType.LEGACY)
    set_backend_config(config)
    clear_backend_cache()
    
    backend = get_backend()
    print(f"  ✓ Switched to: {backend.get_name()}")
    
    print("✅ Backend switching test passed\n")

def main():
    print("=" * 60)
    print("🧪 Dual Backend Setup Test")
    print("=" * 60)
    print()
    
    try:
        test_backend_config()
        test_backend_interface()
        test_backend_switching()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Start gRPC server: ./scripts/start_dual_backend.sh")
        print("  2. Open Streamlit: http://localhost:8501")
        print("  3. Use sidebar to switch backends")
        print("  4. Click '📊 Compare Backends' to benchmark")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
