import numpy as np
import ctypes
from pathlib import Path
import sys
import platform

def main():
    try:
        print("Loading weights...")
        weights = np.load('model/weights.npz')
        transitions = np.ascontiguousarray(weights['transitions'].astype(np.float32))
        emissions = np.ascontiguousarray(weights['emissions'].astype(np.float32))
        state_size = transitions.shape[0]
        vocab_size = emissions.shape[0]
        
        print(f"Shapes: transitions={transitions.shape}, emissions={emissions.shape}")
        print(f"Memory layout: transitions={transitions.strides}, emissions={emissions.strides}")
        print(f"Data ranges: transitions={transitions.min():.2f} to {transitions.max():.2f}")
        print(f"            emissions={emissions.min():.2f} to {emissions.max():.2f}")
        
        print("\nLoading C library...")
        # Find the library file
        lib_files = list(Path('.').glob('libner*.so')) + list(Path('.').glob('libner*.dylib'))
        if not lib_files:
            lib_files = list(Path('build/lib*').glob('libner*.so')) + list(Path('build/lib*').glob('libner*.dylib'))
        
        if not lib_files:
            raise RuntimeError("Could not find libner shared library")
            
        lib_path = lib_files[0]
        print(f"Library path: {lib_path}")
        print(f"Library exists: {lib_path.exists()}")
        print(f"Library size: {lib_path.stat().st_size} bytes")
        
        lib = ctypes.CDLL(str(lib_path))
        print("Library loaded successfully")
        
        print("\nSetting up function signatures...")
        lib.init_model.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        ]
        lib.init_model.restype = ctypes.c_void_p
        print("Function signatures set up")
        
        print("\nPreparing to call init_model...")
        print(f"state_size: {state_size} ({type(state_size)})")
        print(f"vocab_size: {vocab_size} ({type(vocab_size)})")
        print(f"transitions: shape={transitions.shape}, dtype={transitions.dtype}, flags={transitions.flags}")
        print(f"emissions: shape={emissions.shape}, dtype={emissions.dtype}, flags={emissions.flags}")
        
        print("\nCalling init_model...")
        sys.stdout.flush()
        sys.stderr.flush()
        
        model_ptr = lib.init_model(
            ctypes.c_int32(state_size),
            ctypes.c_int32(vocab_size),
            transitions,
            emissions
        )
        
        print(f"Model pointer: {model_ptr}")
        if model_ptr:
            print("Success! Model initialized.")
            lib.free_model.argtypes = [ctypes.c_void_p]
            lib.free_model(model_ptr)
            print("Model freed.")
        else:
            print("Failed to initialize model")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    main()
