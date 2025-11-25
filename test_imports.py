import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

print("Importing src.utils...")
try:
    import src.utils
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing src.models...")
try:
    import src.models
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing src.dataset...")
try:
    import src.dataset
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")

print("Importing src.data_loader...")
try:
    import src.data_loader
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")
