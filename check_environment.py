"""Print Python and package versions for the paper and the README."""
import importlib
import platform
import sys

print("Python", sys.version.split()[0], "|", platform.platform())
for name in ["numpy", "pandas", "sklearn", "matplotlib", "xarray", "netCDF4", "openpyxl", "ee"]:
    try:
        print(f"{name:12s}", importlib.import_module(name).__version__)
    except Exception:
        print(f"{name:12s}", "not installed")
