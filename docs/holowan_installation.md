# Holowan Package Installation Documentation

## Overview
This document describes the process of installing the `holowan` package from the local `third_party` directory using the `uv` package manager.

## Package Details
- **Package Name**: holowan
- **Version**: 2908
- **Type**: Python SDK for HoloWAN_Huawei_edition
- **Location**: `third_party/holowan_python_api-2908`

## Installation Process

### Step 1: Install the Package with uv
```bash
# Install the package in editable mode
uv pip install -e third_party/holowan_python_api-2908
```

### Step 2: Update Project Dependencies
Added the following line to `pyproject.toml`:
```toml
"holowan @ file://${PROJECT_ROOT}/third_party/holowan_python_api-2908",
```

## Verification Results

### Installation Verification
```bash
$ uv pip list | grep holowan
holowan            2908        /home/sangachy/NetFaker/third_party/holowan_python_api-2908
```

### Import Verification
```bash
$ .venv/bin/python -c "import holowan; print('✓ Successfully imported holowan package'); print(f'Package path: {holowan.__file__}')"
✓ Successfully imported holowan package
Package path: /home/sangachy/NetFaker/third_party/holowan_python_api-2908/holowan/__init__.py
```

## Dependencies Installed
The installation also included the following dependencies:
- certifi==2026.1.4
- charset-normalizer==3.4.4
- idna==3.11
- requests==2.32.5
- urllib3==2.6.3

## Key Benefits
1. **Local Development**: Package is installed in editable mode, allowing for easy modifications
2. **Version Control**: Uses the local directory for version management
3. **Fast Installation**: Leverages uv's optimized installation process
4. **Consistent Dependencies**: Properly tracked in pyproject.toml
5. **No External Downloads**: Uses the local third_party directory

## Future Management

### Updating the Package
To update the holowan package:
1. Replace the contents of `third_party/holowan_python_api-2908` with the new version
2. Run `uv pip install -e third_party/holowan_python_api-2908` to reinstall in editable mode
3. Update the version reference in documentation if needed

### Verifying Installation
To verify the installation at any time:
```bash
# Check if installed
uv pip list | grep holowan

# Test import
.venv/bin/python -c "import holowan; print('holowan imported successfully')"
```

## Troubleshooting
- **Issue**: Package not found when importing
  **Solution**: Ensure you're using the virtual environment created by uv (`.venv/bin/python`)

- **Issue**: Workspace member missing pyproject.toml
  **Solution**: Use `uv pip install` instead of `uv add` for local packages without pyproject.toml

## Conclusion
The holowan package has been successfully installed from the local `third_party` directory using uv. It is properly integrated into the project's dependency management system and is ready for use.