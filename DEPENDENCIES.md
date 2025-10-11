# Dependencies Management Guide

This project uses a **two-file approach** for managing dependencies:

## Files

### `requirements.in` (Edit This)
**What it is:** Your direct, minimal dependencies only.
**When to edit:** When you want to add, remove, or update a package.

```
# Core ML dependencies
torch>=2.8.0
torchaudio>=2.8.0

# Audio backend
soundfile>=0.13.0
```

### `requirements.txt` (Auto-generated)
**What it is:** Fully frozen dependencies with exact versions (includes all transitive dependencies).
**When to edit:** Never manually! Regenerate it using `pip-compile`.

### `requirements-dev.txt`
**What it is:** Includes all production dependencies plus development tools.
**Use for:** Local development, testing, and running notebooks.

## Workflow

### Installing Dependencies

**For end users / production:**
```bash
pip install -r requirements.txt
```

**For developers:**
```bash
pip install -r requirements-dev.txt
```

### Adding a New Dependency

1. Edit `requirements.in` and add your package:
   ```
   # Add this line
   numpy>=1.24.0
   ```

2. Regenerate `requirements.txt`:
   ```bash
   pip-compile requirements.in
   ```

3. Install the new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Test your code

5. Commit both `requirements.in` and `requirements.txt`

### Updating All Dependencies

To update all packages to their latest compatible versions:
```bash
pip-compile --upgrade requirements.in
```

### Updating a Single Package

To update just one package (e.g., torch):
```bash
pip-compile --upgrade-package torch requirements.in
```

## Benefits

✅ **requirements.in:**
- Easy to read and maintain
- Only shows what YOU explicitly depend on
- Easy to review in pull requests
- No merge conflicts with transitive dependencies

✅ **requirements.txt:**
- Perfect reproducibility (exact versions)
- Includes all transitive dependencies
- Safe for production deployments
- Deterministic builds

## Examples

### Before (Hard to maintain):
```
# requirements.txt - 127 packages!
torch==2.8.0
numpy==2.3.3
pillow==11.3.0
... 124 more packages
```

### After (Easy to maintain):
```
# requirements.in - 3 packages!
torch>=2.8.0
torchaudio>=2.8.0
soundfile>=0.13.0
```

The `requirements.txt` still has all 127 packages with exact versions, but you only need to manage 3 lines!

