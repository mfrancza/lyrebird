# Contributing to Lyrebird

## Development Process

### Branch Workflow

1. **Create a feature branch** for each change:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   # or: git checkout -b fix/bug-description
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git add <files>
   git commit -m "Brief description of change"
   ```

3. **Push your branch** and create a Pull Request:
   ```bash
   git push -u origin feature/your-feature-name
   gh pr create --title "Your PR title" --body "Description of changes"
   ```

4. **Request reviewers**: Add `mfrancza` and `copilot` as reviewers

5. **Address review feedback** and merge when approved

### PR Requirements

All PRs must pass automated checks before merging:

- **Python library**: Formatting (black), linting (ruff), unit tests (pytest)
- **VST plugin**: Build verification, unit tests (Catch2)
- **Pi runner** *(planned)*: Formatting, linting, unit tests

### Code Style

#### Python
- Format with `black`
- Lint with `ruff`
- Type hints encouraged but not required

#### C++ (VST Plugin)
- Follow existing code style
- Use clang-format if available

### Running Tests Locally

#### Python Library
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest test_lyrebird.py -v

# Format code
black lyrebird.py test_lyrebird.py export_rtneural.py

# Lint code
ruff check lyrebird.py test_lyrebird.py export_rtneural.py
```

#### VST Plugin
```bash
cd vst

# Linux (Docker)
./build.sh test

# Windows (from Developer Command Prompt)
.\build-windows.bat native
```

#### Pi Runner *(planned)*
```bash
cd pi
pytest tests/ -v
```

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Reference issues when applicable: "Fix #123: Description"

### Questions?

Open an issue or reach out to the maintainers.
