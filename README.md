# lyrebird

## Setup

### Create Virtual Environment

Create a Python virtual environment:

```bash
python3 -m venv venv
```

### Activate Virtual Environment

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows:
```cmd
venv\Scripts\activate
```

### Install Dependencies

For production use (core dependencies only):
```bash
pip install -r requirements.txt
```

For development (includes testing and notebooks):
```bash
pip install -r requirements-dev.txt
```

**Requirements Files:**
- `requirements.in` - Minimal direct dependencies (edit this to add/update packages)
- `requirements.txt` - Fully frozen with exact versions (generated from requirements.in)
- `requirements-dev.txt` - Development dependencies (testing, notebooks, etc.)

**To update dependencies:**
```bash
pip install pip-tools
pip-compile requirements.in
```

## Usage

### Start Jupyter Notebook

After activating the virtual environment and installing dependencies:

```bash
jupyter notebook
```

This will open Jupyter in your browser, where you can access `notebook.ipynb`.