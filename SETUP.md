# Project setup

Requires Python 3.11 or newer. From PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

If `py -3.11` is unavailable, install Python from python.org and enable the
Python Launcher during installation. Do not reuse a virtual environment copied
from another machine; recreate `.venv` with the commands above.

For notebooks, select the `.venv` interpreter as the Jupyter kernel.
