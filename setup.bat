@echo off
echo === Setup shuru (5-10 min lagenge) ===
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo === Setup complete ===
pause
