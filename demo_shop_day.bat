@echo off
call venv\Scripts\activate
python surveillance.py video.mp4 --location shop --hour 14
pause
