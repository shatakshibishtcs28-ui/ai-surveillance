@echo off
call venv\Scripts\activate
python surveillance.py video.mp4 --location server_room --hour 2
pause
