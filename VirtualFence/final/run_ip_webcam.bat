@echo off
echo.
echo ================================
echo IBVAP IP WEBCAM
echo ================================
echo.
set /p URL=Enter IP webcam VIDEO URL: 
if "%URL%"=="" (
  echo No URL entered.
  pause
  exit /b 1
)
.venv\Scripts\python.exe main.py --source "%URL%" --camera-id CAM-002
pause
