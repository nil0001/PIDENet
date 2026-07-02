@echo off
REM Windows quick-run script.
REM Assumes you have already:
REM   1) created a virtualenv in .\.venv     (python -m venv .venv)
REM   2) installed dependencies              (pip install -r requirements.txt)
REM Edit LINEMOD_PATH below to point at your dataset root, then double-click
REM this file OR run it from cmd/PowerShell.

setlocal
set LINEMOD_PATH=E:\paper\PIDENet\LINEMOD
set OBJECTS=1 5
set OUT_DIR=outputs

REM activate the virtualenv if it exists
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

echo === Phase 1: object-frame candidates on objects %OBJECTS% ===
python run_phase1.py --linemod "%LINEMOD_PATH%" --objects %OBJECTS% --out "%OUT_DIR%"
if errorlevel 1 goto :err

echo.
echo === (optional) Debug pipeline figures ===
python make_debug_figure.py --linemod "%LINEMOD_PATH%" --objects %OBJECTS% --out "%OUT_DIR%"
if errorlevel 1 goto :err

echo.
echo === Phase 2: per-frame camera-coordinate labels ===
python run_phase2.py --linemod "%LINEMOD_PATH%" --phase1 "%OUT_DIR%" --out "%OUT_DIR%" --objects %OBJECTS%
if errorlevel 1 goto :err

echo.
echo Done. Outputs in %CD%\%OUT_DIR%
pause
exit /b 0

:err
echo.
echo ERROR — see messages above.
pause
exit /b 1
