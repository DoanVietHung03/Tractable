@echo off
echo ==========================================
echo TESTING...
echo ==========================================
python ./Modules/Segments/segment.py
IF %ERRORLEVEL% NEQ 0 (
    echo [LOI] Testing gap van de! Dung quy trinh.
    pause
    exit /b
)

echo.
@REM echo ==========================================
@REM echo [BUOC 2] DANG CHAY THU NGHIEM (run.py)...
@REM echo ==========================================
@REM python ./pythonFile/run.py
@REM IF %ERRORLEVEL% NEQ 0 (
@REM     echo [LOI] Run gap van de! Dung quy trinh.
@REM     pause
@REM     exit /b
@REM )

@REM echo.
@REM echo ==========================================
@REM echo [BUOC 3] HIEN THI KET QUA (show_results.py)...
@REM echo ==========================================
@REM python ./pythonFile/show_results.py

@REM echo.
@REM echo [DONE] Da hoan tat toan bo!
pause