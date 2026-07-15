@echo off
setlocal

cd /d "%~dp0"

rem ===========DATA=============
set "INSTR_PATH=.\data\instr.txt"

rem ===========COE=============
set "INSTR_COE_PATH=.\coe\instr.coe"

rem ===========TARGET COE=============
set "TARGET_COE_PATH=.\target\all.coe"
set "NPU_PARAMS_BIN_PATH=.\target\npu_params.bin"
set "NPU_INSTR_BIN_PATH=.\target\npu_instr.bin"

if "%~1"=="clean" (
    call :clean
    exit /b %ERRORLEVEL%
)

if "%~1"=="distclean" (
    call :clean
    if errorlevel 1 exit /b 1
    call :distclean
    exit /b %ERRORLEVEL%
)

if not "%~1"=="" (
    echo Usage: build.bat [clean^|distclean] 1>&2
    exit /b 1
)

python .\python\generate_model_ir.py
if errorlevel 1 exit /b 1

rem generate memory plan / instructions
python .\python\generate_memory_plan.py
if errorlevel 1 exit /b 1

python .\python\generate_instr.py
if errorlevel 1 exit /b 1

rem interleaved weight/bias parameter coe
python .\python\params_to_bram_coe.py --memory-plan .\data\memory_plan.json --model-params .\data\model_params --out-dir .\coe
if errorlevel 1 exit /b 1

rem instr to coe
python .\python\instr_txt_to_bram_coe.py "%INSTR_PATH%" "%INSTR_COE_PATH%"
if errorlevel 1 exit /b 1

rem merge coe
python .\python\merge.py --memory-plan .\data\memory_plan.json "%TARGET_COE_PATH%"
if errorlevel 1 exit /b 1

rem export DDR binary images
python .\python\coe_to_bin\export_npu_bins.py --memory-plan .\data\memory_plan.json --out-dir .\target
if errorlevel 1 exit /b 1

exit /b 0

:clean
echo [CLEAN] remove generated files
if exist ".\data\model_params" rmdir /s /q ".\data\model_params"
if exist ".\data\tmp_regression" rmdir /s /q ".\data\tmp_regression"
if exist ".\data\model_ir.json" del /q ".\data\model_ir.json"
if exist ".\data\memory_plan.json" del /q ".\data\memory_plan.json"
if exist ".\data\instr.asm" del /q ".\data\instr.asm"
if exist ".\data\instr.txt" del /q ".\data\instr.txt"
del /q ".\coe\layer*_params.coe" 2>nul
del /q ".\coe\linear*_params.coe" 2>nul
if exist ".\coe\instr.coe" del /q ".\coe\instr.coe"
if exist "%TARGET_COE_PATH%" del /q "%TARGET_COE_PATH%"
if exist ".\target\all.coe.map.txt" del /q ".\target\all.coe.map.txt"
if exist "%NPU_PARAMS_BIN_PATH%" del /q "%NPU_PARAMS_BIN_PATH%"
if exist "%NPU_INSTR_BIN_PATH%" del /q "%NPU_INSTR_BIN_PATH%"
for /d /r %%D in (__pycache__) do (
    if exist "%%D" rmdir /s /q "%%D"
)
echo [CLEAN] done
exit /b 0

:distclean
echo [DISTCLEAN] remove regression error logs
if exist ".\test\error" (
    del /q ".\test\error\*.log" 2>nul
    del /q ".\test\error\*.log.txt" 2>nul
    del /q ".\test\error\*.txt" 2>nul
)
echo [DISTCLEAN] done
exit /b 0
