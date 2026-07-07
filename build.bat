@echo off
setlocal

cd /d "%~dp0"

rem ========MODEL========
set "MODEL_NAME=yolov2_fixed.pth"
set "MODEL_PY_NAME=yolov2_14layer_quantized.py"

rem ===========DATA=============
set "IMAGE_PATH=.\data\image.jpg"
set "INSTR_PATH=.\data\instr.txt"

rem ===========COE=============
set "IMAGE_COE_PATH=.\coe\image.coe"
set "INSTR_COE_PATH=.\coe\instr.coe"

rem ===========TARGET COE=============
set "TARGET_COE_PATH=.\target\all.coe"

if "%~1"=="clean" (
    call :clean
    exit /b %ERRORLEVEL%
)

if not "%~1"=="" (
    echo Usage: build.bat [clean] 1>&2
    exit /b 1
)

python .\python\extract_pth_params.py ".\model\%MODEL_NAME%" ".\data\model_params"
if errorlevel 1 exit /b 1

rem generate memory plan / instructions
python .\python\generate_memory_plan.py "%MODEL_PY_NAME%"
if errorlevel 1 exit /b 1

python .\python\generate_instr.py
if errorlevel 1 exit /b 1

rem image to coe
rem python .\python\image_to_bram_coe.py "%IMAGE_PATH%" "%IMAGE_COE_PATH%"
rem if errorlevel 1 exit /b 1

rem interleaved weight/bias parameter coe
python .\python\params_to_bram_coe.py --memory-plan .\data\memory_plan.json --model-params .\data\model_params --out-dir .\coe
if errorlevel 1 exit /b 1

rem instr to coe
python .\python\instr_txt_to_bram_coe.py "%INSTR_PATH%" "%INSTR_COE_PATH%"
if errorlevel 1 exit /b 1

rem merge coe
python .\python\merge.py --memory-plan .\data\memory_plan.json "%TARGET_COE_PATH%"
if errorlevel 1 exit /b 1

exit /b 0

:clean
echo [CLEAN] remove generated files
if exist ".\data\model_params" rmdir /s /q ".\data\model_params"
if exist ".\data\memory_plan.json" del /q ".\data\memory_plan.json"
if exist ".\data\instr.asm" del /q ".\data\instr.asm"
if exist ".\data\instr.txt" del /q ".\data\instr.txt"
del /q ".\coe\layer*_params.coe" 2>nul
if exist ".\coe\instr.coe" del /q ".\coe\instr.coe"
if exist ".\target\all.coe" del /q ".\target\all.coe"
if exist ".\target\all.coe.map.txt" del /q ".\target\all.coe.map.txt"
for /d /r %%D in (__pycache__) do (
    if exist "%%D" rmdir /s /q "%%D"
)
echo [CLEAN] done
exit /b 0
