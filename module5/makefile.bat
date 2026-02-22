@echo off
setlocal

REM Compile
nvcc -O2 -std=c++14 assignment.cu -o assignment.exe
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

REM Run variations
echo.
echo === Run: no args ===
assignment.exe

echo.
echo === Run: -threads 1024 ===
assignment.exe -threads 1024

echo.
echo === Run: -num_blocks 120000 ===
assignment.exe -num_blocks 120000

endlocal