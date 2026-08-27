@echo off
setlocal

set "REPO_ROOT=%~dp0.."
if not defined VS_BUILD_TOOLS set "VS_BUILD_TOOLS=E:\Software\Visual Studio Build Tools\Tools"
if not defined CUDA_HOME set "CUDA_HOME=E:\Software\Scoop\Apps\apps\cuda-12.1\current"

set "VCVARS=%VS_BUILD_TOOLS%\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%".
    exit /b 1
)
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo ERROR: CUDA 12.1 nvcc not found under "%CUDA_HOME%".
    exit /b 1
)
if not exist "%PYTHON%" (
    echo ERROR: Python environment not found at "%PYTHON%".
    exit /b 1
)

set "BUILD_TEMP=%VS_BUILD_TOOLS%\..\Temp"
if not exist "%BUILD_TEMP%" mkdir "%BUILD_TEMP%"
set "TEMP=%BUILD_TEMP%"
set "TMP=%BUILD_TEMP%"
set "PIP_NO_CACHE_DIR=1"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%CUDA_HOME%\bin;%PATH%"

call "%VCVARS%" -vcvars_ver=14.29
if errorlevel 1 exit /b %errorlevel%

where cl
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=7.5"
if not defined MAX_JOBS set "MAX_JOBS=1"

"%PYTHON%" -m pip install --no-build-isolation --no-deps -v "%REPO_ROOT%\submodules\simple-knn"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m pip install --no-build-isolation --no-deps -v "%REPO_ROOT%\submodules\depth-diff-gaussian-rasterization"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -c "import diff_gaussian_rasterization, simple_knn; print('CUDA extensions imported successfully')"
