@echo off
REM build_3dgs.bat - build the 3 CUDA extensions of 3DGS into E:\lkenv
REM needs E:\VS\BuildTools (MSVC) and E:\hetero\cuda (conda-forge nvcc 12.8 + headers + glm)
setlocal
set CODE=%~dp0
set PY=E:\lkenv\Scripts\python.exe
set CUDA_HOME=E:\hetero\cuda\Library
set CUDA_PATH=E:\hetero\cuda\Library
set TEMP=E:\hetero\tmp
set TMP=E:\hetero\tmp
if not exist E:\hetero\tmp mkdir E:\hetero\tmp
call "E:\VS\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || goto :err
set PATH=%CUDA_HOME%\bin;E:\hetero\cuda\Library\bin;%PATH%
set DISTUTILS_USE_SDK=1
set LIB=%CUDA_HOME%\lib;%LIB%
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=16
REM torch looks for CUDA_HOME\lib\x64; conda puts libs in lib\ -> junction
if not exist "%CUDA_HOME%\lib\x64" mklink /J "%CUDA_HOME%\lib\x64" "%CUDA_HOME%\lib"
REM copy glm headers where the rasterizer expects them
if not exist "%CODE%gs\submodules\diff-gaussian-rasterization\third_party\glm\glm\glm.hpp" (
    mkdir "%CODE%gs\submodules\diff-gaussian-rasterization\third_party\glm" 2>nul
    xcopy /E /I /Y "E:\hetero\cuda\Library\include\glm" "%CODE%gs\submodules\diff-gaussian-rasterization\third_party\glm\glm" >nul
)
where nvcc
nvcc --version
where cl
%PY% -m pip install --no-build-isolation "%CODE%gs\submodules\simple-knn" || goto :err
%PY% -m pip install --no-build-isolation "%CODE%gs\submodules\diff-gaussian-rasterization" || goto :err
%PY% -m pip install --no-build-isolation "%CODE%gs\submodules\fused-ssim" || echo fused-ssim build failed (optional)
%PY% -c "import simple_knn._C, diff_gaussian_rasterization; print('3DGS extensions OK')"
echo BUILD_DONE
goto :eof
:err
echo BUILD_FAILED
exit /b 1
