# Python 3.11 Environment

## Selected runtime

The active project interpreter is:

```powershell
.\.venv\Scripts\python.exe
```

The environment was created from Python 3.11.9 with
`--system-site-packages`. It currently inherits:

- PyTorch 2.5.1+cu121;
- torchvision 0.20.1+cu121;
- NumPy 2.2.6;
- Pillow 12.0.0;
- matplotlib from the host Python 3.11 installation.

The project environment additionally contains `tqdm`, `plyfile`, `imageio`,
and `imageio-ffmpeg`. It is intentionally ignored by Git.

The environment was bootstrapped with:

```powershell
& 'E:\Software\Scoop\Apps\apps\python311\current\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e '.[figures]'
.\.venv\Scripts\python.exe -m pip install tqdm plyfile imageio imageio-ffmpeg
```

`--system-site-packages` is specific to this host because its CUDA-enabled
PyTorch installation already lives in the base Python 3.11 environment. On a
different machine, install a matching CUDA PyTorch/torchvision pair inside a
clean Python 3.11 environment instead of relying on inheritance.

## Verify the runtime

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Native extension toolchain

This host uses CUDA Toolkit 12.1 (`nvcc` 12.1.66) from:

```text
E:\Software\Scoop\Apps\apps\cuda-12.1\current
```

MSVC v142/14.29 is installed in the existing Visual Studio Build Tools 2026
instance at `E:\Software\Visual Studio Build Tools\Tools`. Its compiler is
`cl` 19.29.30159, matching the MSVC 19.29 toolchain reported by the installed
PyTorch 2.5.1+cu121 build. No separate Visual Studio 2022 instance is required.

The tools are intentionally not added to the global environment. In a
`cmd.exe` shell, initialize them before a manual build or another command that
needs `nvcc` and `cl`:

```cmd
call "E:\Software\Visual Studio Build Tools\Tools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
set "CUDA_HOME=E:\Software\Scoop\Apps\apps\cuda-12.1\current"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=7.5"
set "DISTUTILS_USE_SDK=1"
where cl
where nvcc
```

In a normal, uninitialized shell,
`from torch.utils.cpp_extension import CUDA_HOME` can still report `None`.
That describes the current process environment; it does not mean the toolkit
is absent.

Both required extensions are installed in `.venv` and their compiled Python
3.11 modules import successfully:

```powershell
.\.venv\Scripts\python.exe -c "import diff_gaussian_rasterization._C, simple_knn._C; print('extensions ok')"
```

To rebuild both modules from the vendored sources, run:

```powershell
cmd /c scripts\build_cuda_extensions.cmd
```

The script pins the v142 toolset, CUDA path, RTX 2060 compute capability 7.5,
and one build worker. It directs temporary build files to
`E:\Software\Visual Studio Build Tools\Temp` and disables the pip cache so the
native build payload does not accumulate on C.

The original upstream environment pins Python 3.7, PyTorch 1.12.1, and CUDA
11.6. Python 3.11/PyTorch 2.5.1 is the selected local port and must be disclosed
with any results; it is not an exact reproduction of the upstream software
environment.
