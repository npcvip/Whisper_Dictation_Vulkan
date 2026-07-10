whisper.cpp - Windows x64, Vulkan GPU backend
=============================================
Prebuilt whisper-server + ggml/whisper DLLs with the Vulkan backend
(-DGGML_VULKAN=ON). One build serves any Vulkan-capable GPU (AMD / NVIDIA / Intel).

Source: https://github.com/ggml-org/whisper.cpp  (unmodified)
Commit: 610e664ba7cfe3af46125ed1b5a1184fccb51bcd
        610e664 whisper : catch C++ exceptions in whisper_init_with_params_no_state (#3831)
Built:  2026-06-03  (LunarG Vulkan SDK 1.4.350.0, MSVC, Ninja, Release)

Runtime requirements on target:
  - A Vulkan-capable GPU + driver (provides vulkan-1.dll)
  - Microsoft Visual C++ Redistributable (x64)

Run (example):
  whisper-server.exe -m ggml-large-v3-turbo.bin --host 127.0.0.1 --port 8080 -t 6 -l en

License: MIT (see LICENSE-whisper.cpp.txt)
