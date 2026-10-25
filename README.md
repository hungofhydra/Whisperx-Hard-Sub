# WhisperX Hardsub Studio

A desktop app that takes one or more Japanese audio files, transcribes them
with WhisperX, and produces a single small mp4 with:

- **Hardsubs** burned in (styled, semi-transparent box behind the text)
- A **static background image** for the whole duration (you choose it)
- **Chapters** (visible in players like VLC/mpv/most phone players), one
  per input file, named after that file's filename
- Multiple input files automatically **merged in the order you arrange them**
- File size kept small on purpose: low fps (since the image never moves),
  a high CRF, and low-bitrate mono audio — you said audio quality doesn't
  matter, so this leans hard into that.

Built and tested logic here in a Linux sandbox (I don't have a Windows/GPU
machine to hand you a finished `.exe` directly) — the ffmpeg pipeline
(merge → subtitle burn → chapters → render) was run end-to-end and verified
to work correctly. What's left is: install WhisperX's ML dependencies on
your machine and run one PyInstaller command. That's normal for apps like
this — CUDA/torch builds are always compiled for the exact machine they'll
run on, not cross-compiled, so it genuinely has to happen on your PC.

---

## 1. One-time setup (on your Windows PC with the NVIDIA GPU)

1. Install **Python 3.10 or 3.11** (64-bit) from python.org, and **ffmpeg**
   (e.g. via `winget install ffmpeg` or from https://www.gyan.dev/ffmpeg/builds/,
   then make sure `ffmpeg.exe` is on your PATH — check with `ffmpeg -version`
   in a new terminal). If you skip this, the app falls back to a bundled
   ffmpeg from the `imageio-ffmpeg` package automatically, but a real system
   ffmpeg install is more reliable and usually faster.

2. Open a terminal in this project folder and create a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install PyTorch **with CUDA support first** — this is the step people
   most often get wrong by just running `pip install torch`, which gives
   you the CPU-only build. Go to https://pytorch.org/get-started/locally/,
   pick your CUDA version (check yours with `nvidia-smi`), and run the
   command it gives you, e.g. for CUDA 12.1:
   ```powershell
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. Now install everything else:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Re-assert the CUDA build of torch, torchvision, and torchaudio.**
   `whisperx` and `transformers` pin/pull specific versions, and installing
   in step 4 will often silently overwrite your CUDA build from step 3
   with plain CPU wheels from PyPI — even though step 3 was correct. Always
   run this immediately after step 4, every time:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   ```
   (Deliberately no `--no-deps` here — torch, torchvision, and torchaudio
   must be mutually compatible builds, and letting pip resolve all three
   together from the same index guarantees that. Installing them
   separately/inconsistently causes cryptic errors like
   `operator torchvision::nms does not exist`.)

   Then confirm it actually stuck:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torchvision; print(torchvision.__version__)"
   ```
   The first must print `True`; the second must print a version with no
   error. If either fails, re-run the reinstall line above. (If
   `torch.cuda.is_available()` is still `False` after that, it's a
   driver/CUDA-version mismatch: run `nvidia-smi` and make sure the CUDA
   version in the `--index-url` above is ≤ what your driver reports.)

6. Run it:
   ```powershell
   python app.py
   ```
   The very first transcription run will download the WhisperX model
   weights (a few GB for `large-v3`) — this needs an internet connection
   once; after that it's cached locally and works offline.

If step 6 works and you can produce a video, you're ready to build the exe.

---

## 2. Building the `.exe`

Still inside the activated venv:

```powershell
pyinstaller build.spec
```

This produces `dist/WhisperXHardsubStudio/` — a folder containing
`WhisperXHardsubStudio.exe` plus all its dependencies (this will be several
GB because of CUDA/torch; that's expected and normal for local ML apps).
**Zip and ship the whole folder**, not just the .exe — the .exe needs the
files next to it. Double-clicking `WhisperXHardsubStudio.exe` inside that
folder launches the GUI, no Python install needed on the machine you copy
it to (as long as that machine also has a compatible NVIDIA GPU + driver,
or you set the app to CPU mode).

**If the build fails or the exe errors on startup / on Start:** ML packages
in this stack (torch/ctranslate2/pyannote/transformers/hydra/omegaconf) do a
lot of *dynamic* imports (importing a class by string name from a config
file) that PyInstaller's static analysis can't see, so it's common to hit
`ModuleNotFoundError` for something not obviously related to your code.
`build.spec` currently has `console=True` for exactly this reason — a real
console window stays open and prints the full Python traceback, which
names the exact missing module. Once you have that name, add its package
to the `for pkg in [...]` collect-loop near the top of `build.spec` and
rebuild. When everything works reliably, you can flip `console=True` back
to `False` for a cleaner windowed app.

---

## 3. Using the app

1. **Add files…** — pick your audio/video files (all `.wav`, all `.mp3`, or
   all `.mp4`, not mixed). `.mp4` works too — the audio track is extracted
   automatically, whatever video it contains is ignored. Use **Move
   up/down** to set the order; that order becomes both
   the playback order and the chapter order.
2. Pick an **Output type**: video with hardsub (default), or **subtitle
   file only (.srt)** — this skips the background image and all
   video/audio rendering entirely, running only transcription and writing
   a plain `.srt`. Since `.srt` has no native chapter metadata, each
   chapter boundary is inserted as its own short cue showing that file's
   name.
3. If making a video: pick a **background image**.
4. Pick where to save the output file.
4. Adjust settings if you want (defaults are sane):
   - **Model size**: `large-v3` is most accurate; `medium`/`small` are
     faster if accuracy is less critical.
   - **Device**: `cuda` (your GPU) or `cpu`.
   - **Precise alignment** (checked by default): re-times each subtitle
     line against the audio for tighter sync, using a small Japanese
     wav2vec2 model downloaded from Hugging Face the first time it's used.
     If that download fails (offline, blocked network, firewall/proxy
     issues), uncheck this — the app will fall back to Whisper's own
     segment timestamps instead, which are still generally fine, just a
     little less tightly synced to speech onset/offset.
   - **fps / CRF / resolution / audio bitrate**: control output size.
     Lower fps and higher CRF = smaller file. 5 fps is already very
     aggressive and looks fine since the picture never moves — subtitles
     just update a few times a second.
5. Click **Start**. Progress and ffmpeg/WhisperX logs stream into the
   log box at the bottom. **Cancel** stops after the current step.

## Project layout

```
app.py              GUI (PySide6)
core/
  transcribe.py      WhisperX wrapper (ASR + Japanese alignment)
  subtitles.py        .ass subtitle generation (styled box)
  chapters.py          ffmpeg chapter metadata generation
  pipeline.py           orchestrates: probe -> transcribe -> merge -> render
  utils.py               ffmpeg process handling, path/time helpers
requirements.txt
build.spec           PyInstaller spec (onedir build)
```

## Notes / things you can tune later

- **"Same format" requirement**: the app checks all input files share one
  extension and stops with a clear error if not, per your spec.
- **Why an onedir build, not onefile**: onefile re-extracts several GB into
  a temp folder every time you launch, which is slow and can trip AV
  scanners. A zipped onedir folder is the standard approach for
  PyTorch-based desktop apps.
- **Font**: the styled box uses "Noto Sans JP" by default; if that font
  isn't installed on the render machine, libass automatically substitutes
  a font that covers the Japanese glyphs (verified in testing), so
  subtitles won't come out blank — but for best looks, install Noto Sans
  JP on the build/run machine.
