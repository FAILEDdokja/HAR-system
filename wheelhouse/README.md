# wheelhouse/ — offline install media (C11)

"Offline standalone system" is only literally true if a fresh machine can be
provisioned with **no network**.  This directory holds the wheels for
`requirements.lock`; the install command is:

```bash
pip install --no-index --find-links wheelhouse/ -r requirements.lock
```

## What is committed and why

Only the **small, pure-Python wheels** are committed to git (a few hundred KB
in total): flask and its dependencies, pyttsx3, and their friends — enough
for the GUI, the voice layer and the text tooling even with nothing
downloaded.

The large binary wheels (numpy ~18 MB, opencv ~90 MB, torch + CUDA family
~2.5 GB, ultralytics, PyYAML's C extension, ...) are **not** committed:
vendoring gigabytes of platform-specific binaries into a git repo makes every
clone pay for them forever, and they are reproducible from PyPI in one
command.  Before going somewhere without network, run:

```bash
bash wheelhouse/download.sh    # fills this directory from requirements.lock
```

Do that on the machine (or OS) you will demo on, so the wheels match the
platform.  After it runs, the `--no-index` command above provisions a clean
virtualenv with the network cable unplugged — that is the acceptance check
for "offline standalone".
