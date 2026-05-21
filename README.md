# ERPM

To run ERPM, from the repo root, drop into the SMART container with this folder mounted at `/repo`:

```bash
docker run -it --rm -v "$(pwd)":/repo ghcr.io/rangamanilabucsd/smart:latest
```