# Task 009 Python/OpenCV unblock plan

The current host Python is `/usr/bin/python3` 3.11.2. It can show `venv`
help but cannot create an environment because `ensurepip` is unavailable; no
`uv`, `virtualenv`, `pip`, `pip3`, or `pipx` executable is present. No system
change was made during Task 009.

## Minimum human action

After reviewing the storage budget, a human with package-install authority can
install the distribution support package (the exact package name is expected
to be `python3.11-venv` on this Debian-family host):

```bash
sudo apt-get install python3.11-venv
```

This changes the system package database and installs the distribution's
Python venv/ensurepip support. It was not executed by Codex. Verify the
package manager result and disk space, then verify without touching the phone:

```bash
python3 -m ensurepip --version
python3 -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip --version
```

The isolated environment can be removed by its owner with `rm -rf .venv`
only after confirming that no work is using it; this does not revert the
system package installation. The package name and final footprint should be
confirmed by `apt` on the host before proceeding. Check `df -B1 /` before and
after every install and stop if free space approaches the Task 009 floor.

## Deferred smoke test

Do not install OpenCV until the venv gate passes. In the venv, install only
the approved headless wheel and let its resolver choose NumPy:

```bash
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: \
  opencv-python-headless==4.14.0.94
.venv/bin/python -m pip freeze
.venv/bin/python - <<'PY'
import cv2, numpy
print("numpy", numpy.__version__, numpy.__file__)
print("opencv", cv2.__version__, cv2.__file__)
a = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
a[8:24, 8:24] = (255, 255, 255)
gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 3)
flow, status, error = cv2.calcOpticalFlowPyrLK(gray, gray, numpy.float32([[[4, 4]]]), None)
affine, inliers = cv2.estimateAffinePartial2D(numpy.float32([[0, 0], [1, 0], [0, 1]]), numpy.float32([[0, 0], [1, 0], [0, 1]]))
warp = numpy.eye(2, 3, dtype=numpy.float32)
ecc = cv2.findTransformECC(gray, gray, warp, cv2.MOTION_TRANSLATION)
kernel = numpy.ones((3, 3), dtype=numpy.uint8)
morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
components = cv2.connectedComponentsWithStats((morph > 0).astype(numpy.uint8))
print("ok", gray.shape, corners is None or corners.shape, flow.shape, affine.shape, ecc[0], components[0])
PY
```

Record the venv size, free bytes before/after, import paths and exact versions.
This smoke test uses only synthetic arrays; it is not a detector tune and must
not inspect Task 008 gameplay until the architecture review authorizes it.
