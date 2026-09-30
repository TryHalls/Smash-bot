# Task 007 diagnostic server

This directory contains only the reproducible Task 007 patch and build
metadata. The full scrcpy checkout and generated APK stay under the ignored
`.task007/` and `artifacts/task007/` directories.

## Pinned upstream

- Repository: `https://github.com/Genymobile/scrcpy.git`
- Annotated tag: `v4.1`
- Tag target commit: `2926c06c5dc3064ae6d8db706f1a98a37cfcf3f0`
- Tag object: `49c9501fb26f456bbf4a341dd68879f670c67452`
- Upstream server build: `./gradlew -p server assembleRelease`
- Gradle wrapper: Gradle `9.3.1`, wrapper distribution SHA-256
  `b266d5ff6b90eada6dc3b20cb090e3731302e553a27c5d3e4df1f0d76beaff06`

The patch is applied only to a clean checkout of that exact commit. It does
not vendor scrcpy or replace the official production server.

## Build

The build is intentionally fail-closed. It requires `git`, a JDK providing
both `java` and `javac`, `ANDROID_HOME` or `ANDROID_SDK_ROOT`, the Android
SDK platform `android-36`, and Android build-tools. It does not install or
select substitutes.

```sh
./task007/build_server.sh
```

The script performs `git apply --check`, applies
`task007/task007-server.patch`, runs the upstream server command exactly, and
writes the APK plus `build-metadata.txt` under the ignored artifact directory.
Before patching it force-checks out the pinned commit, resets tracked state,
removes all untracked and ignored files, and records
`checkout_clean_before_patch=PASS`. After patching it requires the exact set
of Task 007 paths and records `patch_apply_check=PASS`,
`patch_applied=PASS`, and `checkout_after_patch=PASS`.

`task007_server_identity()` accepts an APK only beside metadata proving the
pinned upstream commit/tag, the SHA-256 of the current repository patch, the
exact upstream build command, all clean/apply state checks, and an APK
SHA-256 equal to the selected file. There is deliberately no hardcoded APK
SHA before the first successful diagnostic build; missing or malformed
metadata, arbitrary APKs, and any mismatch fail closed.

The local build gate was completed with the project-local JDK 17.0.20.1 and
Android SDK platform `android-36` / build-tools `36.0.0`; no global Java or
SDK was changed. The resulting diagnostic server SHA-256 is
`45999e50af2365a08391c4393366d8669a9c21db5debf2b70265103615dfe7b5`.
The complete recorded provenance is in the ignored
`artifacts/task007/build/build-metadata.txt`.

## Diagnostic wire extension

The custom server retains the official 12-byte scrcpy framed-video header and
its `payload_size` meaning. It inserts exactly one 56-byte big-endian `T7TM`
sidecar between that header and the H.264/config payload. The host parser in
`smashbot_diagnostics.task007` validates the sidecar, preserves its exact
packet observation timestamps, strips it before the existing CONFIG merger and
FFmpeg FIFO, and rejects invalid magic, version, flags, reserved bits, or
device-clock ordering.

The official Task 005/006 parser and raw-H.264 production path remain
unchanged. The diagnostic source is selected explicitly as
`Task007FramedH264FrameSource`. Physical evidence is preserved in the ignored
`artifacts/task007/20260930T162759Z/report.json`; its formal `stage_b` result
is `INCONCLUSIVE` because the generic classifier requires 30 trials, while
the completed pilot itself had 10/10 causally valid trials and no pipeline
errors.
