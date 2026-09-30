#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${TASK007_UPSTREAM_DIR:-${ROOT_DIR}/.task007/upstream}"
OUTPUT_DIR="${TASK007_OUTPUT_DIR:-${ROOT_DIR}/artifacts/task007/build}"
PATCH_FILE="${ROOT_DIR}/task007/task007-server.patch"
UPSTREAM_COMMIT="2926c06c5dc3064ae6d8db706f1a98a37cfcf3f0"
UPSTREAM_TAG="v4.1"

fail() {
    echo "Task 007 build prerequisite/error: $*" >&2
    exit 2
}

command -v git >/dev/null 2>&1 || fail "missing command: git"
[[ -f "${PATCH_FILE}" ]] || fail "missing Task 007 patch: ${PATCH_FILE}"
mkdir -p "${ROOT_DIR}/.task007"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    git clone --no-checkout https://github.com/Genymobile/scrcpy.git "${UPSTREAM_DIR}"
fi
UPSTREAM_REMOTE="$(git -C "${UPSTREAM_DIR}" remote get-url origin 2>/dev/null || true)"
case "${UPSTREAM_REMOTE}" in
    https://github.com/Genymobile/scrcpy.git|git@github.com:Genymobile/scrcpy.git) ;;
    *) fail "upstream origin is not Genymobile/scrcpy: ${UPSTREAM_REMOTE:-missing}" ;;
esac
git -C "${UPSTREAM_DIR}" cat-file -e "${UPSTREAM_COMMIT}^{commit}" \
    || fail "upstream commit is unavailable locally: ${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout --detach --force "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" reset --hard "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" clean -ffdx
[[ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)" == "${UPSTREAM_COMMIT}" ]] \
    || fail "upstream checkout is not ${UPSTREAM_COMMIT} after deterministic reset"
[[ "$(git -C "${UPSTREAM_DIR}" describe --tags --exact-match 2>/dev/null || true)" == "${UPSTREAM_TAG}" ]] \
    || fail "upstream checkout is not the exact ${UPSTREAM_TAG} tag target"
[[ -x "${UPSTREAM_DIR}/gradlew" ]] || fail "missing upstream Gradle wrapper: ${UPSTREAM_DIR}/gradlew"
[[ -z "$(git -C "${UPSTREAM_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
    || fail "upstream checkout is not clean before patch application"
[[ -z "$(git -C "${UPSTREAM_DIR}" status --porcelain=v1 --ignored)" ]] \
    || fail "ignored files remain in upstream checkout before patch application"
CHECKOUT_CLEAN_BEFORE_PATCH=PASS

git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILE}" \
    || fail "Task 007 patch does not apply cleanly to ${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" apply "${PATCH_FILE}"
PATCH_APPLY_CHECK=PASS
PATCH_APPLIED=PASS

EXPECTED_PATCH_PATHS=(
    server/src/main/java/com/genymobile/scrcpy/Server.java
    server/src/main/java/com/genymobile/scrcpy/control/Controller.java
    server/src/main/java/com/genymobile/scrcpy/device/Streamer.java
    server/src/main/java/com/genymobile/scrcpy/diagnostic/Task007Telemetry.java
    server/src/main/java/com/genymobile/scrcpy/diagnostic/Task007TimingState.java
    server/src/main/java/com/genymobile/scrcpy/video/SurfaceEncoder.java
    server/src/test/java/com/genymobile/scrcpy/diagnostic/Task007TimingStateTest.java
)
ACTUAL_PATCH_PATHS="$({
    git -C "${UPSTREAM_DIR}" diff --name-only
    git -C "${UPSTREAM_DIR}" ls-files --others --exclude-standard
} | sort)"
EXPECTED_PATCH_PATHS_SORTED="$(printf '%s\n' "${EXPECTED_PATCH_PATHS[@]}" | sort)"
[[ "${ACTUAL_PATCH_PATHS}" == "${EXPECTED_PATCH_PATHS_SORTED}" ]] \
    || fail "upstream state after patch contains paths outside Task 007"
[[ -z "$(git -C "${UPSTREAM_DIR}" status --porcelain=v1 --ignored | grep '^!!' || true)" ]] \
    || fail "ignored files remain after Task 007 patch application"
CHECKOUT_AFTER_PATCH=PASS

command -v java >/dev/null 2>&1 || fail "missing command: java (JDK runtime)"
command -v javac >/dev/null 2>&1 || fail "missing command: javac (JDK compiler)"

SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
[[ -n "${SDK_ROOT}" ]] || fail "missing Android SDK: set ANDROID_HOME or ANDROID_SDK_ROOT"
[[ -f "${SDK_ROOT}/platforms/android-36/android.jar" ]] \
    || fail "missing Android SDK platform: ${SDK_ROOT}/platforms/android-36/android.jar"
BUILD_TOOLS_DIR=""
if [[ -d "${SDK_ROOT}/build-tools" ]]; then
    BUILD_TOOLS_DIR="$(find "${SDK_ROOT}/build-tools" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
fi
[[ -n "${BUILD_TOOLS_DIR}" ]] || fail "missing Android SDK build-tools under ${SDK_ROOT}/build-tools"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
BUILD_COMMAND="./gradlew -p server assembleRelease"
(cd "${UPSTREAM_DIR}" && ${BUILD_COMMAND})
cp "${UPSTREAM_DIR}/server/build/outputs/apk/release/server-release-unsigned.apk" \
    "${OUTPUT_DIR}/scrcpy-server"
JAVA_VERSION="$(java -version 2>&1 | head -n 1)"

{
    echo "upstream_repository=https://github.com/Genymobile/scrcpy.git"
    echo "upstream_tag=${UPSTREAM_TAG}"
    echo "upstream_commit=${UPSTREAM_COMMIT}"
    echo "patch_sha256=$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"
    echo "server_sha256=$(sha256sum "${OUTPUT_DIR}/scrcpy-server" | awk '{print $1}')"
    echo "java=${JAVA_VERSION}"
    echo "gradle_wrapper=${UPSTREAM_DIR}/gradlew"
    echo "android_sdk=${SDK_ROOT}"
    echo "android_platform=${SDK_ROOT}/platforms/android-36/android.jar"
    echo "android_build_tools=${BUILD_TOOLS_DIR}"
    echo "build_command=${BUILD_COMMAND}"
    echo "checkout_clean_before_patch=${CHECKOUT_CLEAN_BEFORE_PATCH}"
    echo "patch_apply_check=${PATCH_APPLY_CHECK}"
    echo "patch_applied=${PATCH_APPLIED}"
    echo "checkout_after_patch=${CHECKOUT_AFTER_PATCH}"
} > "${OUTPUT_DIR}/build-metadata.txt"

echo "Task 007 server built: ${OUTPUT_DIR}/scrcpy-server"
cat "${OUTPUT_DIR}/build-metadata.txt"
