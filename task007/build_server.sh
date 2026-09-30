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

[[ -f "${PATCH_FILE}" ]] || fail "missing Task 007 patch: ${PATCH_FILE}"
mkdir -p "${ROOT_DIR}/.task007"
if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    git clone --no-checkout https://github.com/Genymobile/scrcpy.git "${UPSTREAM_DIR}"
    git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
fi
[[ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)" == "${UPSTREAM_COMMIT}" ]] \
    || fail "upstream checkout is not ${UPSTREAM_COMMIT}"
[[ "$(git -C "${UPSTREAM_DIR}" describe --tags --exact-match 2>/dev/null || true)" == "${UPSTREAM_TAG}" ]] \
    || fail "upstream checkout is not the exact ${UPSTREAM_TAG} tag target"
[[ -x "${UPSTREAM_DIR}/gradlew" ]] || fail "missing upstream Gradle wrapper: ${UPSTREAM_DIR}/gradlew"

if git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILE}"; then
    git -C "${UPSTREAM_DIR}" apply "${PATCH_FILE}"
elif git -C "${UPSTREAM_DIR}" apply --reverse --check "${PATCH_FILE}"; then
    : # already applied to this exact temporary checkout
else
    fail "Task 007 patch does not apply cleanly to ${UPSTREAM_COMMIT}"
fi

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
    echo "patch_apply_check=PASS"
} > "${OUTPUT_DIR}/build-metadata.txt"

echo "Task 007 server built: ${OUTPUT_DIR}/scrcpy-server"
cat "${OUTPUT_DIR}/build-metadata.txt"
