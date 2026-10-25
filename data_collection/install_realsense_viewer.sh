#!/usr/bin/env bash
# Install the official RealSense Viewer on this Ubuntu 24.04 system.
set -euo pipefail

source /etc/os-release
if [[ "${ID}" != ubuntu || "${VERSION_CODENAME}" != noble ]]; then
    echo "This installer targets Ubuntu 24.04 (noble)." >&2
    exit 1
fi

sudo -v
installer_tmp=$(mktemp -d)
trap 'rm -rf "$installer_tmp"' EXIT
curl -fsSL https://librealsense.realsenseai.com/Debian/librealsenseai.asc \
    -o "$installer_tmp/librealsenseai.asc"
gpg --batch --dearmor --output "$installer_tmp/librealsenseai.gpg" \
    "$installer_tmp/librealsenseai.asc"
sudo install -d -m 0755 /etc/apt/keyrings
sudo install -m 0644 "$installer_tmp/librealsenseai.gpg" /etc/apt/keyrings/librealsenseai.gpg
printf '%s\n' 'deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo noble main' \
    | sudo tee /etc/apt/sources.list.d/librealsense.list >/dev/null
sudo apt-get update
# Runtime dependencies include device permission rules. Keep existing kernel drivers.
sudo apt-get install -y librealsense2-utils
realsense-viewer --version
echo "Reconnect the camera, then launch: realsense-viewer"
