#!/bin/sh
# Build the LD_AUDIT shim. Needs gcc and glibc headers; no Level Zero headers.
set -e
cd "$(dirname "$0")"
gcc -shared -fPIC -O2 -Wall -o ze_residency_audit.so ze_residency_audit.c -lpthread
echo "built $(pwd)/ze_residency_audit.so"
