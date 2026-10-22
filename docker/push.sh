#!/usr/bin/env bash
# Publish an already reviewed OCI image; the release token arrives on stdin.
set -euo pipefail
set +x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '[image-push] enabled=%s\n' "${IMAGE_PUBLISH:-0}" >&2
[ "${IMAGE_PUBLISH:-0}" = 1 ] || { echo "Set IMAGE_PUBLISH=1 only for an approved release" >&2; exit 2; }
[ "$#" = 4 ] || { echo "Usage: push.sh OCI TARGET EXPECTED_DIGEST GPU_ACCEPTANCE_JSON < token-file" >&2; exit 2; }
oci="$1"; target="$2"; expected="$3"; acceptance="$4"
case "$target" in ghcr.io/morrowmake/vllm-cmp170hx:*) ;; *) echo "Unexpected publication target" >&2; exit 2 ;; esac
python3 - "$oci" "$expected" "$acceptance" "$target" "$HERE" <<'CHECK'
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[5])
from image_identity import check_config
from oci_layers import read_blob
root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
index = json.loads((root / "index.json").read_text())
if len(index["manifests"]) != 1 or index["manifests"][0]["digest"] != expected:
    raise SystemExit("OCI manifest digest mismatch")
for path in (root / "blobs/sha256").iterdir():
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    if h.hexdigest() != path.name:
        raise SystemExit("OCI blob integrity mismatch")
manifest = read_blob(root, index["manifests"][0])
config = read_blob(root, manifest["config"])
commit = check_config(config, sys.argv[4])
record = json.loads(pathlib.Path(sys.argv[3]).read_text())
if record.get("engine_commit") != commit:
    raise SystemExit("GPU acceptance engine commit missing or mismatched")
if record["image_digest"] != expected or record["passed"] is not True:
    raise SystemExit("GPU acceptance missing or mismatched")
if record["artifact_audit_passed"] is not True:
    raise SystemExit("Artifact audit missing")
CHECK
CRANE="${CRANE:-crane}"
private_config="$(mktemp -d)"
chmod 700 "$private_config"
export DOCKER_CONFIG="$private_config"
trap 'rm -rf "$private_config"' EXIT
# No token in argv, environment, source, image, attestation or command output.
"$CRANE" auth login ghcr.io --username Morrowmake --password-stdin >/dev/null
"$CRANE" push "$oci" "$target"
actual="$("$CRANE" digest "$target")"
[ "$actual" = "$expected" ] || { echo "Published digest mismatch" >&2; exit 1; }
printf 'Published %s@%s\n' "${target%:*}" "$actual"
