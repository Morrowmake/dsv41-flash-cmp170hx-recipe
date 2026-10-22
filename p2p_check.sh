# Resolve peer access before the server starts (sourced by docker/entry.sh).
P2P=${P2P:-auto}
case "$P2P" in
  off|auto|force) ;;
  *) echo 'p2p: P2P must be off, auto or force' >&2; exit 2 ;;
esac
if [ "${DRY:-0}" = 1 ]; then
  # A dry run never initializes CUDA and cannot claim a content-check pass.
  if [ "$P2P" = force ] && { [ "$TP" -gt 1 ] || [ "$PP" -gt 1 ]; }; then
    P2P_ENABLED=1
    echo '[p2p] P2P enabled: P2P=force; content check bypassed (dry run)'
  else
    P2P_ENABLED=0
    echo "[p2p] P2P disabled for dry run: mode=$P2P; auto checks before a real multi-GPU start"
  fi
else
  P2P_ENABLED="$("$VENV/bin/python" "$REPO_ROOT/p2p_check.py" \
    --mode "$P2P" --tp "$TP" --pp "$PP")" || {
      P2P_ENABLED=0
      echo '[p2p] P2P disabled: check unavailable' >&2
    }
fi
# After a pass the release launch (serve.sh) sets the peer transports: the
# custom all-reduce over PCIe peer-to-peer and NCCL_P2P_LEVEL=SYS. Otherwise
# every peer transport is turned off explicitly; serve.sh keeps a value it
# inherits and would put its own back for a name left unset.
if [ "$P2P_ENABLED" = 1 ]; then
  unset NCCL_P2P_DISABLE
else
  export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0
  export NCCL_P2P_LEVEL=LOC
  export NCCL_P2P_DISABLE=1
fi
