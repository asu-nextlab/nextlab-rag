#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/index-capped.sh

Build or resume the local RAG index in a user systemd service. The service has
a 2 GB memory cap, no swap, one native CPU thread, a 300-second watchdog, and
a 90-minute runtime limit. Set NEXTLAB_RAG_PYTHON to select another Python
interpreter. The default is .venv/bin/python in the repository.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit="nextlab-rag-memory-rebuild.service"
checkpoint="$repo_root/.rag/staging/manifest.json"

if [[ "${1:-}" == "--service-exec" ]]; then
  shift
  if [[ "$#" -lt 1 ]]; then
    printf 'missing Python command for service execution\n' >&2
    exit 2
  fi
  job_log="${NEXTLAB_RAG_JOB_LOG:-$repo_root/.rag/job.tsv}"
  mkdir -p "$(dirname "$job_log")"
  if [[ ! -s "$job_log" ]]; then
    printf 'owner\tpurpose\tpid\tstart_utc\tdeadline_utc\tcheckpoint\n' >> "$job_log"
  fi
  owner="$(id -un) via scripts/index-capped.sh"
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  deadline="$(date -u -d '+5400 seconds' +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$owner" 'capped local RAG index build' "$$" "$start" "$deadline" "$checkpoint" >> "$job_log"
  exec "$@"
fi

if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 2
fi

python_bin="${NEXTLAB_RAG_PYTHON:-$repo_root/.venv/bin/python}"
if [[ "$python_bin" != /* ]]; then
  python_bin="$repo_root/$python_bin"
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python interpreter is not executable: %s\n' "$python_bin" >&2
  exit 1
fi
for command_name in systemd-run systemctl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'required command is missing: %s\n' "$command_name" >&2
    exit 1
  fi
done

state="$(systemctl --user show "$unit" --property=ActiveState --value 2>/dev/null || true)"
case "$state" in
  active|activating|reloading|deactivating)
    printf 'index service is already %s: %s\n' "$state" "$unit" >&2
    exit 1
    ;;
esac

exec systemd-run --user --wait --collect --unit=nextlab-rag-memory-rebuild \
  --property=Type=notify --property=NotifyAccess=main \
  --property="WorkingDirectory=$repo_root" \
  --property=MemoryMax=2000000000 --property=MemorySwapMax=0 \
  --property=OOMPolicy=stop \
  --property=WatchdogSec=300s --property=WatchdogSignal=SIGTERM \
  --property=TimeoutStartSec=300s --property=TimeoutStopSec=15s \
  --property=RuntimeMaxSec=5400s --property=KillMode=control-group \
  --property=LimitCORE=0 \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=ORT_NUM_THREADS=1 \
  "$repo_root/scripts/index-capped.sh" --service-exec \
  "$python_bin" -u -m nextlab_rag index --repo-root "$repo_root" --batch-size 32
