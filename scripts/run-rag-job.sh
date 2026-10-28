#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  printf 'usage: run-rag-job.sh <purpose> <seconds> <checkpoint> <command> [args...]\n' >&2
  exit 2
fi

purpose="$1"
seconds="$2"
checkpoint="$3"
shift 3
case "$seconds" in
  ''|*[!0-9]*) printf 'seconds must be a positive integer\n' >&2; exit 2 ;;
esac
if [ "$seconds" -lt 1 ]; then
  printf 'seconds must be a positive integer\n' >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log="${NEXTLAB_RAG_JOB_LOG:-$repo_root/.rag/job.tsv}"
mkdir -p "$(dirname "$log")"
if [ ! -s "$log" ]; then
  printf 'owner\tpurpose\tpid\tstart_utc\tdeadline_utc\tcheckpoint\n' >> "$log"
fi

owner="$(id -un) via scripts/run-rag-job.sh"
start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
deadline="$(date -u -d "+${seconds} seconds" +%Y-%m-%dT%H:%M:%SZ)"
exec timeout -k 15s "${seconds}s" bash -c '
  owner="$1"
  purpose="$2"
  start="$3"
  deadline="$4"
  checkpoint="$5"
  log="$6"
  shift 6
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$owner" "$purpose" "$$" "$start" "$deadline" "$checkpoint" >> "$log"
  exec "$@"
' run-rag-job "$owner" "$purpose" "$start" "$deadline" "$checkpoint" "$log" "$@"
