#!/bin/bash
# Queue-aware partition picker — ALWAYS prefer a partition that schedules NOW over a backlogged one.
#
# Standing rule (user directive 2026-06-29): never submit onto a partition with a long waitlist when an
# equivalent one is idle. This helper ranks a candidate list by (pending jobs ASC, idle nodes DESC) --
# queue depth dominates because that IS the waitlist -- after filtering to nodes that meet a minimum
# per-node RAM, and echoes the winner. Source it or call it:
#
#   PART=$(scripts/_pick_partition.sh 192000 mpi priority bigmem day week)   # CPU big-RAM fit
#   PART=$(scripts/_pick_partition.sh 0 gpu_devel scavenge_gpu)              # GPU (min-mem 0)
#   sbatch --partition="$PART" scripts/_fit_transport.sh
#
# Args: MINMEM_MB  CANDIDATE...   (candidates in rough preference order; ties broken by preference).
# Prints the chosen partition to stdout and a one-line rationale to stderr. Falls back to the first
# candidate if sinfo/squeue are unavailable.
set -u
MINMEM=${1:-0}; shift || true
CANDS=("$@")
[ ${#CANDS[@]} -eq 0 ] && { echo "scavenge_gpu"; exit 0; }

best=""; best_idle=-1; best_pend=1000000; rank=0
for p in "${CANDS[@]}"; do
    rank=$((rank + 1))
    # max per-node RAM available in this partition (MB); skip if it can't hold the job
    mx=$(sinfo -h -p "$p" -o "%m" 2>/dev/null | tr -d '+' | sort -n | tail -1)
    [ -z "$mx" ] && continue
    [ "$mx" -lt "$MINMEM" ] && continue
    idle=$(sinfo -h -p "$p" -t idle -o "%D" 2>/dev/null | paste -sd+ | bc 2>/dev/null); idle=${idle:-0}
    pend=$(squeue -h -t PD -p "$p" -o "%i" 2>/dev/null | wc -l)
    # queue depth (pending) IS the waitlist -> minimize it first; break ties by more idle nodes, then
    # by preference order (first candidate wins a full tie since we only replace on a strict improvement)
    if [ "$pend" -lt "$best_pend" ] || { [ "$pend" -eq "$best_pend" ] && [ "$idle" -gt "$best_idle" ]; }; then
        best="$p"; best_idle="$idle"; best_pend="$pend"
    fi
done

if [ -z "$best" ]; then best="${CANDS[0]}"; echo "[pick_partition] no candidate met ${MINMEM}MB; falling back to $best" >&2
else echo "[pick_partition] -> $best (idle=$best_idle pending=$best_pend; min_mem=${MINMEM}MB)" >&2; fi
echo "$best"
