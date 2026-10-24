# Where the Engram tables live. Sourced by docker/entry.sh, which decides
# before the server starts, and by start.sh, whose preflight reports the same
# decision; both read the same figures from here.
#
# VLLM_ENGRAM_PLACEMENT:
#   auto  the default: ram when the host memory available to the server
#         covers NEED_RAM_GIB, ssd when it does not
#   ram   the tables held in host RAM: the fastest, and what the published
#         numbers use
#   ssd   the tables read from the checkpoint files on disk through a host row
#         cache per card: slower
#
# engram_placement sets ENGRAM_PLACEMENT (ram or ssd), ENGRAM_AVAIL_GIB (empty
# when the host memory cannot be read) and ENGRAM_LINE, one line saying what
# was chosen and why. Under ssd it also sets ENGRAM_ROW_CACHE_GIB,
# ENGRAM_PRELOAD and ENGRAM_DIRECT_READ, the values for
# VLLM_ENGRAM_ROW_CACHE_GIB, VLLM_ENGRAM_ROW_CACHE_PRELOAD and
# VLLM_ENGRAM_DIRECT_READ: yours where you set them, else chosen here. It
# exports nothing. An unknown placement or a NEED_RAM_GIB that is not a whole
# number prints why and returns 2.

# Host memory the server takes with the tables in RAM, in GiB: 202-204 GiB
# (~189 GiB of tables) plus the engine's 2 GiB margin per card.
ENGRAM_NEED_RAM_GIB_DEFAULT=212
# Under ssd the server takes ~24 GiB of host memory besides the row caches,
# and 5 GiB more is left free; the rest is split over the four cards' caches.
# A card's cache is at most its whole share of the tables (~48 GiB) and at
# least 1 GiB. Filling a cache at start reads the card's whole share from the
# SSD (~49 GB) whatever the cache size, so it is done only for a cache of a
# quarter of the share or more. Misses are read past the page cache (direct
# reads) at any size.
ENGRAM_SSD_SERVER_GIB=24
ENGRAM_SSD_MARGIN_GIB=5
ENGRAM_CARDS=4
ENGRAM_CACHE_MAX_GIB=48
ENGRAM_PRELOAD_MIN_GIB=12

# Host memory available to the server, in whole GiB: MemAvailable, or what is
# left under the memory limit of the cgroup it runs in when that is lower.
# Prints nothing when /proc/meminfo cannot be read.
engram_available_gib() {
    local meminfo="${RECIPE_MEMINFO:-/proc/meminfo}" cgroup="${RECIPE_CGROUP:-/sys/fs/cgroup}"
    local kb gib limit='' used=0 room
    kb="$(awk '$1 == "MemAvailable:" {print $2}' "$meminfo" 2>/dev/null || true)"
    case "$kb" in ''|*[!0-9]*) return 0 ;; esac
    gib=$((kb / 1048576))
    if [ -r "$cgroup/memory.max" ]; then                       # cgroup v2
        limit="$(cat "$cgroup/memory.max" 2>/dev/null || true)"
        used="$(cat "$cgroup/memory.current" 2>/dev/null || true)"
    elif [ -r "$cgroup/memory/memory.limit_in_bytes" ]; then  # cgroup v1
        limit="$(cat "$cgroup/memory/memory.limit_in_bytes" 2>/dev/null || true)"
        used="$(cat "$cgroup/memory/memory.usage_in_bytes" 2>/dev/null || true)"
    fi
    case "$used" in ''|*[!0-9]*) used=0 ;; esac
    # "max" (v2) or a 19-digit value near 2^63 (v1) means no limit.
    case "$limit" in
        ''|max|*[!0-9]*) ;;
        *)
            if [ "${#limit}" -lt 19 ]; then
                room=$(( (limit - used) / 1073741824 ))
                [ "$room" -ge 0 ] || room=0
                [ "$room" -ge "$gib" ] || gib=$room
            fi ;;
    esac
    echo "$gib"
}

engram_placement() {
    local want="${VLLM_ENGRAM_PLACEMENT:-auto}" need="${NEED_RAM_GIB:-$ENGRAM_NEED_RAM_GIB_DEFAULT}"
    local memory cache whole preload direct
    case "$want" in
        auto|ram|ssd) ;;
        *) echo "VLLM_ENGRAM_PLACEMENT must be auto, ram or ssd, not '$want'" >&2; return 2 ;;
    esac
    case "$need" in
        ''|*[!0-9]*) echo "NEED_RAM_GIB must be a whole number of GiB, not '$need'" >&2; return 2 ;;
    esac
    ENGRAM_AVAIL_GIB="$(engram_available_gib)"
    if [ -n "$ENGRAM_AVAIL_GIB" ]; then
        memory="$ENGRAM_AVAIL_GIB GiB available, RAM placement needs ~$need GiB"
    else
        memory="host memory unreadable, RAM placement needs ~$need GiB"
    fi
    if [ "$want" != auto ]; then
        ENGRAM_PLACEMENT=$want
        ENGRAM_LINE="Engram tables: $want (VLLM_ENGRAM_PLACEMENT=$want: $memory"
    elif [ -n "$ENGRAM_AVAIL_GIB" ] && [ "$ENGRAM_AVAIL_GIB" -ge "$need" ]; then
        ENGRAM_PLACEMENT=ram
        ENGRAM_LINE="Engram tables: ram (auto: $memory"
    else
        ENGRAM_PLACEMENT=ssd
        ENGRAM_LINE="Engram tables: ssd (auto: $memory"
    fi
    ENGRAM_ROW_CACHE_GIB='' ENGRAM_PRELOAD='' ENGRAM_DIRECT_READ=''
    if [ "$ENGRAM_PLACEMENT" = ram ]; then
        ENGRAM_LINE="$ENGRAM_LINE)"
        return 0
    fi
    if [ -n "${VLLM_ENGRAM_ROW_CACHE_GIB:-}" ]; then
        ENGRAM_ROW_CACHE_GIB=$VLLM_ENGRAM_ROW_CACHE_GIB
        cache="row cache $ENGRAM_ROW_CACHE_GIB GiB per card (VLLM_ENGRAM_ROW_CACHE_GIB)"
    else
        ENGRAM_ROW_CACHE_GIB=1
        if [ -n "$ENGRAM_AVAIL_GIB" ]; then
            ENGRAM_ROW_CACHE_GIB=$(( (ENGRAM_AVAIL_GIB - ENGRAM_SSD_SERVER_GIB - ENGRAM_SSD_MARGIN_GIB) / ENGRAM_CARDS ))
            [ "$ENGRAM_ROW_CACHE_GIB" -le "$ENGRAM_CACHE_MAX_GIB" ] || ENGRAM_ROW_CACHE_GIB=$ENGRAM_CACHE_MAX_GIB
            [ "$ENGRAM_ROW_CACHE_GIB" -ge 1 ] || ENGRAM_ROW_CACHE_GIB=1
        fi
        cache="row cache $ENGRAM_ROW_CACHE_GIB GiB per card"
    fi
    if [ -n "${VLLM_ENGRAM_ROW_CACHE_PRELOAD:-}" ]; then
        ENGRAM_PRELOAD=$VLLM_ENGRAM_ROW_CACHE_PRELOAD
        preload="preload $ENGRAM_PRELOAD (VLLM_ENGRAM_ROW_CACHE_PRELOAD)"
    else
        whole="${ENGRAM_ROW_CACHE_GIB%%.*}"
        case "$whole" in ''|*[!0-9]*) whole=0 ;; esac
        if [ "$whole" -ge "$ENGRAM_PRELOAD_MIN_GIB" ]; then
            ENGRAM_PRELOAD=1
            preload="filled at start"
        else
            ENGRAM_PRELOAD=0
            preload="not filled at start (under $ENGRAM_PRELOAD_MIN_GIB GiB)"
        fi
    fi
    if [ -n "${VLLM_ENGRAM_DIRECT_READ:-}" ]; then
        ENGRAM_DIRECT_READ=$VLLM_ENGRAM_DIRECT_READ
        direct="direct reads $ENGRAM_DIRECT_READ (VLLM_ENGRAM_DIRECT_READ)"
    else
        ENGRAM_DIRECT_READ=1
        direct="direct reads"
    fi
    ENGRAM_LINE="$ENGRAM_LINE; slower), $cache, $preload, $direct"
}

# Speculative decoding switched off through EXTRA_ARGS
# (`--speculative-config None`, `null` or `off`, also written with `=`).
# Without the drafter the adaptive verify depth has no state to work on and
# the server stops while capturing its CUDA graphs, and the boot check has no
# draft counters to read. drafting_off returns 0 when drafting is off; the
# entry and start.sh then clear VLLM_DSPARK_ADAPTIVE_DEPTH and run the boot
# check without its draft-counter checks.
drafting_off() {
    local -a args
    local i value
    read -ra args <<< "${EXTRA_ARGS:-}"
    for ((i = 0; i < ${#args[@]}; i++)); do
        case "${args[i]}" in
            --speculative-config=*) value="${args[i]#*=}" ;;
            --speculative-config) value="${args[i+1]:-}" ;;
            *) continue ;;
        esac
        case "${value,,}" in none|null|off|''|\'\'|'""') return 0 ;; esac
    done
    return 1
}
DRAFTING_OFF_LINE="Speculative decoding: off (EXTRA_ARGS); VLLM_DSPARK_ADAPTIVE_DEPTH cleared, boot check without draft counters"
