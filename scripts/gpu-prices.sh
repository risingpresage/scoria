#!/usr/bin/env bash
# Live AMD GPU rental prices for scoria's multi-arch test matrix.
#
# Every run reads public price sources, so no figure shown is older than this
# machine's last fetch (the per-row PRICE-AS-OF column tells you how old each
# upstream quote is). Results are cached to harness/gpu-prices.json with a
# fetched_at stamp; if the network is down the cache is reused and its age is
# printed as a warning.
#
#   ./scripts/gpu-prices.sh                 # table, cheapest first
#   ./scripts/gpu-prices.sh --gpu 9070      # filter by model name
#   ./scripts/gpu-prices.sh --json          # normalized JSON to stdout
#   ./scripts/gpu-prices.sh --refresh       # ignore cache, always fetch
#   ./scripts/gpu-prices.sh --all           # every card any source carries
#
# Sources (all public, no credentials):
#   aggregate  givejensenmoney.app/api/prices   refreshed hourly from provider APIs
#   azure      prices.azure.com/api/retail      Azure's own retail price API
#   consumer   computeprices.com/gpus/<card>    static table, Salad-only AMD cards
#
# The consumer cards are the ones that actually map to gfx1201/gfx1200/gfx1100,
# so they are fetched separately; that source is a scrape and degrades to a
# warning rather than failing the run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${SCORIA_PRICE_CACHE:-$ROOT/harness/gpu-prices.json}"
MAX_AGE_H="${SCORIA_PRICE_MAX_AGE_H:-6}"

PATTERN='MI[0-9]{3}|RX [0-9]{4}|Radeon PRO'
MODE=table
FORCE=0

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --json) MODE=json ;;
    --all) PATTERN='.' ;;
    --gpu) shift; PATTERN="${1:?--gpu needs a pattern}" ;;
    --gpu=*) PATTERN="${1#--gpu=}" ;;
    --refresh) FORCE=1 ;;
    --cache) shift; CACHE="${1:?--cache needs a path}" ;;
    --max-age-hours) shift; MAX_AGE_H="${1:?--max-age-hours needs a number}" ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
  shift
done

die() { echo "gpu-prices: $*" >&2; exit 1; }
command -v curl >/dev/null || die "curl not found"
command -v jq   >/dev/null || die "jq not found"

cache_age_h() {
  [ -f "$CACHE" ] || { echo 999999; return; }
  local ts now
  ts="$(jq -r '.fetched_at // empty' "$CACHE" 2>/dev/null)"
  [ -n "$ts" ] || { echo 999999; return; }
  now="$(date -u +%s)"
  ts="$(date -u -d "$ts" +%s 2>/dev/null || echo 0)"
  awk -v n="$now" -v t="$ts" 'BEGIN { printf "%d", (n - t) / 3600 }'
}

need_fetch=1
age="$(cache_age_h)"
if [ "$FORCE" -eq 0 ] && [ "$age" -lt "$MAX_AGE_H" ]; then
  need_fetch=0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ "$need_fetch" -eq 1 ]; then
  echo "gpu-prices: fetching live prices..." >&2

  curl -fsS --max-time 40 "https://www.givejensenmoney.app/api/prices" \
    -o "$tmp/gj.json" 2>"$tmp/gj.err" &
  gj_pid=$!

  curl -fsS --max-time 60 -G "https://prices.azure.com/api/retail/prices" \
    --data-urlencode "\$filter=serviceName eq 'Virtual Machines' and contains(armSkuName,'_MI') and priceType eq 'Consumption'" \
    -o "$tmp/az.json" 2>"$tmp/az.err" &
  az_pid=$!

  (
    set -e
    : > "$tmp/consumer.raw"
    printf '[' > "$tmp/cards.json"
    first=1
    for pair in "rx-9070-xt:RX 9070 XT" "rx-9060-xt:RX 9060 XT" \
                "rx-7900-xtx:RX 7900 XTX" "rx-7900-xt:RX 7900 XT" \
                "rx-7800-xt:RX 7800 XT"; do
      slug="${pair%%:*}"; name="${pair#*:}"
      if curl -fsS --max-time 30 -A "Mozilla/5.0 (compatible; scoria-price-board)" \
           "https://computeprices.com/gpus/$slug" -o "$tmp/$slug.html" 2>/dev/null; then
        [ "$first" -eq 1 ] || printf ',' >> "$tmp/cards.json"
        printf '{"path":"%s","slug":"%s","gpu":"%s"}' "$tmp/$slug.html" "$slug" "$name" >> "$tmp/cards.json"
        first=0
      fi
    done
    printf ']' >> "$tmp/cards.json"
  ) 2>"$tmp/consumer.err" &
  card_pid=$!

  wait "$gj_pid"       || { echo "gpu-prices: aggregate source failed (continuing)" >&2; }
  wait "$az_pid"       || { echo "gpu-prices: azure source failed (continuing)" >&2; }
  wait "$card_pid"     || { echo "gpu-prices: consumer source failed (continuing)" >&2; }

  [ -s "$tmp/gj.json" ] || printf 'null'          > "$tmp/gj.json"
  [ -s "$tmp/az.json" ] || printf '{"Items":[]}'  > "$tmp/az.json"

  printf '[]' > "$tmp/consumer.json"
  if command -v python3 >/dev/null && [ -s "$tmp/cards.json" ] && [ -s "$(jq -r '.[0].path // empty' "$tmp/cards.json" 2>/dev/null)" ]; then
    if ! python3 - "$tmp/cards.json" > "$tmp/consumer.json" 2>"$tmp/parse.err" <<'PY'
import json, re, sys

cards = json.load(open(sys.argv[1]))
out = []
for card in cards:
    try:
        html = open(card["path"], encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    # Each <tr> after the header is one provider/priority-tier row.
    for row in html.split("<tr")[2:]:
        prov = re.search(r'href="/providers/[^"]+"[^>]*>([^<]+)</a>', row)
        price = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", row)
        if not prov or not price:
            continue
        tier = re.search(r">((?:Lowest|Low|Medium|High) Priority)<", row)
        upd = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", row)
        tier_name = tier.group(1) if tier else ""
        as_of = None
        if upd:
            as_of = f"{upd.group(3)}-{int(upd.group(1)):02d}-{int(upd.group(2)):02d}"
        out.append({
            "gpu": card["gpu"],
            "provider": prov.group(1).strip(),
            # Salad's High tier is never preempted; the other three can be.
            "billing": "on-demand" if tier_name == "High Priority" else "preemptible",
            "usd_per_gpu_hour": round(float(price.group(1)), 4),
            "region": None,
            "as_of": as_of,
            "source": f"https://computeprices.com/gpus/{card['slug']}",
            "detail": tier_name,
        })
print(json.dumps(out))
PY
    then
      echo "gpu-prices: consumer parse failed (continuing)" >&2
      printf '[]' > "$tmp/consumer.json"
    fi
  fi

  cat > "$tmp/normalize.jq" <<'JQ'
def amd_model:
  if   test("MI355X")   then "MI355X"
  elif test("MI350X")   then "MI350X"
  elif test("MI325X")   then "MI325X"
  elif test("MI300A")   then "MI300A"
  elif test("MI300X")   then "MI300X"
  elif test("MI250X")   then "MI250X"
  elif test("MI250")    then "MI250"
  elif test("MI210")    then "MI210"
  elif test("MI100")    then "MI100"
  elif test("W7900")    then "Radeon PRO W7900"
  elif test("W7800")    then "Radeon PRO W7800"
  elif test("9070")     then "RX 9070 XT"
  elif test("9060")     then "RX 9060 XT"
  elif test("7900 XTX") then "RX 7900 XTX"
  elif test("7900 XT")  then "RX 7900 XT"
  elif test("7800 XT")  then "RX 7800 XT"
  elif test("6800 XT")  then "RX 6800 XT"
  else . end;

def bill:
  if . == null then "on-demand"
  elif test("commit|reserv|reservation")      then "reserved"
  elif test("spot|preempt|interrupt")         then "spot"
  elif test("on.demand|pod|per.hour")         then "on-demand"
  else . end;

# GPU count per Azure MI SKU. The SKU name encodes vCPUs, not accelerators.
def az_gpus:
  if   test("MI355X|MI350X|MI325X|MI300X") then 8
  elif test("MI300A|MI250X")               then 4
  else 1 end;

def az_billing:
  if   test("Spot")         then "spot"
  elif test("Low Priority") then "low-priority"
  elif test("DevTest")      then "devtest"
  elif test("Reservation")  then "reserved"
  else "on-demand" end;

def r4: . * 10000 | round / 10000;
def ok($re): $re == "." or test($re; "i");

(.[0].offers // []) as $gj
| (.[1].Items // []) as $az
| (.[2] // [])       as $consumer
| [
    ($gj[]
      | select((.usdPerGpuHour // 0) > 0)
      # The Azure retail API below is authoritative for Azure and carries region
      # detail; drop the aggregate's Azure rows so they do not double-count.
      | select(.provider != "Azure")
      # "Market median" rows are an index, not somewhere you can rent.
      | select((.provider // "") | test("median|index"; "i") | not)
      | select((.gpu // "") | ok($re))
      | {
          gpu: (.gpu | amd_model),
          provider: .provider,
          usd_per_gpu_hour: (.usdPerGpuHour | r4),
          billing: (.pricingModel | bill),
          region: null,
          as_of: .asOf,
          source: .sourceUrl,
          detail: (.instance // "")
        }),
    ($az[]
      | select(.currencyCode == "USD")
      | select(.armSkuName | test("MI[0-9]{3}"))
      | (.armSkuName | az_gpus) as $n
      | {
          gpu: (.armSkuName | amd_model),
          provider: "Azure",
          usd_per_gpu_hour: ((.retailPrice / $n) | r4),
          billing: (.skuName | az_billing),
          region: .armRegionName,
          as_of: null,
          source: "https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
          detail: .armSkuName
        }),
    ($consumer[]
      | select(.gpu | ok($re))
      | .)
  ]
  | unique_by([.gpu, .provider, .billing, .region, .usd_per_gpu_hour])
JQ

  if ! jq -s --arg re "$PATTERN" -f "$tmp/normalize.jq" \
        "$tmp/gj.json" "$tmp/az.json" "$tmp/consumer.json" \
        > "$tmp/offers.json" 2>"$tmp/jq.err"; then
    echo "gpu-prices: normalization failed:" >&2
    cat "$tmp/jq.err" >&2
    if [ -f "$CACHE" ]; then
      echo "gpu-prices: falling back to cache ($(cache_age_h)h old)" >&2
      need_fetch=0
    else
      die "no cache available"
    fi
  fi

  if [ "$need_fetch" -eq 1 ]; then
    n_offers="$(jq 'length' "$tmp/offers.json" 2>/dev/null || echo 0)"
    if [ "${n_offers:-0}" -eq 0 ]; then
      # An all-empty result means every source failed. Never let that clobber a
      # good cache - an empty board is indistinguishable from a broken network.
      echo "gpu-prices: no source returned offers; refusing to overwrite cache" >&2
      if [ -f "$CACHE" ]; then
        echo "gpu-prices: using cache ($(cache_age_h)h old)" >&2
        need_fetch=0
      else
        die "no price data from any source"
      fi
    fi
  fi

  if [ "$need_fetch" -eq 1 ]; then
    mkdir -p "$(dirname "$CACHE")"
    jq -n --arg ts "$(date -u +%FT%TZ)" \
          --argjson offers "$(cat "$tmp/offers.json")" \
          '{fetched_at: $ts, sources: {
              aggregate: "https://www.givejensenmoney.app/api/prices",
              azure: "https://prices.azure.com/api/retail/prices",
              consumer: "https://computeprices.com/gpus/<card>"
            }, offers: $offers}' > "$CACHE.tmp"
    mv "$CACHE.tmp" "$CACHE"
  fi
fi

[ -f "$CACHE" ] || die "no price data (fetch failed and no cache)"
age="$(cache_age_h)"
[ "$age" -eq 999999 ] && die "cache is unreadable"
if [ "$age" -ge "$MAX_AGE_H" ]; then
  echo "gpu-prices: WARNING cache is ${age}h old (max ${MAX_AGE_H}h) - figures may be stale" >&2
fi

if [ "$MODE" = json ]; then
  jq --arg re "$PATTERN" '
    {fetched_at: .fetched_at, sources: .sources,
     offers: (.offers
       | map(select($re == "." or (.gpu | test($re; "i"))))
       | sort_by(.usd_per_gpu_hour))}
  ' "$CACHE"
  exit 0
fi

echo
echo "AMD GPU rental prices  (fetched $(jq -r '.fetched_at' "$CACHE"), cache age ${age}h)"
echo

jq -r --arg re "$PATTERN" '
  def age_days:
    if . == null then null
    else (.[0:10] | strptime("%Y-%m-%d") | mktime) as $t
         | ((now - $t) / 86400 | floor)
    end;
  def stamp:
    if .as_of == null then "live"
    elif ((.as_of | age_days) // 0) > 14 then (.as_of + " *")
    else .as_of end;

  [.offers[] | select($re == "." or (.gpu | test($re; "i")))]
  | group_by([.gpu, .provider, .billing])
  | map(
      min_by(.usd_per_gpu_hour) as $m
      | ($m + {offers: length, region: ($m.region // "-")})
    )
  | sort_by(.gpu, .usd_per_gpu_hour)
  | if length == 0 then "  no offers matched"
    else
      (["GPU","PROVIDER","$/GPU-HR","BILLING","REGION","PRICE-AS-OF","N"] | @tsv),
      (.[] | [
         .gpu,
         .provider,
         ("$" + (.usd_per_gpu_hour | tostring)),
         .billing,
         .region,
         stamp,
         (.offers | tostring)
       ] | @tsv)
    end
' "$CACHE" | column -t -s $'\t' | sed 's/^/  /'

echo
echo "Cheapest non-preemptible rate per model"
echo
jq -r --arg re "$PATTERN" '
  [.offers[] | select($re == "." or (.gpu | test($re; "i")))
            | select(.billing == "on-demand")]
  | group_by(.gpu)
  | map(min_by(.usd_per_gpu_hour))
  | sort_by(.usd_per_gpu_hour)
  | if length == 0 then "  none"
    else
      (["GPU","BEST","$/GPU-HR","$/MONTH (730h)"] | @tsv),
      (.[] | [ .gpu, .provider, ("$" + (.usd_per_gpu_hour|tostring)),
               ("$" + ((.usd_per_gpu_hour * 730) | round | tostring)) ] | @tsv)
    end
' "$CACHE" | column -t -s $'\t' | sed 's/^/  /'

echo
echo "  Prices exclude tax, storage and egress. Preemptible/spot can be reclaimed."
echo "  Model -> ROCm target mapping: specs/MULTI_GPU.md"
echo "  Full record: ${CACHE#"$ROOT"/}"
echo
