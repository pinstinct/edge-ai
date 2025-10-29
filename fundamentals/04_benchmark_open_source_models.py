""" Service Diagnostic & Model Discovery """
import json
import os
import statistics
import time

import requests
from foundry_local import FoundryLocalManager
from openai import OpenAI


def check_foundry_service():
    """Quick diagnostic to verify Foundry Local is running and detect the endpoint automatically."""
    print("[Diagnostic] Checking Foundry Local service...")

    # Strategy 1: Use SDK to detect service automatically
    try:
        # Try to connect to any available model to detect the service
        # This will auto-discover the endpoint
        temp_manager = FoundryLocalManager()
        detected_endpoint = temp_manager.endpoint

        if detected_endpoint:
            print(f"✅ Service auto-detected via SDK at {detected_endpoint}")

            # Verify by listing models
            try:
                model_response = requests.get(f"{detected_endpoint}/models", timeout=2)
                if model_response.status_code == 200:
                    model_data = model_response.json()
                    model_count = len(model_data.get("data", []))
                    print(f"✅ Found {model_count} models available")
                    if model_count > 0:
                        model_ids = [m.get("id", "unknown") for m in
                                     model_data.get("data", [])[:10]]
                        print(f"    Models: {model_ids}")
                return detected_endpoint
            except Exception as e:
                print(f"⚠️    Could not list models: {e}")
                return detected_endpoint
    except Exception as e:
        print(f"⚠️    SDK auto-detection failed: {e}")

    # Strategy 2: Fallback to manual prot scanning
    print("[Diagnostic] Trying manual prot detection...")
    endpoints_to_try = [
        "http://localhost:59959",
        "http://127.0.0.1:59959",
        "http://localhost:55769",
        "http://127.0.0.1:55769",
        "http://localhost:57127",
        "http://127.0.0.1:57127",
    ]

    for endpoint in endpoints_to_try:
        try:
            response = requests.get(f"{endpoint}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Service found at {endpoint}")

                # Try to list models
                try:
                    models_response = requests.get(f"{endpoint}/v1/models", timeout=2)
                    if models_response.status_code == 200:
                        model_data = models_response.json()
                        model_count = len(model_data.get("data", []))
                        print(f"✅ Found {model_count} models available")
                        if model_count > 0:
                            model_ids = [m.get("id", "unknown") for m in
                                         model_data.get("data", [])[:10]]
                            print(f"    Models: {model_ids}")
                        return endpoint
                except Exception as e:
                    print(f"⚠️    Could not list models: {e}")
                    return endpoint
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            print(f"⚠️    Error checking {endpoint}: {e}")

    print("\n❌ Foundry Local service not found!")
    print("\n💡 To fix this:")
    print("    1. Open a terminal")
    print("    2. Run: foundry service start")
    print("    3. Run: foundry model run phi-4-mini")
    print("    4. Run: foundry model run qwen2.5-0.5b")
    print("    5. Re-run this notebook")

    return None


# Run diagnostic
discovered_endpoint = check_foundry_service()

if discovered_endpoint:
    print(f"\n✅ Service detected - ready for benchmarking")
else:
    print(f"\n⚠️ No service detected - benchmarking will likely fail")

""" Benchmark Configuration & Model Filtering (Memory-Optimized) 
- CUDA 보다 CPU 변형을 선호하도록 검색된 모델을 자동으로 필터링
- CPU 모델은 좋은 성능을 유지하면서 메모리를 30-50% 적게 사용
- 우선순위: CPU-optimized > Quantized models > Other variants > CUDA (only if no alternative)
- BENCH_MODELS 환경 변수를 통해 수동 재정의 가능
"""

# Benchmark configuration & model discovery (override via environment variables)
BASE_URL = os.getenv("FOUNDRY_LOCAL_ENDPOINT",
                     discovered_endpoint if "discovered_endpoint" in dir() and discovered_endpoint else "http://127.0.0.1:59959")
if not BASE_URL.endswith("/v1"):
    BASE_URL = f"{BASE_URL}/v1"
API_KEY = os.getenv("API_KEY", "not-needed")

_raw_models = os.getenv("BENCH_MODELS", "").strip()
requested_models = [m.strip() for m in _raw_models.split(",") if m.strip()] if _raw_models else []

ROUNDS = int(os.getenv("BENCH_ROUNDS", "3"))
if ROUNDS < 1:
    raise ValueError("BENCH_ROUNDS must be >= 1")
PROMPT = os.getenv("BENCH_PROMPT", "Explain retrieval augmented generation briefly.")
MAX_TOKENS = int(os.getenv("BENCH_MAX_TOKENS", "120"))
TEMPERATURE = float(os.getenv("BENCH_TEMPERATURE", "0.2"))


def _discover_models():
    try:
        c = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        data = c.models.list().data
        return [m.id for m in data]
    except Exception as e:
        print(f"Model discovery failed: {e}")
        return []


def _prefer_cpu_models(model_list):
    """Filter models to prefer CPU variants over CUDA for memory efficiency.

    Priority order:
    1. CPU-optimized models (e.g., *-cpu, *-cpu-int4)
    2. Quantized models without CUDA (e.g., *-q4, *-int4)
    3. Other models (excluding CUDA variants if CPU available)
    """
    # Group models by base name (removing variant suffixes)
    from collections import defaultdict
    model_groups = defaultdict(list)

    for model in model_list:
        # Extract base name (before variant like -cpu, -cuda, -int4, etc.)
        base_name = model.split("-cpu")[0].split("-cuda")[0].split("-int4")[0].split("-q4")[0]
        model_groups[base_name].append(model)

    selected = []
    for base_name, variants in model_groups.items():
        # Prioritize CPU variants
        cpu_variants = [m for m in variants if "-cpu" in m.lower()]
        cuda_variants = [m for m in variants if "-cuda" in m.lower()]
        other_variants = [m for m in variants if m not in cpu_variants and m not in cuda_variants]

        if cpu_variants:
            # Prefer CPU variants
            selected.extend(cpu_variants)
            print(f"✓ Selected CPU variant for {base_name}: {cpu_variants[0]}")
        elif other_variants:
            # User non-CUDA variants if available
            selected.extend(other_variants[:1])  # Take first one
        elif cuda_variants:
            # Only use CUDA if no other option
            selected.extend(cuda_variants[:1])
            print(
                f"⚠️    Using CUDA variant for {base_name}: {cuda_variants[0]} (no CPU variant found)")
    return selected


_discovered = _discover_models()
if not _discovered:
    print(
        "Warning: No models discovered at BASE_URL. Ensure Foundry Local is running and models "
        "are loaded.")

if not requested_models or requested_models == ["auto"] or "ALL" in requested_models:
    # Auto mode: discover and prefer CPU models
    MODELS = _prefer_cpu_models(_discovered)
    if len(MODELS) < len(_discovered):
        print(
            f"💡    Memory-optimized: Using {len(MODELS)} CPU models instead of all {len(_discovered)} variants")
else:
    # Filter requested models to those actually discovered
    MODELS = [m for m in requested_models if
              m in _discovered] or requested_models  # fallback to requested even if not discovered
    missing = [m for m in requested_models if m not in _discovered]
    if missing:
        print(
            f"Notice: THe following requested models were not discovered and may fail during "
            f"benchmarking: {missing}")

MODELS = [m for m in MODELS if m]
if not MODELS:
    raise ValueError(
        "No models available to benchmark. Start a model (e.g., 'foundry model run phi-4-mini') "
        "or set BENCH_MODELS.")

print(
    f"Benchmarking models: {MODELS}\nRounds: {ROUNDS} Max Tokens: {MAX_TOKENS} Temp: {TEMPERATURE}")

""" Model Access Helper (Memory-Optimized) """


def ensure_loaded(alias):
    """This follows the official Foundry Local SDK pattern with CPU preference:

    1. FoundryLocalManager(alias) - Automatically starts service and loads model if needed
    2. Prefers CPU variants over CUDA for memory efficiency
    3. Create OpenAI client with manager's endpoint
    4. Resolve model ID from alias

    :return: (manager, client, model_id) ensuring the alias is accessible.
    Raises RuntimeError with guidance if the model cannot be accessed.
    """
    try:
        # Initialize manager - this auto-starts service and loads model if needed
        # Note: By default, Foundry Local may select CUDA if available
        # For memory efficiency, we recommend using CPU-optimized aliases explicitly
        m = FoundryLocalManager(alias)

        # Get resolved model ID
        info = m.get_model_info(alias)
        model_id = getattr(info, "id", alias)

        # Warn if CUDA variant was loaded
        if "cuda" in model_id.lower():
            print(f"⚠️ Loaded CUDA variant: '{alias}' -> '{model_id}'")
            print(
                f"💡 For lower memory usage, use CPU variant with: foundry model run {alias.split('-cuda')[0]}-cpu")
        else:
            print(f"✓ Loaded model: '{alias}' -> '{model_id}' at {m.endpoint}")
            if "cpu" in model_id.lower():
                print(f"✅ Using memory-optimized CPU variant")

        c = OpenAI(base_url=m.endpoint, api_key=m.api_key or "not-needed")
        return m, c, model_id

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{alias}'.\n"
            f"Original error: {e}\n\n"
            f"💡 To fix:\n"
            f"    1. Ensure Foundry Local service is running: foundry servce start\n"
            f"    2. Verify model is available: foundry model ls\n"
            f"    3. For CPU-optimized models: foundry model run {alias}\n"
            f"    4. Check available variants with: foundry model search {alias.split('-')[0]}"
        )


""" Single Round Execution 
채팅 완료를 한번 수행하고, 지연시간+토큰사용량 필드를 반환한다. 
만약 API가 토큰 수를 제공하지 않는 경우, ~4 chars/token 휴리스틱을 사용해 추정한다.
"""


def run_round(client, model_id, prompt):
    """Execute one chat completion round with comprehensive metric capture.

    :return: Tuple of (latency_sec, total_tokens, prompt_tokens, completion_tokens, response_text)
    Token counts are estimate if API doesn't provide them.
    """
    start = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    end = time.time()
    latency = end - start

    # Extract response content
    content = resp.choices[0].message.content if resp.choices else ""

    # Try to get usage from API
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    # Estimate tokens if API doesn't provide them (~4 chars per token per English)
    if prompt_tokens is None:
        prompt_tokens = len(prompt) // 4
    if completion_tokens is None:
        completion_tokens = len(content) // 4
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    return latency, total_tokens, prompt_tokens, completion_tokens, content


""" Benchmark Loop & Aggregation 
- 콜드 스타트를 완화하기 위한 Warmup 통계에서 제외
- 평균, p95, tokens/sec 집계
"""

summary = []
for alias in MODELS:
    try:
        m, client, model_id = ensure_loaded(alias.strip())
    except Exception as e:
        print(e)
        continue

    # Warmup (not recorded)
    try:
        run_round(client, model_id, PROMPT)
    except Exception as e:
        print(f"Warmup failed for {alias}: {e}")
        continue

    latencies, tps = [], []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_sum = 0
    sample_output = None

    for round_num in range(ROUNDS):
        try:
            latency, total_tokens, p_tokens, c_tokens, content = run_round(client, model_id,
                                                                           PROMPT)
        except Exception as e:
            print(f"Round {round_num + 1} failed for {alias}: {e}")
            continue

        latencies.append(latency)
        prompt_tokens_total += p_tokens
        completion_tokens_total += c_tokens
        total_tokens_sum += total_tokens

        # Calculate tokens per second
        if total_tokens and latency > 0:
            tps.append(total_tokens / latency)

        # Capture first successful output as sample
        if sample_output is None:
            sample_output = content[:200]  # First 200 chars

    if not latencies:
        print(f"Skipping {alias}: no successful rounds.")
        continue

    # Calculate statistics
    rounds_ok = len(latencies)
    latency_avg = statistics.mean(latencies)
    latency_min = min(latencies)
    latency_max = max(latencies)
    latency_p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0]
    tokens_per_sec_avg = statistics.mean(tps) if tps else None

    # Average tokens per round
    avg_prompt_tokens = prompt_tokens_total / rounds_ok if rounds_ok else 0
    avg_completion_tokens = completion_tokens_total / rounds_ok if rounds_ok else 0
    avg_total_tokens = total_tokens_sum / rounds_ok if rounds_ok else 0

    summary.append({
        "alias": alias,
        "model_id": model_id,
        "latency_avg_s": latency_avg,
        "latency_min_s": latency_min,
        "latency_max_s": latency_max,
        "latency_p95_s": latency_p95,
        "tokens_per_sec_avg": tokens_per_sec_avg,
        "avg_prompt_tokens": avg_completion_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_sum": total_tokens_sum,
        "rounds_ok": rounds_ok,
        "configured_rounds": ROUNDS,
        "sample_output": sample_output,
    })

""" Results Rendering """
print("=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)

if not summary:
    print("No results to display")
else:
    # Calculate best/worst for highlighting
    if len(summary) > 0:
        best_latency = min(r["latency_avg_s"] for r in summary)
        worst_latency = max(r["latency_avg_s"] for r in summary)
        best_tps = max((r["tokens_per_sec_avg"] for r in summary if r["tokens_per_sec_avg"]),
                       default=None)
        worst_tps = min((r["tokens_per_sec_avg"] for r in summary if r["tokens_per_sec_avg"]),
                        default=None)

    # Enhanced comprehensive table with performance indicators
    print("\n📊 PERFORMANCE SUMMARY TABLE")
    print("=" * 80)
    headers = ["Model", "Latency (avg)", "Latency (P95)", "Throughput", "Tokens", "Success",
               "Rating"]
    rows = []

    for r in summary:
        # Performance indicators
        lat_indicator = "🟢" if r["latency_avg_s"] == best_latency else (
            "🔴" if r["latency_avg_s"] == worst_latency else "🟡")

        tps_indicator = ""
        if r["tokens_per_sec_avg"]:
            if best_tps and r["tokens_per_sec_avg"] == best_tps:
                tps_indicator = "🟢"
            if worst_tps and r["tokens_per_sec_avg"] == worst_tps:
                tps_indicator = "🔴"
            else:
                tps_indicator = "🟡"

        # Overall rating based on latency and throughput
        rating = ""
        if r["latency_avg_s"] == best_latency or (
                r["tokens_per_sec_avg"] and r["tokens_per_sec_avg"] == best_tps):
            rating = "⭐⭐⭐"
        elif r["latency_avg_s"] == worst_latency or (
                r["tokens_per_sec_avg"] and r["tokens_per_sec_avg"] == worst_tps):
            rating = "⭐"
        else:
            rating = "⭐⭐"

        rows.append([
            r["alias"][:20],
            f"{lat_indicator} {r['latency_avg_s']:.3f}s",
            f"{r['latency_p95_s']:.3f}s",
            f"{tps_indicator} {r['tokens_per_sec_avg']:.1f}" if r["tokens_per_sec_avg"] else "-",
            f"{r['avg_total_tokens']:.0f}",
            f"{r['rounds_ok']}/{r['configured_rounds']}",
            rating,
        ])

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]


    def fmt_row(row):
        return " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths))


    print(fmt_row(headers))
    print("-" + "-+-".join("-" * w for w in col_widths) + "-")
    for row in rows:
        print(fmt_row(row))

    print("\n" + "=" * 80)
    print(
        "Legend: 🟢 Best  🟡 Average  🔴 Worst  |  Rating: ⭐⭐⭐ Excellent  ⭐⭐ Good  ⭐ Needs Improvement")
    print("=" * 80)

    # Detailed metrics per model
    print("=" * 80)
    print("DETAILED METRICS PER MODEL")
    print("=" * 80)

    for r in summary:
        print(f"\n📊 {r['alias']} ({r['model_id']})")
        print(f"    Latency:")
        print(f"        Average: {r['latency_avg_s']:.3f}s")
        print(f"        Min:     {r['latency_min_s']:.3f}s")
        print(f"        Max:     {r['latency_max_s']:.3f}s")
        print(f"        P95:     {r['latency_p95_s']:.3f}s")
        print(f"    Tokens:")
        print(f"        Avg Prompt:     {r['avg_prompt_tokens']:.0f}")
        print(f"        Avg Completion: {r['avg_completion_tokens']:.0f}")
        print(f"        Avg Total:      {r['avg_total_tokens']:.0f}")
        if r["tokens_per_sec_avg"]:
            print(f"    Throughput: {r['tokens_per_sec_avg']:.1f} tok/s")
            print(f"    Rounds: {r['rounds_ok']}/{r['configured_rounds']} successful")
        if r.get("sample_output"):
            print(f"    Sample Output: {r['sample_output'][:150]}...")

    # Comparative analysis
    if len(summary) > 1:
        print("\n" + "=" * 80)
        print("🔍 PERFORMANCE COMPARISON")
        print("=" * 80)

        # Sort by latency for speed comparison
        sorted_by_speed = sorted(summary, key=lambda x: x["latency_avg_s"])
        fastest = sorted_by_speed[0]
        slowest = sorted_by_speed[-1]

        # Create performance comparison table
        print("\n📈 Relative Performance (normalized to fasted model)")
        print("-" * 80)
        comp_headers = ["Model", "Speed vs Fastest", "Latency Delta", "Throughput", "Efficiency"]
        comp_rows = []

        for r in sorted_by_speed:
            speedup = r["latency_avg_s"] / fastest["latency_avg_s"]
            latency_delta = r["latency_avg_s"] - fastest["latency_avg_s"]

            # Speed indicator
            if speedup <= 1.1:
                speed_bar = "█████ 100%"
                speed_emoji = "🚀"
            elif speedup <= 1.5:
                speed_bar = "████░ 80%"
                speed_emoji = "⚡"
            elif speedup <= 2.0:
                speed_bar = "███░░ 60%"
                speed_emoji = "🏃"
            else:
                speed_bar = "██░░░ 40%"
                speed_emoji = "🐌"

            # Efficiency score (lower is better: combines latency and throughput)
            if r["tokens_per_sec_avg"]:
                efficiency = f"{r['tokens_per_sec_avg']:.1f} tok/s"
            else:
                efficiency = "N/A"

            comp_rows.append([
                f"{speed_emoji} {r['alias'][:18]}",
                speed_bar,
                f"+{latency_delta:.3f}s" if latency_delta > 0 else "baseline",
                efficiency,
                f"{(1 / speedup) * 100:.0f}%"
            ])

        comp_widths = [max(len(str(cell)) for cell in col) for col in
                       zip(comp_headers, *comp_rows)]


        def comp_fmt_row(row):
            return " | ".join(str(c).ljust(w) for c, w in zip(row, comp_widths))


        print(comp_fmt_row(comp_headers))
        print("-+-".join("-" * w for w in comp_widths))
        for row in comp_rows:
            print(comp_fmt_row(row))

        # Summary statistics
        print("\n" + "=" * 80)
        print("📊 KEY FINDINGS")
        print("=" * 80)

        print(f"\n🏃 Fasted Model: {fastest['alias']}")
        print(f"    ├─ Average latency: {fastest['latency_avg_s']:.3f}s")
        print(f"    ├─ P95 latency: {fastest['latency_p95_s']:.3f}s")
        if fastest["tokens_per_sec_avg"]:
            print(f"    └─ Throughput: {fastest['tokens_per_sec_avg']:.1f} tok/s")

        if len(summary) > 1:
            print(f"\n🐌 Slowest Model: {slowest['alias']}")
            print(f"    ├─ Average latency: {slowest['latency_avg_s']:.3f}s")
            speedup = slowest["latency_avg_s"] / fastest["latency_avg_s"]
            print(f"    └─ Performance gap: {speedup:.2f}x slower than fastest")

        # Throughput comparison
        with_throughput = [r for r in summary if r["tokens_per_sec_avg"]]
        if len(with_throughput) > 1:
            sorted_by_tps = sorted(with_throughput, key=lambda x: x["tokens_per_sec_avg"],
                                   reverse=True)
            highest_tps = sorted_by_tps[0]
            lowest_tps = sorted_by_tps[-1]

            print(f"\n⚡ Highest Throughput: {highest_tps['alias']}")
            print(f"    ├─ Throughput: {highest_tps['tokens_per_sec_avg']:.1f} tok/s")
            print(f"    └─ Latency: {highest_tps['latency_avg_s']:.3f}s")

            if highest_tps["alias"] != lowest_tps["alias"]:
                throughput_gap = highest_tps["tokens_per_sec_avg"] / lowest_tps[
                    "tokens_per_sec_avg"]
                print(
                    f"\n💡 Throughput Range: {throughput_gap:.2f}x difference between best and worst")

        # Memory efficiency note
        print("\n💾 Memory Efficiency:")
        cpu_models = [r for r in summary if "cpu" in r["model_id"].lower()]
        if cpu_models:
            print(
                f"    ├─ {len(cpu_models) / len(summary)} models using CP variants (30-50% memory savings)")
            print(f"    └─ Recommended for systems with limited memory")

    # Expert Json
    print("\n" + "=" * 80)
    print("JSON SUMMARY (for programmatic analysis)")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

print("\n" + "=" * 80)
print(f"Benchmark completed: {len(summary)} models tested")
print(f"Configuration: {ROUNDS} rounds, {MAX_TOKENS} max tokens, temp={TEMPERATURE}")
print(f"Prompt: {PROMPT[:60]}...")
print("=" * 80)

""" Summary and Next Steps """
# Final Validation Check
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

validation_checks = []

# Check service detection
if "discovered_endpoint" in dir() and discovered_endpoint:
    validation_checks.append(("✅", "Service Auto-Detection", f"Found at {discovered_endpoint}"))
else:
    validation_checks.append(("⚠️", "Service Auto-Detection", "Not detected - using default"))

# Check configuration
if "MODELS" in dir() and MODELS:
    validation_checks.append(
        ("✅", "Models Configuration", f"{len(MODELS)} models configured: {MODELS}"))
else:
    validation_checks.append(("❌", "Models Configuration", "No models configured"))

# Check benchmark results
if "summary" in dir() and summary:
    successful = [r for r in summary if r['rounds_ok'] > 0]
    validation_checks.append(("✅", "Benchmark Execution",
                             f"{len(successful)}/{len(summary)} models complete"))

    # Check all have complete metrics
    all_have_metrics = all(
        r.get("latency_avg_s") and
        r.get("tokens_per_sec_avg") and
        r.get("avg_total_tokens")
        for r in successful
    )

    if all_have_metrics:
        validation_checks.append(
            ("✅", "Metrics Completeness", "All models have comprehensive metrics"))
    else:
        validation_checks.append(("⚠️", "Metrics Completeness", "Some metrics missing"))
else:
    validation_checks.append(("❌", "Benchmark Execution", "No results yet"))

# Display validation results
for icon, check_name, status in validation_checks:
    print(f"{icon} {check_name:<25} {status}")

print("=" * 80)

# Overall status
all_passed = all(icon == "✅" for icon, _, _ in validation_checks)
if all_passed:
    print("\n🎉 ALL VALIDATIONS PASSED! Benchmark completed successfully.")
    if "summary" in dir() and len(summary) > 0:
        print(f"    Successfully benchmarked {len(summary)} models")
        print(f"    Configuration: {ROUNDS} rounds, {MAX_TOKENS} tokens, temp={TEMPERATURE}")
else:
    print("\n⚠️ Some validations did not pass. Review the issues above.")
    print("\n💡 Common fixes:")
    print("    1. Ensure Foundry Local service is running: foundry service start")
    print("    2. Load models: foundry model run phi-4-mini && foundry model run qwen2.5-0.5b")
    print("    3. Check model availability: foundry model ls")
    print("    4. Re-run the benchmark cells")

print("=" * 80)
