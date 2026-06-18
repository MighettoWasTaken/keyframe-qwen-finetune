"""
Tests for the generation cascade, validation composition, dedup, and usage
accounting in generate_data.py. These use fake `generate`/`validate` callables
so the escalation logic can be exercised without any API calls or a browser.
"""

from types import SimpleNamespace

import generate_data as gd


# --- helpers ----------------------------------------------------------------

def make_prompts(n):
    return [{"description": f"motion brief number {i}", "techniques": []} for i in range(n)]


def anim_text(batch, marker="ok"):
    """Valid <animation> blocks (one per batch item) that parse_animations accepts."""
    blocks = []
    for i, _ in enumerate(batch, start=1):
        code = (f'<div class="a"></div><style>/* {marker} */'
                f'@keyframes k{{from{{opacity:0}}to{{opacity:1}}}}'
                f'.a{{animation:k 1s infinite}}</style>')
        blocks.append(f'<animation index="{i}">{code}</animation>')
    return "\n".join(blocks)


def all_pass(rows):
    return [True] * len(rows)


# --- parse / cascade basics -------------------------------------------------

def test_all_pass_on_first_tier_no_escalation():
    prompts = make_prompts(4)
    calls = []

    def generate(tier, batch):
        calls.append(tier)
        return anim_text(batch), None

    results, stats = gd.run_cascade(
        prompts, tiers=["haiku", "sonnet", "opus"], batch_size=2, workers=2,
        generate=generate, validate=all_pass, log=lambda *a: None,
    )

    assert len(results) == 4
    assert set(calls) == {"haiku"}            # cheapest tier only
    assert stats[0]["tier"] == "haiku"
    assert stats[0]["kept"] == 4 and stats[0]["failed"] == 0
    assert len(stats) == 1                      # later tiers never ran


def test_parse_failure_escalates_to_next_tier():
    prompts = make_prompts(4)

    def generate(tier, batch):
        # Haiku emits junk (no <animation> blocks) -> all parse-fail -> escalate.
        if tier == "haiku":
            return "sorry, here is some prose with no animation blocks", None
        return anim_text(batch), None

    results, stats = gd.run_cascade(
        prompts, tiers=["haiku", "sonnet", "opus"], batch_size=2, workers=2,
        generate=generate, validate=all_pass, log=lambda *a: None,
    )

    assert len(results) == 4
    assert stats[0]["tier"] == "haiku" and stats[0]["kept"] == 0 and stats[0]["failed"] == 4
    assert stats[1]["tier"] == "sonnet" and stats[1]["kept"] == 4 and stats[1]["failed"] == 0


def test_validation_failure_escalates():
    """Animation parses fine but fails validation (e.g. render/judge) -> escalate."""
    prompts = make_prompts(3)

    def generate(tier, batch):
        marker = "BLANK" if tier == "haiku" else "good"
        return anim_text(batch, marker=marker), None

    def validate(rows):
        return ["BLANK" not in r["code"] for r in rows]

    results, stats = gd.run_cascade(
        prompts, tiers=["haiku", "sonnet", "opus"], batch_size=2, workers=2,
        generate=generate, validate=validate, log=lambda *a: None,
    )

    assert len(results) == 3
    assert stats[0]["kept"] == 0 and stats[0]["failed"] == 3   # all blank on haiku
    assert stats[1]["kept"] == 3                                # sonnet renders fine


def test_partial_failure_only_failures_escalate():
    """One item per batch fails on haiku; only those should reach sonnet."""
    prompts = make_prompts(4)

    def generate(tier, batch):
        if tier == "haiku":
            # Emit a block for every item EXCEPT the one whose description ends in "0".
            blocks = []
            for i, p in enumerate(batch, start=1):
                if p["description"].endswith("0"):
                    continue  # drop -> parse failure for this one
                code = f'<div></div><style>@keyframes k{{}}</style>'
                blocks.append(f'<animation index="{i}">{code}</animation>')
            return "\n".join(blocks), None
        return anim_text(batch), None

    results, stats = gd.run_cascade(
        prompts, tiers=["haiku", "sonnet", "opus"], batch_size=4, workers=2,
        generate=generate, validate=all_pass, log=lambda *a: None,
    )

    assert len(results) == 4
    assert stats[0]["kept"] == 3 and stats[0]["failed"] == 1   # only "...0" failed
    assert stats[1]["attempted"] == 1 and stats[1]["kept"] == 1


def test_last_tier_failure_is_dropped_not_crash():
    prompts = make_prompts(2)

    def generate(tier, batch):
        return "junk, no blocks", None

    results, stats = gd.run_cascade(
        prompts, tiers=["haiku", "sonnet", "opus"], batch_size=2, workers=2,
        generate=generate, validate=all_pass, log=lambda *a: None,
    )

    assert results == []
    assert stats[-1]["tier"] == "opus" and stats[-1]["failed"] == 2


def test_leftover_partial_batch_runs():
    """5 prompts with batch_size 2 -> batches of 2,2,1; the trailing 1 must run."""
    prompts = make_prompts(5)
    seen = []

    def generate(tier, batch):
        seen.append(len(batch))
        return anim_text(batch), None

    results, _ = gd.run_cascade(
        prompts, tiers=["haiku"], batch_size=2, workers=3,
        generate=generate, validate=all_pass, log=lambda *a: None,
    )

    assert len(results) == 5
    assert sorted(seen) == [1, 2, 2]


# --- validator composition (render + judge) ---------------------------------

def test_validator_judge_only_runs_on_render_passers():
    rows = [{"code": "a", "description": "x"},
            {"code": "b", "description": "y"},
            {"code": "c", "description": "z"}]
    # Render fails the middle row.
    render_fn = lambda codes: [True, False, True]
    judged = {}

    def judge_fn(passed_rows):
        judged["seen"] = [r["code"] for r in passed_rows]
        return [True] * len(passed_rows)

    validate = gd.make_validator(render_fn, judge_fn)
    mask = validate(rows)

    assert mask == [True, False, True]
    assert judged["seen"] == ["a", "c"]   # the render-failed row was never judged


def test_validator_judge_can_reject_a_match():
    rows = [{"code": "a", "description": "x"}, {"code": "b", "description": "y"}]
    render_fn = lambda codes: [True, True]
    judge_fn = lambda passed: [True, False]   # 2nd animation doesn't match its brief
    validate = gd.make_validator(render_fn, judge_fn)
    assert validate(rows) == [True, False]


def test_validator_without_judge_is_render_only():
    rows = [{"code": "a", "description": "x"}, {"code": "b", "description": "y"}]
    render_fn = lambda codes: [False, True]
    validate = gd.make_validator(render_fn, judge_fn=None)
    assert validate(rows) == [False, True]


# --- dedup ------------------------------------------------------------------

def test_dedupe_drops_exact_and_near_duplicates():
    existing = ["A spinning blue square that rotates 360 degrees and fades out"]
    new = [
        {"description": "A spinning blue square that rotates 360 degrees and fades out"},  # exact
        {"description": "A spinning blue square which rotates 360 degrees then fades out"},  # near
        {"description": "Three dots bouncing up and down in a staggered wave"},  # distinct
    ]
    kept = gd.dedupe_prompts(new, existing)
    descs = [p["description"] for p in kept]
    assert "Three dots bouncing up and down in a staggered wave" in descs
    assert len(kept) == 1


def test_dedupe_drops_within_batch():
    new = [
        {"description": "a glowing orb pulsing outward with a soft shadow"},
        {"description": "a glowing orb pulsing outward with a soft shadow"},   # dup of #1
        {"description": "a triangle morphing into a circle while skewing"},
    ]
    kept = gd.dedupe_prompts(new, existing=[])
    assert len(kept) == 2


# --- usage accounting -------------------------------------------------------

def fake_usage(in_=100, out=200, cw=0, cr=0):
    return SimpleNamespace(
        input_tokens=in_, output_tokens=out,
        cache_creation_input_tokens=cw, cache_read_input_tokens=cr,
    )


def test_usage_tracks_each_tier_separately():
    prompts = make_prompts(2)

    def generate(tier, batch):
        if tier == "claude-haiku-4-5":
            return "junk", fake_usage()            # fails -> escalates
        return anim_text(batch), fake_usage()

    usage = gd.Usage()
    results, _ = gd.run_cascade(
        prompts, tiers=["claude-haiku-4-5", "claude-sonnet-4-6"], batch_size=2,
        workers=2, generate=generate, validate=all_pass, usage=usage, log=lambda *a: None,
    )

    assert len(results) == 2
    # Both tiers were called, so both must appear in the usage breakdown.
    assert "claude-haiku-4-5" in usage._by_model
    assert "claude-sonnet-4-6" in usage._by_model
    assert usage.cost() > 0


def test_cost_uses_per_model_pricing():
    usage = gd.Usage()
    usage.add(fake_usage(in_=1_000_000, out=0), "claude-haiku-4-5")    # $1/MTok in
    usage.add(fake_usage(in_=1_000_000, out=0), "claude-opus-4-8")     # $5/MTok in
    # 1.0 (haiku) + 5.0 (opus) = 6.0
    assert abs(usage.cost() - 6.0) < 1e-6
