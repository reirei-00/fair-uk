"""Published WinoPron role/order protocol and monolingual bundle checks, offline."""

import json
from copy import deepcopy

import pytest

from lm_eval.fair_uk.capabilities import resolve_tasks, task_metadata, validate_backend
from lm_eval.fair_uk.data import (
    PROTOCOLS,
    REGISTRY,
    adapt,
    digest,
    load,
    prompts,
    validate,
)
from lm_eval.fair_uk.metric_specs import metric_catalog
from lm_eval.fair_uk.metrics import (
    WINOPRON_POLICY,
    make_report,
    native_summaries,
    score_all,
    score_item,
)
from lm_eval.fair_uk.runner import predict


def source(
    pair="p1", condition="double", pronoun="male", role="occupation", first="occupation"
):
    other = "participant" if first == "occupation" else "occupation"
    row_id = f"{pair}:{role}:{condition}:{pronoun}"
    return {
        "prompt_id": f"{row_id}:{first}_first",
        "row_id": row_id,
        "pair_id": pair,
        "context": condition,
        "pronoun_type": "nominative",
        "pronoun_set": pronoun,
        "option_order": f"{first}_first",
        "option_a_role": first,
        "option_b_role": other,
        "gold_role": role,
        "gold_option": "A" if role == first else "B",
        "prompt_uk": "Речення: Технік подякував клієнту.\nКого позначає займенник?\nA. перша особа\nB. друга особа\nВідповідь:",
    }


def panel(pair="p1"):
    return [
        adapt("winopron_uk", source(pair, context, gender, role, first), language="uk")
        for context in ("double", "single")
        for gender in ("male", "female", "plural")
        for role in ("occupation", "participant")
        for first in ("occupation", "participant")
    ]


def correct(rows):
    return [{"id": r["id"], "response": "AB"[r["gold"]]} for r in rows]


def test_native_adapter_preserves_released_prompt_and_has_no_english_counterpart():
    row = adapt("winopron_uk", source(), language="uk")
    assert prompts(row) == [{"context": row["source"]["prompt_uk"]}]
    assert row["cluster"] == row["pair_id"]
    assert row["comparison_unit"] == "p1:double:occupation"
    assert row["option_roles"] == ["occupation", "participant"]
    assert task_metadata("winopron_uk")["counterpart"] is None
    assert task_metadata("winopron_uk")["required_capability"] == "generate_text"
    validate_backend(["winopron_uk"], "hf")
    validate_backend(["winopron_uk"], "vllm")
    with pytest.raises(ValueError, match="hosted API protocol"):
        validate_backend(["winopron_uk"], "api")
    with pytest.raises(ValueError, match="no published en"):
        resolve_tasks({}, languages=["en"], benchmarks=["winopron"])
    with pytest.raises(ValueError, match="no English"):
        adapt("winopron_uk", source(), language="en")


@pytest.mark.parametrize(
    "response,role",
    [
        ("A", "occupation"),
        ("Answer: a.", "occupation"),
        ("A A", "occupation"),
        ("b", "participant"),
        ("A or B", "invalid"),
        ("А", "invalid"),
        ("C", "invalid"),
        ("refusal", "invalid"),
    ],
)
def test_versioned_parser_matches_standalone_ab_reference(response, role):
    row = adapt("winopron_uk", source(), language="uk")
    result = score_item(row, {"response": response})
    assert result["prediction_role"] == role
    assert result["response"] == response
    assert result["scoring_policy"] == WINOPRON_POLICY
    assert result["prompt_accuracy"] == float(role == "occupation")
    assert result["prompt_invalid"] == float(role == "invalid")


def test_counterbalanced_role_agreement_and_pair_metrics_have_correct_denominators():
    rows = panel()
    predictions = correct(rows)
    # Returning A in both orders gets one prompt right but refers to different roles.
    affected = "p1:occupation:double:male"
    for row, prediction in zip(rows, predictions, strict=True):
        if row["row_id"] == affected:
            prediction["response"] = "A"
    scored = score_all(rows, predictions)
    bad = [r for r in scored if r["row_id"] == affected]
    assert sum(r["prompt_accuracy"] for r in bad) == 1
    assert all(r["accuracy"] == 0 and r["invalid"] == 1 for r in bad)
    native = native_summaries(scored)
    summary = native[0]
    assert summary["n_prompts"] == 24 and summary["n_rows"] == 12
    assert summary["prompt_accuracy"] == pytest.approx(100 * 23 / 24)
    assert summary["option_order_consistency"] == pytest.approx(100 * 11 / 12)
    assert summary["double_accuracy"] == pytest.approx(100 * 5 / 6)
    assert summary["single_accuracy"] == 100
    assert summary["female_minus_male_accuracy"] == 50
    assert summary["gender_form_consistency"] == 50
    assert summary["full_gender_pair_consistency"] == 0
    male = next(
        r
        for r in native
        if r["scope"] == "context_pronoun_set"
        and r["condition"] == "double"
        and r["group"] == "male"
    )
    assert male["n_rows"] == 2 and male["independent_pairs"] == 1
    assert male["accuracy"] == 50 and male["pair_consistency"] == 0
    assert not any("bias_score" in r or "stereotype_rate" in r for r in native)
    assert native == native_summaries(
        score_all(list(reversed(rows)), list(reversed(predictions)))
    )


def test_all_invalid_is_incorrect_and_not_low_bias_or_success():
    rows = panel()
    scored = score_all(rows, [{"id": r["id"], "response": "C"} for r in rows])
    report = make_report(scored, rows, bootstrap=0)
    summary = report["native"][0]
    assert summary["double_accuracy"] == summary["single_accuracy"] == 0
    assert summary["option_order_consistency"] == 0
    assert report["scoring_policy"] == WINOPRON_POLICY
    assert all(
        v["rate"] == 1
        for c in report["comparisons"]
        if c["metric"] == "invalid"
        for v in c["groups"].values()
    )
    assert metric_catalog("winopron_uk")["group_outcomes"] == [
        "accuracy",
        "invalid",
        "position_consistent",
    ]
    json.dumps(report, allow_nan=False)


def test_incomplete_orders_pairs_and_forged_mapping_fail():
    rows = panel()
    validate(rows)
    with pytest.raises(ValueError, match="option-order block"):
        validate(rows[1:])
    with pytest.raises(ValueError, match="both option orders"):
        score_all(rows[1:], correct(rows[1:]))
    removed_role = [
        r
        for r in rows
        if not (
            r["condition"] == "double"
            and r["pronoun_set"] == "male"
            and r["gold_role"] == "participant"
        )
    ]
    with pytest.raises(ValueError, match="complementary-reference"):
        validate(removed_role)
    forged = deepcopy(rows)
    forged[0]["row_id"] = "forged"
    with pytest.raises(ValueError, match="released source mapping"):
        validate(forged)


def write_bundle(tmp_path, rows):
    for row in rows:
        row.update(
            paired_eligible=False, pairing_exclusion_reason="published UK-only dataset"
        )
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "winopron_uk": {
                        "format": "adapted_jsonl_v1",
                        "filename": data.name,
                        "sha256": digest(data),
                        "rows": len(rows),
                        "language": "uk",
                        "protocol": PROTOCOLS["winopron_uk"],
                        "pairing": {
                            "mode": "unpaired",
                            "reason": "published UK-only dataset",
                        },
                    }
                },
            }
        )
    )
    return manifest


def test_explicit_monolingual_bundle_loads_without_global_registry_mutation(tmp_path):
    rows = panel()
    before = deepcopy(REGISTRY)
    manifest = write_bundle(tmp_path, rows)
    assert load("winopron_uk", dataset_bundle=manifest) == rows
    assert before == REGISTRY


@pytest.mark.parametrize(
    "failure", ["eligible", "missing_row_reason", "missing_contract_reason"]
)
def test_unpaired_bundle_cannot_claim_pairing_or_omit_reasons(tmp_path, failure):
    rows = panel()
    manifest = write_bundle(tmp_path, rows)
    if failure == "missing_contract_reason":
        content = json.loads(manifest.read_text())
        content["datasets"]["winopron_uk"]["pairing"].pop("reason")
        manifest.write_text(json.dumps(content))
    else:
        if failure == "eligible":
            rows[0]["paired_eligible"] = True
        else:
            rows[0].pop("pairing_exclusion_reason")
        data = tmp_path / "rows.jsonl"
        data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        content = json.loads(manifest.read_text())
        content["datasets"]["winopron_uk"]["sha256"] = digest(data)
        manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="unpaired"):
        load("winopron_uk", dataset_bundle=manifest)


def test_staged_stereo_adaptation_supports_full_and_masked_language_fields():
    before = deepcopy(REGISTRY)
    full = {
        "item_id": "s",
        "target_en": "target",
        "bias_type_en": "gender",
        "forget_en": "S",
        "retain_en": "C",
        "control_en": "U",
    }
    row = adapt("stereoset_en", full, language="en")
    assert row["sentences"] == ["S", "C", "U"]
    assert row["protocol"] == PROTOCOLS["stereoset_uk"]
    assert prompts(row) == [
        {"context": "", "continuation": s, "candidate": i}
        for i, s in enumerate(["S", "C", "U"])
    ]
    masked = {
        "item_id": "s",
        "target_uk": "target",
        "bias_type_uk": "gender",
        "template_uk": "A [BLANK].",
        "stereotype_fill_uk": "S",
        "anti_stereotype_fill_uk": "C",
        "unrelated_fill_uk": "U",
    }
    assert adapt("stereoset_uk", masked)["sentences"] == ["A S.", "A C.", "A U."]
    assert before == REGISTRY


def test_checkpoint_runner_uses_released_prompt_as_one_native_chat_message():
    row = adapt("winopron_uk", source(), language="uk")

    class FixtureBackend:
        def apply_chat_template(self, messages, add_generation_prompt):
            assert messages == [{"role": "user", "content": row["source"]["prompt_uk"]}]
            assert add_generation_prompt is True
            return "rendered released prompt"

        def generate_until(self, requests):
            assert requests[0].args[0] == "rendered released prompt"
            assert requests[0].args[1]["do_sample"] is False
            return ["A"]

    assert predict(FixtureBackend(), [row], generation_prompt_format="native-chat") == [
        {"id": row["id"], "response": "A"}
    ]
