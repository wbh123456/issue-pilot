"""Schema tests for Phase 5 diagnosis, attempt-history, and evaluator contracts."""

from __future__ import annotations

import json

import pytest

from agent.state import (
    AttemptSummary,
    AttemptSummaryValidationError,
    DiagnosisValidationError,
    EvaluationValidationError,
    PatchEvaluation,
    StructuredDiagnosis,
    initial_state,
    parse_attempt_history,
    parse_attempt_summary,
    parse_patch_evaluation,
    parse_structured_diagnosis,
    patch_evaluation_passed,
)

VALID_DIAGNOSIS = {
    "root_cause": "decode_token does not catch ExpiredSignatureError",
    "failure_category": "WRONG_HYPOTHESIS",
    "new_hypothesis": "Expired JWT must map to 401",
    "next_actions": ["catch ExpiredSignatureError", "return 401"],
}

VALID_EVALUATION = {
    "issue_resolved": True,
    "patch_scope": "appropriate",
    "regression_risk": "low",
    "missing_tests": False,
    "feedback": "",
}

VALID_ATTEMPT = {
    "attempt_index": 0,
    "hypothesis": "missing catch",
    "deterministic_pass": False,
    "evaluator_pass": None,
    "failure_source": "deterministic",
    "failure_category": "TEST_FAILURE",
    "root_cause": "pytest failed",
}


def _round_trip(parse, payload: dict) -> dict:
    parsed = parse(payload)
    dumped = json.dumps(parsed.model_dump())
    return parse(json.loads(dumped)).model_dump()


class TestStructuredDiagnosis:
    def test_valid_parse_and_json_round_trip(self) -> None:
        parsed = parse_structured_diagnosis(VALID_DIAGNOSIS)
        assert parsed.failure_category == "WRONG_HYPOTHESIS"
        assert _round_trip(parse_structured_diagnosis, VALID_DIAGNOSIS) == parsed.model_dump()
        fenced = json.dumps(VALID_DIAGNOSIS)
        assert parse_structured_diagnosis(fenced).new_hypothesis == VALID_DIAGNOSIS["new_hypothesis"]

    def test_instance_passthrough(self) -> None:
        model = StructuredDiagnosis.model_validate(VALID_DIAGNOSIS)
        assert parse_structured_diagnosis(model) is model

    def test_reject_prose_missing_fields_and_extras(self) -> None:
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis("the tests failed because of auth")
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis("")
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis({"root_cause": "x", "failure_category": "BAD_PATCH"})
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis({**VALID_DIAGNOSIS, "extra_key": "nope"})
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis({**VALID_DIAGNOSIS, "root_cause": ""})
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis({**VALID_DIAGNOSIS, "new_hypothesis": ""})
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis(
                {**VALID_DIAGNOSIS, "failure_category": "TIMEOUT"}
            )
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis({**VALID_DIAGNOSIS, "next_actions": []})
        with pytest.raises(DiagnosisValidationError):
            parse_structured_diagnosis(
                {
                    **VALID_DIAGNOSIS,
                    "next_actions": ["a", "b", "c", "d", "e", "f"],
                }
            )


class TestPatchEvaluation:
    def test_valid_parse_and_json_round_trip(self) -> None:
        parsed = parse_patch_evaluation(VALID_EVALUATION)
        assert parsed.issue_resolved is True
        assert _round_trip(parse_patch_evaluation, VALID_EVALUATION) == parsed.model_dump()

    def test_instance_passthrough(self) -> None:
        model = PatchEvaluation.model_validate(VALID_EVALUATION)
        assert parse_patch_evaluation(model) is model

    def test_reject_prose_extras_and_invalid_enums(self) -> None:
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation("the patch looks good")
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation("")
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation({"issue_resolved": True})
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation({**VALID_EVALUATION, "passed": True})
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation({**VALID_EVALUATION, "patch_scope": "ok"})
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation({**VALID_EVALUATION, "regression_risk": "none"})
        with pytest.raises(EvaluationValidationError):
            parse_patch_evaluation({**VALID_EVALUATION, "feedback": "x" * 801})

    def test_mechanical_pass_all_green(self) -> None:
        assert patch_evaluation_passed(VALID_EVALUATION) is True
        assert patch_evaluation_passed(parse_patch_evaluation(VALID_EVALUATION)) is True

    @pytest.mark.parametrize(
        "override",
        [
            {"issue_resolved": False},
            {"patch_scope": "too_broad"},
            {"patch_scope": "too_narrow"},
            {"patch_scope": "unrelated"},
            {"regression_risk": "medium"},
            {"regression_risk": "high"},
            {"missing_tests": True},
        ],
    )
    def test_mechanical_fail_each_blocking_negative(self, override: dict) -> None:
        payload = {**VALID_EVALUATION, **override}
        assert patch_evaluation_passed(payload) is False


class TestAttemptSummary:
    def test_valid_parse_and_json_round_trip(self) -> None:
        parsed = parse_attempt_summary(VALID_ATTEMPT)
        assert parsed.attempt_index == 0
        assert parsed.evaluator_pass is None
        assert _round_trip(parse_attempt_summary, VALID_ATTEMPT) == parsed.model_dump()

    def test_instance_passthrough(self) -> None:
        model = AttemptSummary.model_validate(VALID_ATTEMPT)
        assert parse_attempt_summary(model) is model

    def test_reject_prose_extras_and_bounds(self) -> None:
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary("first attempt failed")
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary("")
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({"attempt_index": 0})
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({**VALID_ATTEMPT, "extra_key": "nope"})
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({**VALID_ATTEMPT, "attempt_index": 3})
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({**VALID_ATTEMPT, "attempt_index": -1})
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({**VALID_ATTEMPT, "failure_source": "human"})
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_summary({**VALID_ATTEMPT, "failure_category": "TIMEOUT"})

    def test_history_round_trip_and_max_length(self) -> None:
        items = [
            {**VALID_ATTEMPT, "attempt_index": 0},
            {**VALID_ATTEMPT, "attempt_index": 1, "hypothesis": "retry catch"},
            {**VALID_ATTEMPT, "attempt_index": 2, "failure_source": "evaluator"},
        ]
        parsed = parse_attempt_history(items)
        assert len(parsed) == 3
        dumped = json.dumps([item.model_dump() for item in parsed])
        assert len(parse_attempt_history(json.loads(dumped))) == 3
        assert parse_attempt_history([]) == []
        with pytest.raises(AttemptSummaryValidationError, match="max_length=3"):
            parse_attempt_history(items + [{**VALID_ATTEMPT, "attempt_index": 0}])
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_history("not a list")
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_history("")
        with pytest.raises(AttemptSummaryValidationError):
            parse_attempt_history([VALID_ATTEMPT, "prose"])


class TestInitialStateSerialization:
    def test_json_dumps_and_excludes_runtime_objects(self) -> None:
        state = initial_state("Expired JWT returns 500")
        encoded = json.dumps(state)
        loaded = json.loads(encoded)
        assert loaded["issue"] == "Expired JWT returns 500"
        assert loaded["diagnosis"] == ""
        assert loaded["structured_diagnosis"] == {}
        assert loaded["attempt_history"] == []
        assert loaded["patch_evaluation"] == {}
        assert loaded["retry_count"] == 0
        assert loaded["human_retry_count"] == 0
        assert loaded["human_feedback"] == ""
        for forbidden in ("client", "sandbox", "progress", "feedback_provider"):
            assert forbidden not in loaded
            assert forbidden not in state
