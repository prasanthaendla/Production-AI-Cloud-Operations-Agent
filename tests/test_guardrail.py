"""
Tests for the input guardrail.
"""

from src.guardrails.input_guardrail import InputGuardrail


def test_cloud_operations_question_is_allowed():
    guardrail = InputGuardrail()

    assert guardrail.check(
        "Why is instance i-demo-001 unhealthy?"
    )


def test_cpu_question_is_allowed():
    guardrail = InputGuardrail()

    assert guardrail.check(
        "What is the CPU utilization?"
    )


def test_application_logs_question_is_allowed():
    guardrail = InputGuardrail()

    assert guardrail.check(
        "What errors are occurring in the application logs?"
    )


def test_deployment_question_is_allowed():
    guardrail = InputGuardrail()

    assert guardrail.check(
        "Was there a recent deployment?"
    )


def test_personal_question_is_rejected():
    guardrail = InputGuardrail()

    assert not guardrail.check(
        "Do you know what my name is?"
    )


def test_general_ai_question_is_rejected():
    guardrail = InputGuardrail()

    assert not guardrail.check(
        "What is Agentic AI?"
    )


def test_empty_question_is_rejected():
    guardrail = InputGuardrail()

    assert not guardrail.check("")