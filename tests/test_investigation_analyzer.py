"""
Tests for InvestigationAnalyzer.
"""

from src.agent.investigation_analyzer import (
    InvestigationAnalyzer,
)


def test_high_cpu_and_memory_generate_findings():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {
                "instance_id": "i-demo-001",
            },
            "result": {
                "instance_id": "i-demo-001",
                "status": "running",
                "health": "degraded",
                "cpu_utilization": 92.4,
                "memory_utilization": 81.7,
                "network_status": "normal",
                "application_status": "unhealthy",
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert "High CPU utilization detected: 92.4%." in findings

    assert "High memory utilization detected: 81.7%." in findings

    assert "Instance health is degraded." in findings

    assert "Application status is unhealthy." in findings


def test_normal_instance_generates_no_findings():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {
                "instance_id": "i-demo-002",
            },
            "result": {
                "instance_id": "i-demo-002",
                "status": "running",
                "health": "healthy",
                "cpu_utilization": 35.0,
                "memory_utilization": 42.0,
                "network_status": "normal",
                "application_status": "healthy",
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert findings == []


def test_unhealthy_network_generates_finding():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {
                "instance_id": "i-demo-003",
            },
            "result": {
                "instance_id": "i-demo-003",
                "status": "running",
                "health": "degraded",
                "cpu_utilization": 45.0,
                "memory_utilization": 50.0,
                "network_status": "unavailable",
                "application_status": "healthy",
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert "Network status is unavailable." in findings


def test_application_log_errors_generate_findings():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_application_logs",
            "arguments": {
                "instance_id": "i-demo-001",
            },
            "result": {
                "errors": [
                    "Database connection timeout",
                    "HTTP 500 error",
                ],
                "warnings": [
                    "Connection pool exhausted",
                ],
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert (
        "Application errors detected: "
        "['Database connection timeout', "
        "'HTTP 500 error']"
        in findings
    )

    assert (
        "Application warnings detected: "
        "['Connection pool exhausted']"
        in findings
    )


def test_failed_deployment_generates_finding():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_recent_deployments",
            "arguments": {
                "instance_id": "i-demo-001",
            },
            "result": {
                "status": "failed",
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert (
        "A recent deployment failed "
        "and may be related to the incident."
        in findings
    )


def test_unknown_tool_is_ignored():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "unknown_tool",
            "arguments": {},
            "result": {
                "some_value": "something",
            },
        }
    ]

    findings = analyzer.analyze(evidence)

    assert findings == []


def test_invalid_tool_result_is_ignored():

    analyzer = InvestigationAnalyzer()

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {},
            "result": "invalid-result",
        }
    ]

    findings = analyzer.analyze(evidence)

    assert findings == []