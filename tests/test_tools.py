from src.tools.cloud_tools import (
    get_instance_health,
    get_application_logs,
    get_recent_deployments,
)

def test_get_instance_health():
    result = get_instance_health("i-demo-001")

    assert result["instance_id"] == "i-demo-001"
    assert result["status"] == "running"
    assert result["health"] == "degraded"
    assert result["cpu_utilization"] > 90


def test_get_application_logs():
    result = get_application_logs("i-demo-001")

    assert result["instance_id"] == "i-demo-001"
    assert result["log_count"] == 4
    assert len(result["logs"]) == 4

    assert result["logs"][0]["level"] == "ERROR"
    assert "Database connection timeout" in (
        result["logs"][0]["message"]
    )

def test_get_recent_deployments():
    result = get_recent_deployments("i-demo-001")

    assert result["instance_id"] == "i-demo-001"
    assert result["deployment_count"] == 2
    assert len(result["deployments"]) == 2

    assert result["deployments"][0]["deployment_id"] == (
        "deploy-184"
    )

    assert result["deployments"][0]["version"] == "v2.8.1"

    assert result["deployments"][0]["status"] == "SUCCESS"    