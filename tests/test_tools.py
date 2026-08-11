from src.tools.cloud_tools import get_instance_health


def test_get_instance_health():
    result = get_instance_health("i-demo-001")

    assert result["instance_id"] == "i-demo-001"
    assert result["status"] == "running"
    assert result["health"] == "degraded"
    assert result["cpu_utilization"] > 90