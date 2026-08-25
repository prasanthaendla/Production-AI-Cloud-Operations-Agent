from src.memory.incident_memory import (
    IncidentMemory,
)


def test_memory_starts_empty(tmp_path):
    memory = IncidentMemory(
        storage_path=tmp_path / "incidents.json"
    )

    assert memory.count() == 0


def test_incident_is_saved(tmp_path):
    memory = IncidentMemory(
        storage_path=tmp_path / "incidents.json"
    )

    incident = memory.save_incident(
        question="Why is instance unhealthy?",
        capability="instance_health",
        findings=[
            "High CPU utilization detected: 92.4%.",
            "Application status is unhealthy.",
        ],
        hypotheses=[
            {
                "hypothesis": (
                    "Resource saturation may be "
                    "contributing to application degradation."
                )
            }
        ],
        root_cause_assessment={
            "root_cause": "Resource saturation",
            "score": 4,
        },
        confidence_assessment={
            "confidence_level": "HIGH",
            "confidence_score": 0.9,
        },
    )

    assert incident["incident_id"] == "INC-00001"
    assert memory.count() == 1


def test_similar_incident_is_retrieved(tmp_path):
    memory = IncidentMemory(
        storage_path=tmp_path / "incidents.json"
    )

    memory.save_incident(
        question="Why is instance unhealthy?",
        capability="instance_health",
        findings=[
            "High CPU utilization detected.",
            "High memory utilization detected.",
            "Application status is unhealthy.",
        ],
        hypotheses=[],
        root_cause_assessment={
            "root_cause": "Resource saturation",
            "score": 4,
        },
        confidence_assessment={
            "confidence_level": "HIGH",
            "confidence_score": 0.9,
        },
    )

    results = memory.retrieve_similar(
        question=(
            "Why is the instance unhealthy "
            "with high CPU utilization?"
        ),
        findings=[
            "Application status is unhealthy."
        ],
        capability="instance_health",
    )

    assert results
    assert results[0]["incident_id"] == "INC-00001"


def test_unrelated_incident_is_not_returned(tmp_path):
    memory = IncidentMemory(
        storage_path=tmp_path / "incidents.json"
    )

    memory.save_incident(
        question="Why did deployment fail?",
        capability="deployments",
        findings=[
            "A recent deployment failed."
        ],
        hypotheses=[],
        root_cause_assessment={
            "root_cause": "Deployment failure",
            "score": 4,
        },
        confidence_assessment={
            "confidence_level": "HIGH",
            "confidence_score": 0.9,
        },
    )

    results = memory.retrieve_similar(
        question=(
            "Why is instance CPU utilization high?"
        ),
        findings=[
            "High CPU utilization detected."
        ],
        capability="instance_health",
    )

    assert results == []


def test_memory_persists_between_instances(tmp_path):
    storage = tmp_path / "incidents.json"

    memory_one = IncidentMemory(
        storage_path=storage
    )

    memory_one.save_incident(
        question="Why is EC2 unhealthy?",
        capability="instance_health",
        findings=[
            "High CPU utilization detected."
        ],
        hypotheses=[],
        root_cause_assessment={
            "root_cause": "Resource saturation",
            "score": 4,
        },
        confidence_assessment={
            "confidence_level": "HIGH",
            "confidence_score": 0.9,
        },
    )

    memory_two = IncidentMemory(
        storage_path=storage
    )

    assert memory_two.count() == 1
