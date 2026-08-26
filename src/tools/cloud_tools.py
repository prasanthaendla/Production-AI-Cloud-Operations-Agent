"""
Cloud Operations Tools

Step 20 - Real AWS Data

Real AWS EC2 instance health and CPU monitoring.

Current real-data integrations:
- EC2 instance state
- EC2 system status
- EC2 instance status
- CloudWatch CPU utilization

Application logs and deployments remain unchanged for now.
They will be connected to real AWS services in later validation
work only if required.

No mock data is used by get_instance_health().
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError


# --------------------------------------------------
# AWS Clients
# --------------------------------------------------

ec2 = boto3.client("ec2")
cloudwatch = boto3.client("cloudwatch")


# --------------------------------------------------
# EC2 Instance Health
# --------------------------------------------------

def get_instance_health(
    instance_id: str,
) -> Dict[str, Any]:
    """
    Return real AWS EC2 instance health information.

    Data sources:
        EC2:
            - Instance state
            - System status check
            - Instance status check

        CloudWatch:
            - CPUUtilization

    Args:
        instance_id:
            AWS EC2 instance ID.

    Returns:
        Dictionary containing real AWS health information.
    """

    if not instance_id:
        raise ValueError(
            "Instance ID cannot be empty."
        )

    try:

        # --------------------------------------------------
        # Get EC2 instance information
        # --------------------------------------------------

        response = ec2.describe_instances(
            InstanceIds=[instance_id]
        )

        reservations = response.get(
            "Reservations",
            [],
        )

        if not reservations:
            return {
                "instance_id": instance_id,
                "status": "unknown",
                "health": "unknown",
                "message": (
                    "Instance was not found "
                    "in the configured AWS region."
                ),
            }

        instances = reservations[0].get(
            "Instances",
            [],
        )

        if not instances:
            return {
                "instance_id": instance_id,
                "status": "unknown",
                "health": "unknown",
                "message": (
                    "Instance was not found "
                    "in the configured AWS region."
                ),
            }

        instance = instances[0]

        instance_state = (
            instance.get(
                "State",
                {},
            ).get(
                "Name",
                "unknown",
            )
        )

        # --------------------------------------------------
        # Get EC2 status checks
        # --------------------------------------------------

        status_response = (
            ec2.describe_instance_status(
                InstanceIds=[instance_id],
                IncludeAllInstances=True,
            )
        )

        statuses = status_response.get(
            "InstanceStatuses",
            [],
        )

        system_status = "unknown"
        instance_status = "unknown"

        if statuses:

            status = statuses[0]

            system_status = (
                status.get(
                    "SystemStatus",
                    {},
                ).get(
                    "Status",
                    "unknown",
                )
            )

            instance_status = (
                status.get(
                    "InstanceStatus",
                    {},
                ).get(
                    "Status",
                    "unknown",
                )
            )

        # --------------------------------------------------
        # Determine overall health
        # --------------------------------------------------

        if (
            system_status == "ok"
            and instance_status == "ok"
        ):

            health = "healthy"

        elif (
            system_status == "impaired"
            or instance_status == "impaired"
        ):

            health = "degraded"

        elif instance_state != "running":

            health = "not_running"

        else:

            health = "unknown"

        # --------------------------------------------------
        # Get CloudWatch CPU utilization
        # --------------------------------------------------

        cpu_utilization = _get_cpu_utilization(
            instance_id
        )

        # --------------------------------------------------
        # Network status
        # --------------------------------------------------

        if (
            system_status == "ok"
            and instance_status == "ok"
        ):

            network_status = "normal"

        elif (
            system_status == "impaired"
            or instance_status == "impaired"
        ):

            network_status = "impaired"

        else:

            network_status = "unknown"

        # --------------------------------------------------
        # Memory and application status
        # --------------------------------------------------
        #
        # EC2 does not provide memory utilization through
        # standard EC2 metrics.
        #
        # Memory requires CloudWatch Agent/custom metrics.
        #
        # Application health is also not directly provided
        # by EC2.
        #

        memory_utilization = None

        application_status = "unknown"

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {
            "instance_id": instance_id,
            "status": instance_state,
            "health": health,
            "system_status": system_status,
            "instance_status": instance_status,
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "network_status": network_status,
            "application_status": application_status,
            "data_source": "AWS EC2 + CloudWatch",
        }

    except ClientError as exc:

        error = exc.response.get(
            "Error",
            {},
        )

        error_code = error.get(
            "Code",
            "Unknown",
        )

        error_message = error.get(
            "Message",
            str(exc),
        )

        raise RuntimeError(
            f"AWS EC2 API error "
            f"{error_code}: "
            f"{error_message}"
        ) from exc


# --------------------------------------------------
# CloudWatch CPU
# --------------------------------------------------

def _get_cpu_utilization(
    instance_id: str,
) -> float | None:
    """
    Get the latest average CPU utilization
    from Amazon CloudWatch.

    Looks back over the last 10 minutes.

    Returns:
        Latest available CPU percentage,
        or None if no metric data is available.
    """

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        - timedelta(minutes=10)
    )

    response = (
        cloudwatch.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[
                {
                    "Name": "InstanceId",
                    "Value": instance_id,
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=["Average"],
            Unit="Percent",
        )
    )

    datapoints = response.get(
        "Datapoints",
        [],
    )

    if not datapoints:
        return None

    latest = max(
        datapoints,
        key=lambda item: item[
            "Timestamp"
        ],
    )

    average = latest.get(
        "Average"
    )

    if average is None:
        return None

    return round(
        float(average),
        2,
    )


# --------------------------------------------------
# Application Logs
# --------------------------------------------------

def get_application_logs(
    instance_id: str,
) -> Dict[str, Any]:
    """
    Temporary application log implementation.

    Real CloudWatch Logs integration will be added
    separately after EC2 health validation.
    """

    return {
        "instance_id": instance_id,
        "log_count": 0,
        "logs": [],
        "message": (
            "Application logs are not connected "
            "to real AWS CloudWatch Logs yet."
        ),
        "data_source": "not_connected",
    }


# --------------------------------------------------
# Recent Deployments
# --------------------------------------------------

def get_recent_deployments(
    instance_id: str,
) -> Dict[str, Any]:
    """
    Temporary deployment implementation.

    Real deployment history will be connected to
    the appropriate AWS deployment service later.
    """

    return {
        "instance_id": instance_id,
        "deployment_count": 0,
        "deployments": [],
        "message": (
            "Deployment history is not connected "
            "to a real AWS deployment service yet."
        ),
        "data_source": "not_connected",
    }