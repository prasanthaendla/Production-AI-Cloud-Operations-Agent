# EC2 Application Degradation Runbook

## Problem

An EC2 instance is running but the application is unhealthy.

## Initial Checks

1. Check CPU utilization.
2. Check memory utilization.
3. Check application health.
4. Check application logs.
5. Check database connectivity.
6. Check connection pool utilization.
7. Check recent deployments.

## High CPU and Memory

If CPU utilization is consistently high and memory utilization is also elevated:

- Check application workload.
- Check application logs for timeouts.
- Check database connection behavior.
- Check recent deployments.
- Do not immediately assume that CPU is the root cause.

## Database Connection Errors

If application logs contain database connection timeout errors:

- Check database availability.
- Check database connection count.
- Check application connection pool utilization.
- Check database latency.
- Check for recent configuration or deployment changes.

## Recent Deployments

If a recent deployment failed:

- Investigate the deployment.
- Review application errors after deployment.
- Compare the deployed version with the previous known-good version.

If recent deployments were successful:

- Do not treat deployment failure as the primary hypothesis.
- Continue investigating infrastructure and application evidence.

## Important

A single metric should not automatically be treated as the root cause.

The root cause should be determined using multiple pieces of evidence.
