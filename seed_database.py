import random
from datetime import datetime, timedelta

from database import create_tables, insert_incident

create_tables()

categories = {

    "Docker": {
        "service": "Docker Engine",
        "team": "DevOps Team",
        "titles": [
            "Container exited unexpectedly",
            "Image pull failed",
            "Port already allocated",
            "Docker daemon stopped",
            "Container restart loop"
        ]
    },

    "Kubernetes": {
        "service": "Kubernetes",
        "team": "Platform Team",
        "titles": [
            "CrashLoopBackOff",
            "ImagePullBackOff",
            "Pod pending",
            "Node not ready",
            "Deployment failed"
        ]
    },

    "AWS": {
        "service": "Amazon EC2",
        "team": "Cloud Team",
        "titles": [
            "EC2 unreachable",
            "IAM permission denied",
            "EBS volume full",
            "Load balancer unhealthy",
            "CloudWatch alarm triggered"
        ]
    },

    "Linux": {
        "service": "Ubuntu Server",
        "team": "Linux Team",
        "titles": [
            "Disk full",
            "Permission denied",
            "Memory usage exceeded",
            "CPU utilization high",
            "Service failed to start"
        ]
    },

    "NGINX": {
        "service": "NGINX",
        "team": "Web Team",
        "titles": [
            "502 Bad Gateway",
            "504 Gateway Timeout",
            "Configuration error",
            "Reverse proxy failure",
            "SSL certificate expired"
        ]
    },

    "Networking": {
        "service": "Enterprise Network",
        "team": "Network Team",
        "titles": [
            "DNS resolution failed",
            "TCP connection timeout",
            "Packet loss detected",
            "Gateway unreachable",
            "High network latency"
        ]
    }

}

severity_levels = [
    "Critical",
    "High",
    "Medium",
    "Low"
]

statuses = [
    "Resolved",
    "Closed"
]

symptoms = [
    "Application unavailable",
    "Connection timeout",
    "Users unable to login",
    "High CPU usage",
    "Memory exhausted",
    "Service unavailable",
    "Slow response",
    "Pod restarting continuously",
    "Container stopped",
    "DNS lookup failed"
]

root_causes = [
    "Configuration error",
    "Application bug",
    "Insufficient resources",
    "Network failure",
    "Permission issue",
    "Disk exhaustion",
    "Memory leak",
    "Image version mismatch",
    "Firewall blocked traffic",
    "Dependency failure"
]

resolutions = [
    "Restart service",
    "Update configuration",
    "Increase memory",
    "Increase disk space",
    "Restart pod",
    "Restart container",
    "Fix permissions",
    "Deploy latest version",
    "Update firewall rule",
    "Replace SSL certificate"
]

start_date = datetime(2025, 1, 1)

for i in range(100):

    category = random.choice(list(categories.keys()))

    info = categories[category]

    title = random.choice(info["titles"])

    severity = random.choice(severity_levels)

    symptom = random.choice(symptoms)

    cause = random.choice(root_causes)

    resolution = random.choice(resolutions)

    status = random.choice(statuses)

    created_at = (
        start_date +
        timedelta(days=random.randint(0, 500))
    ).strftime("%Y-%m-%d")

    insert_incident(

        title,

        category,

        info["service"],

        info["team"],

        severity,

        symptom,

        cause,

        resolution,

        status,

        created_at

    )

print("100 realistic incidents inserted successfully.")