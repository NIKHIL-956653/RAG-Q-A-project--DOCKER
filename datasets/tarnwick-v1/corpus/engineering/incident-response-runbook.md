# Incident Response Runbook

Document owner: Site Reliability Engineering
Applies to: Tarnwick Cloud production services

## Severity levels

- SEV-1: complete outage of Tarnwick Cloud for customers, or any confirmed loss of customer data.
- SEV-2: major degradation, such as delayed sensor data ingestion or a failed region, with a workaround available.
- SEV-3: minor issue affecting a small number of customers or internal tools only.

## Paging and escalation

A SEV-1 alert pages the primary on-call engineer immediately. If the page is not acknowledged within 5 minutes, it escalates to the secondary on-call engineer. An incident commander must be assigned within 15 minutes of the first page for every SEV-1.

SEV-2 alerts page the primary on-call engineer during working hours and create a ticket outside working hours.

## Customer communication

For a SEV-1, the public status page must be updated within 30 minutes of the incident being declared, and then at least every hour until the incident is resolved. Customer success managers inform Enterprise customers directly.

## After the incident

Every SEV-1 and SEV-2 incident requires a written postmortem. The postmortem must be published within five working days of the incident being resolved. SEV-1 postmortems are approved by the VP of Engineering; SEV-2 postmortems are approved by the engineering manager of the affected team.

Postmortems are blameless: they describe systems and decisions, not individual mistakes. Every postmortem action item must have a named owner and a due date.
