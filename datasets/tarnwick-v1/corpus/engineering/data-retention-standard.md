# Data Retention Standard

Document owner: Security and Compliance
Version: 2.1 (approved 3 November 2025)

## Purpose

This standard defines how long Tarnwick Cloud keeps each category of data, and how deletion requests are handled.

## Retention periods

- Application logs are retained for 90 days and then deleted automatically.
- Audit logs, which record administrative actions, are retained for 400 days.
- Database backups are taken daily and retained for 35 days. In addition, one backup per month is kept for 12 months.
- Customer sensor telemetry is retained for 25 months by default. Customers on the Enterprise plan can configure a retention period between 6 and 60 months.
- Support tickets are retained for three years after they are closed.

## Deletion requests

A verified customer deletion request is completed within 30 days. Deleted data can remain in database backups until those backups expire under the backup retention period above; it is not restored into production.

## Exceptions

Data under a legal hold is excluded from automatic deletion until Legal releases the hold. Exceptions to this standard must be approved by the Head of Security and Compliance.
