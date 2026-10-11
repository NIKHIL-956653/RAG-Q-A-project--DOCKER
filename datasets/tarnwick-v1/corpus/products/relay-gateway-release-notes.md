# Tarnwick Relay Gateway: Release Notes

The Relay Gateway collects data from KX sensors and forwards it to Tarnwick Cloud. Releases are listed newest first. The latest release is the supported configuration.

## Version 3.4.0 (10 February 2026)

- The local message buffer now holds 72 hours of sensor data when the uplink to Tarnwick Cloud is offline (previously 24 hours).
- The default MQTT port changed from 1883 to 8883, and TLS is now enabled by default.
- Support for the discontinued KX-100 sensor has been removed.
- Added support for the KX-220 LTE-M fallback mode.

## Version 3.3.0 (2 September 2025)

- Firmware updates are now installed during a maintenance window, by default between 02:00 and 04:00 local time.
- Added an LED status code for a failed uplink: three short red flashes.
- The web console now requires a password change on first login.

## Version 3.2.0 (15 April 2025)

- Introduced a local message buffer: the gateway stores up to 24 hours of sensor data when the uplink is offline and forwards it once the connection returns.
- Each gateway supports up to 500 connected sensors.
- MQTT is published on port 1883 without TLS by default.

## Upgrade notes

Gateways are upgraded over the air. Downgrading from 3.4.0 to an earlier release is not supported because the buffer storage format changed.
