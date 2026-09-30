import unittest

from smashbot_diagnostics.parsing import (
    parse_adb_version,
    parse_devices,
    parse_display_sizes,
    parse_getprop,
    parse_package_info,
    parse_refresh_rates,
    parse_density_output,
    transport_info,
)


class ParsingTests(unittest.TestCase):
    def test_adb_devices_preserves_state_and_details(self):
        devices = parse_devices(
            """List of devices attached
            emulator-5554\tdevice product:sdk_gphone_x86 model:sdk_gphone_x86 device:generic_x86
            USB123\tunauthorized usb:1-1
            """
        )
        self.assertEqual(devices[0]["state"], "device")
        self.assertEqual(devices[0]["details"]["model"], "sdk_gphone_x86")
        self.assertEqual(devices[0]["transport"]["detected"], "emulator")
        self.assertEqual(devices[1]["state"], "unauthorized")

    def test_wireless_transport_is_detected_from_network_serial(self):
        info = transport_info("192.168.1.42:37123")
        self.assertEqual(info["detected"], "wireless_tcp")
        self.assertTrue(info["network_endpoint"])
        self.assertEqual(transport_info("SERIAL123")["detected"], "unknown")

    def test_device_property_parsers(self):
        self.assertEqual(parse_adb_version("Android Debug Bridge version 35.0.1\n"), "Android Debug Bridge version 35.0.1")
        self.assertEqual(parse_getprop("[ro.build.version.sdk]: [34]\n"), {"ro.build.version.sdk": "34"})
        self.assertEqual(
            parse_display_sizes("Physical size: 1080x2400\nOverride size: 720x1600\n"),
            {"physical": {"width": 1080, "height": 2400}, "override": {"width": 720, "height": 1600}},
        )
        self.assertEqual(parse_density_output("Physical density: 420\nOverride density: 320\n"), {"physical": 420, "override": 320})
        self.assertEqual(parse_density_output("Override density: 320\n"), {"override": 320})
        self.assertEqual(parse_refresh_rates("mRefreshRate=60.0\npeakRefreshRate=120.0Hz\n"), [60.0, 120.0])

    def test_package_versions(self):
        parsed = parse_package_info("versionCode=42 minSdk=26\nversionName=1.2.3\n")
        self.assertEqual(parsed, {"version_name": "1.2.3", "version_code": 42})


if __name__ == "__main__":
    unittest.main()
