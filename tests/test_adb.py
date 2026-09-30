import unittest

from smashbot_diagnostics.adb import AdbClient, AdbError


class AdbTransportTests(unittest.TestCase):
    def test_wireless_preference_is_recorded_for_network_serial(self):
        client = AdbClient("/bin/true", transport="wireless_tcp")
        selected = client.select_ready_device(
            [{"serial": "192.168.1.42:37123", "state": "device", "details": {}}]
        )
        self.assertEqual(selected, "192.168.1.42:37123")
        self.assertEqual(client.transport_info()["effective"], "wireless_tcp")

    def test_conflicting_known_transport_is_rejected(self):
        client = AdbClient("/bin/true", transport="usb")
        with self.assertRaises(AdbError):
            client.select_ready_device(
                [{"serial": "192.168.1.42:37123", "state": "device", "details": {}}]
            )


if __name__ == "__main__":
    unittest.main()
