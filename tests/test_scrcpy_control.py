import tempfile
import time
import unittest
from pathlib import Path

from smashbot_diagnostics.realtime import DisplayCoordinateTransform, Swipe
from smashbot_diagnostics.scrcpy_control import (
    ACTION_DOWN,
    ACTION_MOVE,
    ACTION_UP,
    POINTER_ID_GENERIC_FINGER,
    ScrcpyControlGestureController,
    ScrcpyControlError,
    control_socket_configuration,
    serialize_touch_event,
    verify_server_identity,
)


class RecordingSocket:
    def __init__(self, fail_on_write: int | None = None):
        self.writes: list[bytes] = []
        self.write_times: list[float] = []
        self.fail_on_write = fail_on_write
        self.closed = False

    def sendall(self, payload: bytes) -> None:
        if self.fail_on_write is not None and len(self.writes) + 1 == self.fail_on_write:
            raise BrokenPipeError("simulated control disconnect")
        self.write_times.append(time.monotonic())
        self.writes.append(payload)

    def shutdown(self, _how: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class ScrcpyControlTests(unittest.TestCase):
    def test_v41_socket_configuration_is_two_connections_video_then_control(self):
        self.assertEqual(
            control_socket_configuration(),
            {
                "video": True,
                "audio": False,
                "control": True,
                "socket_count": 2,
                "socket_order": ["video", "control"],
                "transport": "localhost TCP forwarded by ADB to Android localabstract socket",
                "persistent_control_connection": True,
            },
        )

    def test_v41_touch_down_golden_bytes(self):
        payload = serialize_touch_event(
            ACTION_DOWN,
            x=432,
            y=960,
            screen_width=864,
            screen_height=1920,
            pressure=1.0,
        )

        self.assertEqual(len(payload), 32)
        self.assertEqual(
            payload.hex(),
            "0200fffffffffffffffe000001b0000003c003600780ffff0000000000000000",
        )

    def test_v41_touch_up_golden_bytes_uses_zero_pressure(self):
        payload = serialize_touch_event(
            ACTION_UP,
            x=560,
            y=960,
            screen_width=864,
            screen_height=1920,
            pressure=0.0,
        )

        self.assertEqual(
            payload.hex(),
            "0201fffffffffffffffe00000230000003c00360078000000000000000000000",
        )

    def test_persistent_controller_maps_android_coordinates_and_writes_bounded_order(self):
        transform = DisplayCoordinateTransform(1080, 2400, 864, 1920)
        socket = RecordingSocket()
        controller = ScrcpyControlGestureController(
            socket,
            frame_width=864,
            frame_height=1920,
            map_swipe=transform.map_swipe,
        )

        record = controller.dispatch_swipe(Swipe(540, 1200, 700, 1200, 25))

        self.assertTrue(record["success"])
        self.assertEqual(record["event_actions"], ["DOWN", "MOVE", "MOVE", "UP"])
        self.assertEqual(record["events_written_for_gesture"], 4)
        self.assertTrue(record["bounded_event_queue"])
        self.assertEqual(record["mapped_frame_parameters"], {"x1": 432, "y1": 960, "x2": 560, "y2": 960, "duration_ms": 25})
        self.assertEqual(record["host_down_write_start_monotonic_seconds"], record["host_dispatch_start_monotonic_seconds"])
        self.assertLessEqual(record["host_dispatch_start_monotonic_seconds"], socket.write_times[0])

        first = socket.writes[0]
        last = socket.writes[-1]
        self.assertEqual(first[0:2], bytes((2, ACTION_DOWN)))
        self.assertEqual(last[0:2], bytes((2, ACTION_UP)))
        self.assertEqual(first[2:10], POINTER_ID_GENERIC_FINGER.to_bytes(8, "big"))
        self.assertEqual(first[10:22], bytes.fromhex("000001b0000003c003600780"))
        self.assertEqual(first[22:24], bytes.fromhex("ffff"))
        self.assertEqual(last[22:24], bytes.fromhex("0000"))
        self.assertGreaterEqual(record["command_duration_ms"], 20)

    def test_control_disconnect_is_reported_without_fallback(self):
        transform = DisplayCoordinateTransform(1080, 2400, 864, 1920)
        socket = RecordingSocket(fail_on_write=1)
        controller = ScrcpyControlGestureController(
            socket,
            frame_width=864,
            frame_height=1920,
            map_swipe=transform.map_swipe,
        )

        record = controller.dispatch_swipe(Swipe(540, 1200, 540, 1200, 450))
        diagnostics = controller.diagnostics()

        self.assertFalse(record["success"])
        self.assertIsNotNone(record["control_write_error"])
        self.assertEqual(record["events_written_for_gesture"], 0)
        self.assertEqual(diagnostics["unexpected_disconnects"], 1)
        self.assertEqual(diagnostics["scheduling_errors"], [])
        cleanup = controller.close()
        self.assertTrue(cleanup["cleanup_success"])
        self.assertTrue(socket.closed)
        self.assertEqual(controller.close(), cleanup)

    def test_server_identity_fails_closed_for_unpinned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scrcpy-server"
            path.write_bytes(b"not official v4.1")
            with self.assertRaises(ScrcpyControlError):
                verify_server_identity(path)


if __name__ == "__main__":
    unittest.main()
