import unittest

from smashbot_diagnostics.task018_temporal import (
    TemporalConfirmedStateMachine,
    Task018Error,
)


class Task018StateMachineTests(unittest.TestCase):
    def test_acquire_seed_is_internal_and_tentative_does_not_emit(self):
        machine = TemporalConfirmedStateMachine()
        first = machine.step(0, 100, global_seed=(10.0, 20.0))
        self.assertEqual(first.state_after, "TENTATIVE")
        self.assertEqual(first.path, "global")
        self.assertIsNone(first.emitted)
        self.assertEqual(first.event, "internal_seed")

    def test_tentative_local_confirmation_is_first_emission(self):
        machine = TemporalConfirmedStateMachine()
        machine.step(10, 1000, global_seed=(10.0, 20.0))
        second = machine.step(11, 2000, local_observation=(11.0, 21.0))
        self.assertEqual(second.state_after, "TRACK")
        self.assertEqual(second.path, "local")
        self.assertEqual(second.emitted, (11.0, 21.0))

    def test_tentative_miss_discards_seed(self):
        machine = TemporalConfirmedStateMachine()
        machine.step(0, 1, global_seed=(1.0, 1.0))
        missed = machine.step(1, 2, local_observation=None)
        self.assertEqual(missed.state_after, "ACQUIRE")
        self.assertIsNone(missed.emitted)
        self.assertIsNone(machine.seed)

    def test_track_coast_and_reacquire_limits(self):
        machine = TemporalConfirmedStateMachine()
        machine.step(0, 1, global_seed=(1.0, 1.0))
        machine.step(1, 2, local_observation=(1.0, 1.0))
        self.assertEqual(machine.step(2, 3).state_after, "COAST")
        self.assertEqual(machine.step(3, 4).state_after, "COAST")
        self.assertEqual(machine.step(4, 5).state_after, "REACQUIRE")
        reacquire = machine.step(5, 6, global_seed=(5.0, 5.0))
        self.assertEqual(reacquire.state_after, "TENTATIVE")
        self.assertIsNone(reacquire.emitted)

    def test_latest_frame_semantics_reject_non_monotonic_input(self):
        machine = TemporalConfirmedStateMachine()
        machine.step(2, 20, global_seed=(1.0, 1.0))
        with self.assertRaises(Task018Error):
            machine.step(2, 21, global_seed=None)


if __name__ == "__main__":
    unittest.main()
