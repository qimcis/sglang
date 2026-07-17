"""Unit tests for Mooncake checksum-manifest capability negotiation."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.srt.mem_cache.kv_page_tags import KVProtectionConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeChecksumManifest(CustomTestCase):
    def _manager_and_socket(self):
        manager = object.__new__(MooncakeKVManager)
        socket = MagicMock()
        manager._socket_lock = threading.Lock()
        manager._socket_send_locks = {}
        manager._connect = MagicMock(return_value=socket)
        return manager, socket

    def _decode_manager(
        self, *, allow_legacy=False, attention_tags=False, checksum=True
    ):
        manager = object.__new__(MooncakeKVManager)
        config = KVProtectionConfig(
            enable_attention_tags=attention_tags,
            enable_transfer_checksum=checksum,
            allow_legacy_completion=allow_legacy,
        )
        manager.kv_args = SimpleNamespace(
            transfer_page_tag_manager=SimpleNamespace(config=config)
        )
        manager.request_status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.checksum_manifest_table = defaultdict(dict)
        manager.checksum_nonce_table = {7: 9}
        manager.legacy_checksum_rooms = set()
        manager.prefill_response_tracker = defaultdict(set)
        manager.required_prefill_response_num_table = {7: 1}
        manager.transfer_page_tag_expected_table = {}
        manager.transfer_page_tag_seen_table = {}
        manager.transfer_page_tag_event_table = {}
        manager.enable_staging = False
        manager._chunk_writer_counts = {}
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock()
        return manager

    def test_legacy_receiver_gets_three_frame_completion(self):
        manager, socket = self._manager_and_socket()
        plan = MagicMock()
        manager.sync_status_to_decode_endpoint(
            "127.0.0.1",
            1234,
            7,
            KVPoll.Success,
            0,
            checksum_plan=plan,
            transfer_nonce=0,
        )
        frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(len(frames), 3)
        plan.to_wire_bytes.assert_not_called()

    def test_capable_receiver_gets_atomic_manifest_completion(self):
        manager, socket = self._manager_and_socket()
        plan = MagicMock()
        plan.to_wire_bytes.return_value = b"manifest"
        manager.sync_status_to_decode_endpoint(
            "127.0.0.1",
            1234,
            7,
            KVPoll.Success,
            0,
            checksum_plan=plan,
            transfer_nonce=9,
        )
        frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(
            frames,
            [
                b"7",
                str(KVPoll.Success).encode("ascii"),
                b"0",
                manager.CHECKSUM_MANIFEST_HEADER,
                b"manifest",
            ],
        )
        plan.to_wire_bytes.assert_called_once_with(transfer_nonce=9)

    def test_attention_only_receiver_gets_nonce_completion(self):
        manager, socket = self._manager_and_socket()
        manager.sync_status_to_decode_endpoint(
            "127.0.0.1",
            1234,
            7,
            KVPoll.Success,
            0,
            checksum_plan=None,
            transfer_nonce=9,
        )
        frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(
            frames,
            [
                b"7",
                str(KVPoll.Success).encode("ascii"),
                b"0",
                manager.PROTECTION_NONCE_HEADER,
                b"9",
            ],
        )

    def test_legacy_receiver_gets_four_frame_page_tags(self):
        manager, socket = self._manager_and_socket()
        page_ids = np.asarray([2], dtype=np.int32)
        tags = np.asarray([3], dtype=np.int32)

        manager.sync_transfer_page_tags_to_decode_endpoint(
            "127.0.0.1", 1234, 7, page_ids, tags, transfer_nonce=0
        )

        frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(
            frames,
            [
                manager.TRANSFER_PAGE_TAG_HEADER,
                b"7",
                page_ids.tobytes(),
                tags.tobytes(),
            ],
        )

    def test_protected_receiver_rejects_legacy_completion_by_default(self):
        manager = self._decode_manager()

        manager._handle_decode_completion(
            [b"7", str(KVPoll.Success).encode("ascii"), b"0"]
        )

        self.assertNotIn(7, manager.legacy_checksum_rooms)
        manager.update_status.assert_called_once_with(7, KVPoll.Failed)
        self.assertIn(
            "legacy protection completion is disabled",
            manager.record_failure.call_args.args[1],
        )

    def test_compatibility_mode_accepts_legacy_completion(self):
        manager = self._decode_manager(allow_legacy=True)

        manager._handle_decode_completion(
            [b"7", str(KVPoll.Success).encode("ascii"), b"0"]
        )

        self.assertIn(7, manager.legacy_checksum_rooms)
        manager.update_status.assert_called_once_with(7, KVPoll.Success)

    def test_attention_only_completion_validates_nonce(self):
        manager = self._decode_manager(attention_tags=True, checksum=False)

        manager._handle_decode_completion(
            [
                b"7",
                str(KVPoll.Success).encode("ascii"),
                b"0",
                manager.PROTECTION_NONCE_HEADER,
                b"9",
            ]
        )

        manager.update_status.assert_called_once_with(7, KVPoll.Success)

    def test_staging_room_defers_transfer_tag_completeness(self):
        manager = self._decode_manager(attention_tags=True, checksum=False)
        manager.transfer_page_tag_expected_table = {7: {2: 3}}
        manager.enable_staging = True
        manager._staging_handler = MagicMock()
        manager._staging_handler.is_staging_room.return_value = True

        manager._handle_decode_completion(
            [
                b"7",
                str(KVPoll.Success).encode("ascii"),
                b"0",
                manager.PROTECTION_NONCE_HEADER,
                b"9",
            ]
        )

        manager._staging_handler.submit_last_scatter_async.assert_called_once_with(7)
        manager.update_status.assert_called_once_with(7, KVPoll.Success)

    def test_receiver_clear_waits_for_transfer_tag_write(self):
        manager = self._decode_manager()
        event = MagicMock()
        manager.transfer_page_tag_event_table[7] = event
        manager.req_to_decode_prefix_len = {}
        manager.transfer_infos = {}
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 7

        receiver.clear()

        event.synchronize.assert_called_once_with()
        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.transfer_page_tag_event_table)

    def test_malformed_decode_message_is_rejected(self):
        manager = self._decode_manager()

        with self.assertRaises(ValueError):
            manager._handle_decode_message(
                [b"not-a-room", str(KVPoll.Success).encode("ascii"), b"0"]
            )

    def test_transfer_page_tags_require_active_nonce(self):
        manager = object.__new__(MooncakeKVManager)
        tag_manager = MagicMock()
        manager.kv_args = MagicMock(transfer_page_tag_manager=tag_manager)
        manager.request_status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.checksum_nonce_table = {7: 9}
        manager.transfer_page_tag_expected_table = {7: {2: 3}}
        manager.transfer_page_tag_seen_table = {7: set()}
        manager.transfer_page_tag_event_table = {}
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        page_ids = np.asarray([2], dtype=np.int32).tobytes()
        tags = np.asarray([3], dtype=np.int32).tobytes()

        manager._handle_transfer_page_tags(
            [manager.TRANSFER_PAGE_TAG_HEADER, b"7", b"9", page_ids, tags]
        )
        tag_manager.write_transfer_page_tags.assert_called_once()
        self.assertEqual(manager.transfer_page_tag_seen_table[7], {2})

        manager._handle_transfer_page_tags(
            [manager.TRANSFER_PAGE_TAG_HEADER, b"7", b"10", page_ids, tags]
        )
        self.assertEqual(manager.request_status[7], KVPoll.Failed)

    def test_receiver_accepts_idempotent_duplicate_expected_page_tags(self):
        manager = self._decode_manager(attention_tags=True, checksum=False)
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_infos = []
        receiver.bootstrap_room = 7
        receiver.transfer_nonce = 9

        receiver.send_metadata(
            np.asarray([1], dtype=np.int32),
            transfer_page_tag_ids=np.asarray([2, 2], dtype=np.int32),
            transfer_page_tags=np.asarray([3, 3], dtype=np.int32),
        )

        self.assertEqual(manager.transfer_page_tag_expected_table[7], {2: 3})
        manager.update_status.assert_not_called()

    def test_receiver_rejects_conflicting_duplicate_expected_page_tags(self):
        manager = self._decode_manager(attention_tags=True, checksum=False)
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_infos = []
        receiver.bootstrap_room = 7
        receiver.transfer_nonce = 9

        receiver.send_metadata(
            np.asarray([1], dtype=np.int32),
            transfer_page_tag_ids=np.asarray([2, 2], dtype=np.int32),
            transfer_page_tags=np.asarray([3, 4], dtype=np.int32),
        )

        manager.update_status.assert_called_once_with(7, KVPoll.Failed)
        self.assertIn(
            "Conflicting expected KV transfer tags",
            manager.record_failure.call_args.args[1],
        )

    def test_transfer_nonce_is_optional_for_old_senders(self):
        message = [
            b"7",
            b"127.0.0.1",
            b"1234",
            b"session",
            b"",
            b"",
            b"",
            b"1",
            b"",
        ]
        self.assertEqual(TransferInfo.from_zmq(message).transfer_nonce, 0)
        message.extend([b"", b"", b"9"])
        self.assertEqual(TransferInfo.from_zmq(message).transfer_nonce, 9)

    def test_fake_transfer_skips_checksum_verification(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(disaggregation_transfer_backend="mooncake")
        )
        req = SimpleNamespace(bootstrap_host="2.2.2.2")
        transfer_queue = object.__new__(DecodeTransferQueue)
        transfer_queue.scheduler = scheduler

        self.assertFalse(transfer_queue._verify_transfer_checksum(req, MagicMock()))

        prealloc_queue = object.__new__(DecodePreallocQueue)
        prealloc_queue.scheduler = scheduler
        self.assertEqual(
            prealloc_queue._prepare_transfer_page_tags(
                req=req,
                page_indices=[1],
                page_position_start=0,
                state_indices=None,
                state_types=(),
                seq_len=1,
            ),
            (None, None),
        )
        self.assertIsNone(req.kv_transfer_page_tag_manifest)


if __name__ == "__main__":
    unittest.main()
