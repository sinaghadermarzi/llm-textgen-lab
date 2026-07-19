import os
import unittest
from unittest.mock import patch

from llm_textgen_lab.device import detect_device


class DeviceTests(unittest.TestCase):
    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_device("tpu")

    def test_environment_can_force_cpu(self):
        with patch.dict(os.environ, {"LLM_TEXTGEN_DEVICE": "cpu"}):
            self.assertEqual(detect_device("auto"), "cpu")

