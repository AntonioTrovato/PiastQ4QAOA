"""Synthetic unit tests for piastq_execution.backend_config.

No real backend/network calls -- load_backend_config() only reads a YAML
file (or falls back to defaults) and never touches AQTProvider itself;
get_backend() is exercised indirectly by construction, not called here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.backend_config import load_backend_config


class TestLoadBackendConfig(unittest.TestCase):
    def setUp(self):
        self._env_backup = {
            "PIASTQ_API_TOKEN": os.environ.pop("PIASTQ_API_TOKEN", None),
            "PIASTQ_BACKEND_NAME": os.environ.pop("PIASTQ_BACKEND_NAME", None),
        }

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _write_config(self, tmp_path, contents):
        with open(tmp_path, "w") as f:
            f.write(contents)

    def test_reads_from_yaml_file(self):
        path = "/tmp/piastq_test_backend_config.yaml"
        self._write_config(path, "api_token: MY_TOKEN\nbackend_name: my_backend\n")
        try:
            api_token, backend_name = load_backend_config(config_path=path)
            self.assertEqual(api_token, "MY_TOKEN")
            self.assertEqual(backend_name, "my_backend")
        finally:
            os.remove(path)

    def test_missing_file_falls_back_to_defaults(self):
        api_token, backend_name = load_backend_config(config_path="/tmp/piastq_does_not_exist.yaml")
        self.assertEqual(api_token, "ACCESS_TOKEN")
        self.assertEqual(backend_name, "offline_simulator_no_noise")

    def test_env_vars_take_precedence_over_yaml(self):
        path = "/tmp/piastq_test_backend_config_env.yaml"
        self._write_config(path, "api_token: FILE_TOKEN\nbackend_name: file_backend\n")
        os.environ["PIASTQ_API_TOKEN"] = "ENV_TOKEN"
        os.environ["PIASTQ_BACKEND_NAME"] = "env_backend"
        try:
            api_token, backend_name = load_backend_config(config_path=path)
            self.assertEqual(api_token, "ENV_TOKEN")
            self.assertEqual(backend_name, "env_backend")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
