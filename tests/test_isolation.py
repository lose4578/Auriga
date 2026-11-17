import json
import os
import subprocess
import sys
import tempfile
import unittest
import socket
from pathlib import Path
from unittest.mock import Mock, patch

from auriga.runtime.programs.isolation import CREATE_RULESET, restrict
from support.host_capabilities import landlock, require


class IsolationCheck(unittest.TestCase):
    def test_ruleset_rights_follow_kernel_abi(self):
        # Linux UAPI rights: REFER since ABI 2, TRUNCATE since 3, IOCTL_DEV since 5.
        for abi, rights in [(1, 0x1fff), (2, 0x3fff), (3, 0x7fff),
                            (4, 0x7fff), (5, 0xffff), (6, 0xffff)]:
            with self.subTest(abi=abi):
                libc = Mock()
                created = []

                def syscall(number, *args):
                    if number == CREATE_RULESET:
                        if args[-1] == 1:
                            return abi
                        attrs, size, _ = args
                        created.append((attrs._obj.handled_access_fs,
                                        attrs._obj.handled_access_net, size))
                        return 42
                    return 0

                libc.syscall.side_effect = syscall
                libc.prctl.return_value = 0
                with patch('auriga.runtime.programs.isolation.ctypes.CDLL', return_value=libc), \
                        patch('auriga.runtime.programs.isolation.os.close') as close:
                    self.assertEqual(restrict([], [], tcp_ports=[] if abi >= 4 else None), abi)
                self.assertEqual(created, [(rights, 3 if abi >= 4 else 0, 16 if abi >= 4 else 8)])
                close.assert_called_once_with(42)

    def test_tcp_isolation_still_requires_abi_four(self):
        libc = Mock()
        libc.syscall.return_value = 3
        with patch('auriga.runtime.programs.isolation.ctypes.CDLL', return_value=libc):
            with self.assertRaisesRegex(RuntimeError, 'ABI >=4 required'):
                restrict([], [], tcp_ports=[])
        libc.syscall.assert_called_once_with(CREATE_RULESET, 0, 0, 1)

    @require(landlock, "the program launcher denies reads through a real Landlock ruleset")
    def test_children_cannot_read_hidden_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            allowed = root / "actor"; allowed.mkdir()
            hidden = root / "hidden"; hidden.write_text("oracle")
            spec = root / "spec.json"
            spec.write_text(json.dumps(dict(read_paths=[p for p in ["/usr", "/bin", "/lib", "/lib64"] if Path(p).exists()],
                                            write_paths=[str(allowed)], env=dict(PATH="/usr/bin:/bin"))))
            command = f'echo permitted > "{allowed}/result"; cat "{hidden}"'
            result = subprocess.run([sys.executable, "-m", "auriga.runtime.programs.isolation", str(spec), "/bin/sh", "-c", command], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Permission denied", result.stderr)
            self.assertEqual((allowed / "result").read_text(), "permitted\n")
            self.assertNotIn(b"oracle", result.stdout)
            with socket.socket() as service:
                service.bind(("127.0.0.1", 0)); service.listen()
                spec_value = json.loads(spec.read_text()); spec_value["tcp_ports"] = [443]
                spec.write_text(json.dumps(spec_value))
                program = f'import socket; socket.create_connection(("127.0.0.1", {service.getsockname()[1]}))'
                result = subprocess.run([sys.executable, "-m", "auriga.runtime.programs.isolation", str(spec), "/usr/bin/python3", "-c", program], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"Permission denied", result.stderr)


if __name__ == "__main__":
    unittest.main()
