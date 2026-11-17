"""Runtime grants exclude system-installed private applications and sources."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from auriga.cortex.session.system_runtime import (runtime_library_mounts,
                                     stdlib_runtime_entries, system_runtime_mounts)
from support.host_capabilities import linux_runtime_tools, require


class SystemRuntimeTests(unittest.TestCase):
    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_grants_are_files_or_standard_library_namespaces_not_usr_roots(self):
        mounts = system_runtime_mounts()
        destinations = {str(destination): source for source, destination in mounts}
        for root in ('/usr', '/usr/local', '/usr/bin', '/usr/lib', '/bin', '/lib', '/lib64'):
            self.assertNotIn(root, destinations)
        for source, destination in mounts:
            self.assertTrue(source.is_absolute() and destination.is_absolute())
            self.assertEqual(source, source.resolve())
            self.assertNotIn('site-packages', destination.parts)
            self.assertNotIn('dist-packages', destination.parts)
            self.assertNotIn('/usr/local/src', str(destination))
            if source.is_dir():
                self.assertIn('python', str(source.parent))
        for command in ('sh', 'bash', 'env', 'find', 'sed', 'awk', 'python3'):
            self.assertIn('/usr/bin/' + command, destinations)
        self.assertTrue(any('sysconfigdata' in path for path in destinations))
        for soname in ('libstdc++.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1'):
            self.assertTrue(any(Path(path).name == soname for path in destinations), soname)

    def test_stdlib_selection_excludes_synthetic_system_installed_applications(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('site-packages/robodojo', 'dist-packages/auriga', 'private_backend', 'encodings'):
                directory = root / name
                directory.mkdir(parents=True)
                (directory / '__init__.py').write_text('SYNTHETIC_PRIVATE_IMPLEMENTATION')
            (root / 'sitecustomize.py').write_text('SYNTHETIC_HOST_SETUP')
            (root / 'os.py').write_text('SYNTHETIC_STDLIB_FIXTURE')
            entries = list(stdlib_runtime_entries(root, {'os', 'encodings'}))
            self.assertEqual({entry.relative_to(root).as_posix() for entry in entries}, {'os.py', 'encodings'})

    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_library_dependencies_preserve_loader_alias_without_parent_grants(self):
        mounts = runtime_library_mounts([Path('/usr/bin/python3')])
        self.assertTrue(mounts)
        self.assertTrue(any('ld-linux' in str(source) for source, _ in mounts))
        for source, destination in mounts:
            self.assertTrue(source.is_file())
            self.assertEqual(source, source.resolve())
            self.assertNotIn(str(destination), ('/usr/lib', '/lib', '/lib64'))

    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_ldd_paths_with_spaces_remain_complete(self):
        with tempfile.TemporaryDirectory(prefix='runtime path ') as temporary:
            root = Path(temporary)
            executable, library = root / 'program', root / 'shared library.so'
            executable.write_bytes(b'\x7fELFsynthetic')
            library.write_bytes(b'synthetic shared library')
            result = SimpleNamespace(returncode=0, stdout=' libfixture.so => ' + str(library) + ' (0x1234)\n', stderr='')
            with mock.patch('auriga.cortex.session.system_runtime.subprocess.run', return_value=result):
                mounts = runtime_library_mounts([executable])
            self.assertEqual(mounts, ((library, library),))


if __name__ == '__main__':
    unittest.main()
