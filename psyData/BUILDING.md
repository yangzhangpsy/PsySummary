# Building PsySummary

Use the target platform's Python environment with PsySummary's dependencies and
PyInstaller 6 or newer installed. Activate that environment before running:

- macOS: `./makeRunMacNew` (DropDMG CLI and the `psySummary` configuration required).
- Windows: `makeRunWin.bat` or `makeRunWinYang.bat` (x64 Python, Inno Setup 6 in its
  default installation folder, PowerShell and Windows `tar` required).
- Debian/Ubuntu: `./makeRunLinux` (`dpkg` and `dpkg-deb` required).

Like PsyBuilder, these scripts call PyInstaller directly using the local `.spec`
files, then run DropDMG, Inno Setup or `dpkg-deb`. Temporary build output stays in
`build`/`dist`; timestamped release files go into the parent directory of `psyData`.
No additional Python build framework, automatic package installation or `sudo`
is involved. Real-platform installer testing remains a separate release step.

## Optional macOS signing

Without a signing identity, the default build uses PyInstaller's ad-hoc signing.
To use a certificate, specify its full name from the current keychain:

```sh
PSYSUMMARY_CODESIGN_IDENTITY='Developer ID Application: Your Name (TEAMID)' ./makeRunMacNew
```

Like PsyBuilder, signed builds use `entitlements.plist`. The script checks the
identity before building and verifies the resulting app signature before creating
the DMG. Signing or verification failure stops the release. This does not perform
Apple notarization; notarization and Gatekeeper testing remain separate steps.

## PsySummary background workers

The source and frozen entry point dispatches `--psysummary-fit-worker` and
`--psysummary-write-worker` before importing GUI widgets. Keep this dispatch and
the worker guard in `app/lib/__init__.py` when changing the launcher. Fits and
diagnostic-curve preparation, data exports, and result saves use owned child
processes; loading and analysis preparation use background threads.

Run the regression tests from the repository root with the application's Python:

```sh
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -p 'test_*.py'
```

These tests exercise source-mode child processes and Qt lifecycle handling. They
do not replace testing fitting, export, cancellation, and shutdown in the frozen
application on each supported operating system.
