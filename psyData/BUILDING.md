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
