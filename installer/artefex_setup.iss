; Artefex Windows Installer - Inno Setup Script
; Builds a proper Windows installer with Start Menu, Desktop shortcut,
; and uninstaller.
;
; Prerequisites:
;   1. Run build_windows.py first to create dist/Artefex/
;   2. Install Inno Setup from https://jrsoftware.org/isinfo.php
;   3. Open this file in Inno Setup Compiler and click Build
;
; Or from command line:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\artefex_setup.iss

#define MyAppName "Artefex"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Artefex Contributors"
#define MyAppURL "https://github.com/turnert2005/artefex"
#define MyAppExeName "Artefex.exe"

[Setup]
AppId={{B8E2F4A1-3C7D-4E9F-A5B2-1D8F6E3C9A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist
OutputBaseFilename=Artefex-{#MyAppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Include everything from the PyInstaller dist/Artefex/ folder
Source: "..\dist\Artefex\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Artefex"; Flags: nowait postinstall skipifsilent
