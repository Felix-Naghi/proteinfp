@echo off
REM Create .vscode\launch.json with Module 14 configs
REM Save this file as: C:\Users\adria\Documents\proteinFP\create_launch.bat
REM Then double-click it to run

cd /d C:\Users\adria\Documents\proteinFP

REM Create .vscode folder if it doesn't exist
if not exist ".vscode" mkdir ".vscode"

REM Create launch.json with the Module 14 configs
(
echo {
echo     "version": "0.2.0",
echo     "configurations": [
echo         {
echo             "name": "Python: Current File",
echo             "type": "python",
echo             "request": "launch",
echo             "program": "${file}",
echo             "console": "integratedTerminal"
echo         },
echo         {
echo             "name": "Module 14: MD (prompt for UniProt)",
echo             "type": "debugpy",
echo             "request": "launch",
echo             "program": "${workspaceFolder}/pipeline/14_molecular_dynamics.py",
echo             "console": "integratedTerminal",
echo             "cwd": "${workspaceFolder}",
echo             "args": [
echo                 "--uniprot", "${input:uniprotID}",
echo                 "--ns",      "${input:nsLength}",
echo                 "--implicit"
echo             ],
echo             "env": {
echo                 "PYTHONPATH": "${workspaceFolder}"
echo             },
echo             "justMyCode": false
echo         },
echo         {
echo             "name": "Module 14: MD P04637 (GPU, explicit)",
echo             "type": "debugpy",
echo             "request": "launch",
echo             "program": "${workspaceFolder}/pipeline/14_molecular_dynamics.py",
echo             "console": "integratedTerminal",
echo             "cwd": "${workspaceFolder}",
echo             "args": [
echo                 "--uniprot", "P04637",
echo                 "--ns",      "2.0",
echo                 "--gpu"
echo             ],
echo             "env": {
echo                 "PYTHONPATH": "${workspaceFolder}"
echo             },
echo             "justMyCode": false
echo         }
echo     ],
echo     "inputs": [
echo         {
echo             "id": "uniprotID",
echo             "type": "promptString",
echo             "description": "UniProt accession (e.g. P04637)",
echo             "default": "P04637"
echo         },
echo         {
echo             "id": "nsLength",
echo             "type": "promptString",
echo             "description": "Production length in ns",
echo             "default": "1.0"
echo         }
echo     ]
echo }
) > .vscode\launch.json

echo.
echo.
echo ===================================================================
echo Success! launch.json created at:
echo C:\Users\adria\Documents\proteinFP\.vscode\launch.json
echo ===================================================================
echo.
echo Next steps:
echo 1. Close VS Code completely
echo 2. Reopen the proteinFP.code-workspace
echo 3. Press Ctrl+Shift+D to open Run and Debug
echo 4. Click the dropdown - you should see:
echo    - Python: Current File
echo    - Module 14: MD (prompt for UniProt)
echo    - Module 14: MD P04637 (GPU, explicit)
echo.
echo Press any key to close this window...
pause