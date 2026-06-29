$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$pythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

& $pythonBin (Join-Path $rootDir "scripts/install_from_source.py") @args
exit $LASTEXITCODE
