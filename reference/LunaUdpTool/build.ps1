# Auto-detect VS installation and compile
$vsRoot = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>$null
if (-not $vsRoot) {
    $vsDirs = Get-ChildItem "${env:ProgramFiles}\Microsoft Visual Studio" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    if ($vsDirs) { $vsRoot = Join-Path $vsDirs[0].FullName "Community" }
}
if (-not $vsRoot -or -not (Test-Path "$vsRoot\VC\Auxiliary\Build\vcvarsall.bat")) {
    Write-Error "Visual Studio not found. Install VS 2022+ with C++ desktop workload."
    exit 1
}

Write-Host "VS: $vsRoot" -ForegroundColor Gray

cmd /c "call ""$vsRoot\VC\Auxiliary\Build\vcvarsall.bat"" x64 >nul && cl /EHsc demo.cpp net_cln_udp.cpp /Fe:demo.exe"