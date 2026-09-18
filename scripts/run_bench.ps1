<#
.SYNOPSIS
  run_bench.ps1 <threads> [dataset_dir] [repeat] [-Config Release]

  Windows/PowerShell port of run_bench.sh.

  Builds jcparallelbench-static (if needed) and runs it over dataset_dir at
  a fixed thread count, saving its per-image CSV output to
  bench_results\threads_<N>.csv. See src\jcparallelbench.c for what's
  measured (encode-only wall-clock time, mean+min over `repeat` encodes per
  image).

  Typically invoked via bench_2vs4.ps1 rather than directly.

.EXAMPLE
  .\scripts\run_bench.ps1 4
  .\scripts\run_bench.ps1 2 C:\datasets\photos 100
  .\scripts\run_bench.ps1 4 testimages 50 -Config Debug
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [int]$Threads,

  [Parameter(Position = 1)]
  [string]$DatasetDir = "testimages",

  [Parameter(Position = 2)]
  [int]$Repeat = 50,

  # Visual Studio (and other multi-config) generators build into
  # build\<Config>\; single-config generators (Ninja) ignore this.
  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

if ($Threads -lt 1)  { throw "threads must be >= 1 (got $Threads)" }
if ($Repeat  -lt 1)  { throw "repeat must be >= 1 (got $Repeat)" }

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$BuildDir   = Join-Path $RepoRoot "build"
$ResultsDir = Join-Path $RepoRoot "bench_results"

# Resolve a relative dataset_dir against the repo root (not the caller's
# cwd), so `.\scripts\run_bench.ps1 4` works the same from anywhere.
if (-not [System.IO.Path]::IsPathRooted($DatasetDir)) {
  $DatasetDir = Join-Path $RepoRoot $DatasetDir
}
if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) {
  throw "dataset dir not found: $DatasetDir"
}

if (-not (Test-Path -LiteralPath $BuildDir -PathType Container)) {
  throw @"
$BuildDir not found -- configure it first (see Steps_to_build_multi_core.txt):

  `$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
  New-Item -ItemType Directory -Force "$BuildDir" | Out-Null
  Set-Location "$BuildDir"
  & `$cmake -G "Visual Studio 18 2026" -A x64 -DWITH_SIMD=0 -DWITH_OPENMP=1 -DWITH_JPEG7=0 ..
"@
}

# --- locate cmake ------------------------------------------------------------
# Prefer one on PATH; otherwise fall back to the copy bundled inside a
# Visual Studio install (standalone CMake often isn't on PATH on Windows).
function Resolve-CMake {
  $onPath = Get-Command cmake -ErrorAction SilentlyContinue
  if ($onPath) { return $onPath.Source }

  $candidates = @()
  foreach ($pf in @(${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {
    if (-not $pf) { continue }
    $candidates += Get-ChildItem -Path (Join-Path $pf "Microsoft Visual Studio") `
      -Recurse -Filter "cmake.exe" -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "CommonExtensions\\Microsoft\\CMake" } |
      Select-Object -ExpandProperty FullName
    $candidates += (Join-Path $pf "CMake\bin\cmake.exe")
  }
  foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) { return $c }
  }
  throw "cmake.exe not found on PATH or in any Visual Studio install -- add it to PATH."
}
$CMake = Resolve-CMake

# --- build -----------------------------------------------------------------
# `cmake --build --parallel N` translates to the right thing for whatever
# generator configured this build dir (MSBuild's /m, Ninja's -j, ...) --
# passing native flags after `--` is fragile under PowerShell's arg parsing.
$nproc = if ($env:NUMBER_OF_PROCESSORS) { [int]$env:NUMBER_OF_PROCESSORS } else { 4 }

& $CMake --build $BuildDir --config $Config --target jcparallelbench-static --parallel $nproc
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed (exit $LASTEXITCODE)" }

# --- locate the freshly built exe ----------------------------------------
$exeCandidates = @(
  (Join-Path $BuildDir "$Config\jcparallelbench-static.exe"),  # multi-config (VS)
  (Join-Path $BuildDir "jcparallelbench-static.exe")            # single-config (Ninja)
)
$Exe = $exeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Exe) {
  throw "jcparallelbench-static.exe not found after build; looked in:`n  $($exeCandidates -join "`n  ")"
}

# --- run -----------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$OutCsv = Join-Path $ResultsDir "threads_$Threads.csv"

# The exe writes the CSV to stdout and per-image "skip ..." notes to stderr.
# In Windows PowerShell, any native-command stderr line becomes an error
# record that $ErrorActionPreference='Stop' would turn fatal -- so drop to
# 'Continue' for the call, split the merged 2>&1 stream by record type
# (stderr -> passed straight through to the console, stdout -> captured),
# and gate purely on the real exit code.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$csvLines = & $Exe $DatasetDir --threads $Threads --repeat $Repeat 2>&1 |
  ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
      [Console]::Error.WriteLine([string]$_)
    } else {
      $_
    }
  }
$exeExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
if ($exeExit -ne 0) { throw "jcparallelbench-static failed (exit $exeExit)" }

# Write the file ourselves as UTF-8 *without* a BOM -- Windows PowerShell's
# Tee-Object/Set-Content would emit UTF-16 or a BOM-prefixed file, either of
# which trips up merge_bench.py's csv reader.
$csvLines | ForEach-Object { Write-Host $_ }
[System.IO.File]::WriteAllLines(
  $OutCsv, [string[]]$csvLines, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Wrote $OutCsv"
