<#
.SYNOPSIS
  bench_2vs4.ps1 [dataset_dir] [repeat] [-Config Release]

  Windows/PowerShell port of bench_2vs4.sh.

  Runs jcparallelbench-static over dataset_dir once at --threads 2 and once
  at --threads 4, then merges the two per-image CSVs into
  bench_results\dataset.csv -- the per-image (2-core time, 4-core time)
  dataset for the complexity-labeling / classifier-training step.

  dataset_dir defaults to the repo's testimages\ (a handful of small images,
  good for validating this pipeline end-to-end, not for training on).
  repeat defaults to 50 encodes/image, since these images are small enough
  that a single encode is sub-millisecond / noisy.

.EXAMPLE
  .\scripts\bench_2vs4.ps1
  .\scripts\bench_2vs4.ps1 C:\datasets\photos 100
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$DatasetDir = "testimages",

  [Parameter(Position = 1)]
  [int]$Repeat = 50,

  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$ResultsDir = Join-Path $RepoRoot "bench_results"

# --- locate a Python interpreter (for merge_bench.py) ---------------------
# Returns @{ Exe = <path>; PreArgs = @(...) } -- PreArgs carries "-3" for the
# `py` launcher so it selects Python 3.
function Resolve-Python {
  foreach ($name in @("python", "python3", "py")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
      $pre = if ($name -eq "py") { @("-3") } else { @() }
      return @{ Exe = $cmd.Source; PreArgs = $pre }
    }
  }
  throw "No Python interpreter found on PATH (need python/python3/py for merge_bench.py)."
}
$Python = Resolve-Python

& (Join-Path $ScriptDir "run_bench.ps1") 2 $DatasetDir $Repeat -Config $Config
& (Join-Path $ScriptDir "run_bench.ps1") 4 $DatasetDir $Repeat -Config $Config

$mergeArgs = @($Python.PreArgs) + @(
  (Join-Path $ScriptDir "merge_bench.py"),
  (Join-Path $ResultsDir "threads_2.csv"),
  (Join-Path $ResultsDir "threads_4.csv"),
  "-o", (Join-Path $ResultsDir "dataset.csv")
)
# merge_bench.py prints its warnings and summary to stderr; same
# stderr-becomes-fatal-under-'Stop' dance as in run_bench.ps1.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python.Exe @mergeArgs 2>&1 | ForEach-Object {
  if ($_ -is [System.Management.Automation.ErrorRecord]) {
    [Console]::Error.WriteLine([string]$_)
  } else {
    Write-Host $_
  }
}
$mergeExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
if ($mergeExit -ne 0) { throw "merge_bench.py failed (exit $mergeExit)" }

Write-Host ""
Write-Host "Per-image 2-core-vs-4-core dataset: $(Join-Path $ResultsDir 'dataset.csv')"
