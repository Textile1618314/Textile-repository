#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "02_analysis"
RUN = ANALYSIS / "_run"
RUN.mkdir(parents=True, exist_ok=True)

REQUIRED = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn"]
OPTIONAL = ["pypdf"]

CPSC_JSON = ROOT / "01_data" / "cpsc_recalls_all.json"

STEPS = [
    ("09", "09_dataset_hardening", True),
    ("10", "10_hazard_channel_regime", True),
    ("11", "11_violation_vs_incident", True),
    ("12", "12_boundary_archetypes", True),
    ("13", "13_firm_concentration_repeat", True),
    ("14", "14_bayes_hierarchical_rate", True),
    ("15", "15_bayes_changepoint", True),
    ("16", "16_ml_remedy_and_scale", True),
    ("17", "17_text_topics_recall_narratives", True),
]
LEGACY = [("02", "02_annual_trends"), ("08", "08_recall_rate_per_import")]


def log(msg, fh=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def ensure_packages(fh):
    missing = []
    import importlib
    for spec in REQUIRED + OPTIONAL:
        mod = {"scikit-learn": "sklearn"}.get(spec, spec)
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(spec)
    if not missing:
        log("dependencies: all present", fh)
        return True
    log(f"dependencies: installing {missing}", fh)
    for flags in (["--user"], ["--break-system-packages"], []):
        cmd = [sys.executable, "-m", "pip", "install", "-q", *flags, *missing]
        p = subprocess.run(cmd, capture_output=True, text=True)
        fh.write(p.stdout + p.stderr)
        if p.returncode == 0:
            log(f"dependencies: installed with {flags or ['(default)']}", fh)
            return True
    log("dependencies: pip FAILED - install manually: "
        f"pip install {' '.join(missing)}", fh)
    return False


def ensure_cpsc_json(fh):
    if CPSC_JSON.exists() and CPSC_JSON.stat().st_size > 1_000_000:
        log(f"cpsc json: present ({CPSC_JSON.stat().st_size/1e6:.0f} MB)", fh)
        return True
    log("cpsc json: downloading (large file, this can take several minutes)", fh)
    sys.path.insert(0, str(ROOT))
    from download_data import download
    if download(CPSC_JSON):
        log(f"cpsc json: saved {CPSC_JSON.stat().st_size/1e6:.0f} MB", fh)
        return True
    log("cpsc json: download failed. Analyses will run on the title-only "
        "dataset; text results will be thinner.", fh)
    return False


def run_script(path: Path, quick: bool, step: str, kind: str):
    logfile = RUN / f"{step}_{kind}.log"
    cmd = [sys.executable, str(path.name)] + (["--quick"] if quick else [])
    t0 = time.time()
    p = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True)
    dt = time.time() - t0
    logfile.write_text(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{p.stdout}"
                       f"\n--- stderr ---\n{p.stderr}\n")
    return {"step": step, "kind": kind, "ok": p.returncode == 0,
            "seconds": round(dt, 1), "returncode": p.returncode,
            "stdout_tail": p.stdout.strip().splitlines()[-12:],
            "stderr_tail": p.stderr.strip().splitlines()[-12:],
            "log": str(logfile.relative_to(ANALYSIS))}


def collect_key_numbers():
    out = {}
    for _, folder, _ in STEPS:
        res = ANALYSIS / folder / "results"
        if not res.exists():
            continue
        picks = {}
        for f in sorted(res.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (int, float, str, bool)) or (
                            isinstance(v, dict) and len(v) <= 12):
                        picks[f"{f.stem}.{k}"] = v
        out[folder] = picks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke settings everywhere")
    ap.add_argument("--only", nargs="*", default=None,
                    help="step ids to run, e.g. --only 14 15")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step ids to skip, e.g. --skip 16")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="also re-run analysis 08")
    args = ap.parse_args()

    fh = open(RUN / "run_log.txt", "a", encoding="utf-8")
    t_start = time.time()
    log("=" * 68, fh)
    log(f"RUN START  {datetime.now(timezone.utc).isoformat()}  "
        f"quick={args.quick}", fh)
    log(f"python {sys.version.split()[0]}  at {sys.executable}", fh)

    if not args.no_install:
        ensure_packages(fh)
    have_json = False if args.no_download else ensure_cpsc_json(fh)

    steps = [s for s in STEPS if (args.only is None or s[0] in args.only)
             and s[0] not in args.skip]
    if args.legacy:
        steps += [(sid, folder, False) for sid, folder in LEGACY
                  if args.only is None or sid in args.only]

    results = []
    for sid, folder, has_fig in steps:
        d = ANALYSIS / folder
        if not d.exists():
            log(f"{sid}: folder missing, skipped", fh)
            continue
        if not args.figures_only:
            a = d / "analysis.py"
            if a.exists():
                log(f"{sid} analysis: running ...", fh)
                r = run_script(a, args.quick, sid, "analysis")
                results.append(r)
                log(f"{sid} analysis: {'OK' if r['ok'] else 'FAILED'} "
                    f"({r['seconds']}s)", fh)
                if not r["ok"]:
                    for line in r["stderr_tail"]:
                        log(f"    | {line}", fh)
        if has_fig and not args.analysis_only:
            f = d / "figure.py"
            if f.exists():
                log(f"{sid} figure: drawing ...", fh)
                r = run_script(f, args.quick, sid, "figure")
                results.append(r)
                log(f"{sid} figure: {'OK' if r['ok'] else 'FAILED'} "
                    f"({r['seconds']}s)", fh)
                if not r["ok"]:
                    for line in r["stderr_tail"]:
                        log(f"    | {line}", fh)

    total = time.time() - t_start
    ok = sum(r["ok"] for r in results)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "quick": args.quick,
        "raw_cpsc_json_available": have_json,
        "total_seconds": round(total, 1),
        "steps_ok": ok, "steps_total": len(results),
        "steps": results,
        "outputs": {
            folder: sorted(p.name for p in (ANALYSIS / folder).glob("figure_*.*"))
            for _, folder, _ in STEPS if (ANALYSIS / folder).exists()
        },
        "key_numbers": collect_key_numbers(),
    }
    (RUN / "run_report.json").write_text(json.dumps(report, indent=2,
                                                    default=str),
                                         encoding="utf-8")

    md = [f"# Run report", "",
          f"- started: {report['started_utc']}",
          f"- mode: {'QUICK (smoke settings)' if args.quick else 'PRODUCTION'}",
          f"- raw CPSC json available: {have_json}",
          f"- total time: {total/60:.1f} min",
          f"- steps: {ok}/{len(results)} OK", "",
          "| step | kind | status | seconds |", "|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['step']} | {r['kind']} | "
                  f"{'OK' if r['ok'] else '**FAILED**'} | {r['seconds']} |")
    fails = [r for r in results if not r["ok"]]
    if fails:
        md += ["", "## Failures", ""]
        for r in fails:
            md.append(f"### {r['step']} {r['kind']}  (see `_run/{r['log']}`)")
            md += ["```"] + r["stderr_tail"] + ["```"]
    md += ["", "## Figures produced", ""]
    for folder, files in report["outputs"].items():
        if files:
            md.append(f"- **{folder}**: {', '.join(files)}")
    (RUN / "run_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    log(f"RUN END  {ok}/{len(results)} OK  in {total/60:.1f} min", fh)
    log(f"report: {RUN/'run_report.md'}", fh)
    fh.close()
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
