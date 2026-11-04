"""Measure fresh REGAL stages on this computer; delete all generated analysis bundles."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE = Path('/Users/wuyang/Documents/SC-ST data').resolve()
SAMPLES = [('BRAIN', 'MouseBrain_Xenium'), ('BREAST', 'Breast_BlockA_Section1'), ('MOB', 'Puck_200127_15')]
STAGES = ('rss', 'rsp', 'rsmd')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def swap_info():
    import psutil
    try:
        return dict(psutil.swap_memory()._asdict())
    except OSError:
        return {'used': None, 'unavailable': 'macOS access restriction'}


def protect_sources(event, args):
    """Deny Python file mutations anywhere under SC-ST data, including symlinks."""
    paths = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            paths = [path]
    elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate'}:
        paths = [args[0]]
    elif event in {'os.rename', 'os.link', 'os.symlink'}:
        paths = args[:2]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved == SOURCE or SOURCE in resolved.parents:
                raise PermissionError(f'Benchmark source write prohibited: {event}: {resolved}')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_snapshot():
    snapshot = {}
    for dataset, sample in SAMPLES:
        base = SOURCE / dataset / 'ST'
        files = [base / f'{sample}.h5ad']
        if (base / sample).exists():
            files.extend(p for p in (base / sample).rglob('*') if p.is_file())
        for path in files:
            stat = path.stat()
            snapshot[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return snapshot


def configs(dataset):
    from RSS.schema import RSSPipelineConfig
    from RSP.schema import RSPPipelineConfig
    from RSMD.schema import RSMDPipelineConfig
    from Utils.dataset_registry import get_dataset
    spec = get_dataset(dataset)
    rss = RSSPipelineConfig()
    rss.preprocess.data_layer = spec.data_layer
    rss.preprocess.spot_id_field = spec.spot_id_field
    return {'rss': rss, 'rsp': RSPPipelineConfig(), 'rsmd': RSMDPipelineConfig()}


def worker(args):
    sys.addaudithook(protect_sources)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    dataset, sample = next(x for x in SAMPLES if x[1] == args.sample)
    cfg = configs(dataset)[args.stage]
    out = args.scratch / 'outputs'
    target = out / sample / f'{args.stage}_out'
    if target.exists():
        raise FileExistsError(target)
    save(args.record / f'{sample}.{args.stage}.config.json', asdict(cfg))
    start = time.perf_counter()
    print(f'START {sample} {args.stage} pid={os.getpid()}', flush=True)
    if args.stage == 'rss':
        from RSS.pipeline import run_rss_pipeline
        result = run_rss_pipeline(SOURCE / dataset / 'ST' / f'{sample}.h5ad', out, sample, cfg)
    elif args.stage == 'rsp':
        from RSP.pipeline import run_rsp_pipeline
        result = run_rsp_pipeline(out / sample / 'rss_out', out, sample, cfg)
    else:
        from RSMD.pipeline import run_rsmd_pipeline
        result = run_rsmd_pipeline(out / sample / 'rsp_out', out, sample, cfg)
    seconds = time.perf_counter() - start
    manifest = json.loads((result / 'manifest.json').read_text())
    save(args.record / f'{sample}.{args.stage}.manifest.json', manifest)
    details = {'sample': sample, 'stage': args.stage, 'pipeline_seconds': seconds, 'self_peak_resident_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if args.stage == 'rss':
        details['input_shape_after_filtering'] = json.loads((result / 'qc/qc_report.json').read_text())['summary']
        details['rss_meta'] = json.loads((result / 'rss/rss_meta.json').read_text())
    save(args.record / f'{sample}.{args.stage}.worker.json', details)
    print(f'DONE {sample} {args.stage} seconds={seconds:.3f}', flush=True)


def report(record, results):
    save(record / 'measurements.json', results)
    fields = ['sample', 'stage', 'status', 'process_wall_seconds', 'pipeline_seconds', 'peak_resident_gib', 'user_cpu_seconds', 'system_cpu_seconds', 'exit_code']
    with (record / 'measurements.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# Mac REGAL resource measurements', '', 'Fresh current-default pipelines; each stage runs in a separate process. No pre-existing analysis outputs are reused.', 'Peak resident memory: macOS per-process high-water mark from wait4/getrusage, converted from bytes to GiB. GPU memory is not a separate metric here; execution uses CPU.', 'Process wall time includes Python startup/imports, input loading, computation, QC, and output serialization. Pipeline time excludes startup and configuration imports.', 'Complete-analysis elapsed time is the sum of the three process wall times; the sequential-pipeline peak is the maximum of their peaks. Preflight, source hashing, monitoring gaps, and cleanup are excluded.', 'Configurations, versions, and output manifests are retained for interpretation; generated scientific bundles are deleted after each dataset.', '', '| Sample | Stage | Status | Wall time (min) | Peak resident memory (GiB) |', '|---|---|---|---:|---:|']
    for row in results:
        lines.append(f"| {row['sample']} | {row['stage']} | {row['status']} | {row['process_wall_seconds']/60:.2f} | {row['peak_resident_gib']:.3f} |")
    for _, sample in SAMPLES:
        rows = [r for r in results if r['sample'] == sample and r['status'] == 'completed']
        if len(rows) == 3:
            lines.append(f"\n{sample}: total {sum(r['process_wall_seconds'] for r in rows)/60:.2f} min; peak {max(r['peak_resident_gib'] for r in rows):.3f} GiB.")
    (record / 'README.md').write_text('\n'.join(lines) + '\n')


def main(args):
    import psutil
    sys.addaudithook(protect_sources)
    args.record.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix='regal_mac_resources_', dir='/private/tmp'))
    results, child = [], None
    state = {'status': 'preflight', 'scratch': str(scratch), 'record': str(args.record), 'controller_pid': os.getpid()}
    save(args.record / 'status.json', state)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    before = source_snapshot()
    save(args.record / 'source_metadata_before.json', before)
    hashes = {}
    try:
        from Utils.dataset_registry import get_dataset
        from Utils.h5ad_schema import require_rss_h5ad_schema
        for dataset, sample in SAMPLES:
            path = SOURCE / dataset / 'ST' / f'{sample}.h5ad'
            schema = require_rss_h5ad_schema(path, get_dataset(dataset), sample)
            print(schema.format_report(), flush=True)
            hashes[str(path)] = file_hash(path)
        save(args.record / 'input_sha256.json', hashes)
        packages = {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'scanpy', 'pandas', 'scipy', 'anndata', 'scikit-learn', 'pyarrow', 'psutil']}
        import torch
        save(args.record / 'environment.json', {'recorded_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'machine': platform.machine(), 'hardware': 'Apple M2 MacBook Air (Mac14,2), 8 CPU cores, 16 GB unified memory', 'ram_bytes': psutil.virtual_memory().total, 'packages': packages, 'resolved_device': 'cuda' if torch.cuda.is_available() else 'cpu', 'torch_threads': torch.get_num_threads(), 'mps_available': torch.backends.mps.is_available(), 'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}, 'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), 'git_status': subprocess.check_output(['git','status','--short'],cwd=REPO,text=True), 'initial_memory': dict(psutil.virtual_memory()._asdict()), 'initial_swap': swap_info()})
        (scratch / 'cache').mkdir()
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(scratch/'cache/numba'), MPLCONFIGDIR=str(scratch/'cache/matplotlib'), TMPDIR=str(scratch))
        for dataset, sample in SAMPLES:
            sample_dir = scratch / 'outputs' / sample
            sample_start = time.perf_counter()
            for stage in STAGES:
                state.update(status='running', sample=sample, stage=stage)
                save(args.record / 'status.json', state)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--sample', sample, '--stage', stage, '--scratch', str(scratch), '--record', str(args.record)]
                print(f'LAUNCH {sample} {stage}', flush=True)
                start = time.perf_counter()
                with (args.record / f'{sample}.{stage}.log').open('w') as log, (args.record / f'{sample}.{stage}.memory.csv').open('w') as mem:
                    writer = csv.writer(mem);writer.writerow(['elapsed_seconds','resident_bytes','system_available_bytes','swap_used_bytes','child_process_count'])
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env)
                    state['worker_pid'] = child.pid;save(args.record/'status.json',state)
                    while True:
                        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                        if pid:
                            child.returncode = os.waitstatus_to_exitcode(status)
                            break
                        try:
                            proc = psutil.Process(child.pid)
                            writer.writerow([round(time.perf_counter()-start,3),proc.memory_info().rss,psutil.virtual_memory().available,swap_info()['used'],len(proc.children(recursive=True))]);mem.flush()
                        except psutil.NoSuchProcess:
                            pass
                        time.sleep(2)
                row = {'sample':sample,'stage':stage,'status':'completed' if child.returncode == 0 else 'failed','exit_code':child.returncode,'process_wall_seconds':time.perf_counter()-start,'peak_resident_gib':usage.ru_maxrss/2**30,'peak_resident_bytes':usage.ru_maxrss,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime}
                worker_record = args.record / f'{sample}.{stage}.worker.json'
                if worker_record.exists():row.update(json.loads(worker_record.read_text()))
                results.append(row);report(args.record,results)
                print(json.dumps(row),flush=True)
                child = None
                if row['status'] != 'completed':break
                assert (sample_dir / f'{stage}_out/manifest.json').is_file()
            save(args.record / f'{sample}.cleanup.json', {'generated_bytes_before_deletion':sum(p.stat().st_size for p in sample_dir.rglob('*') if p.is_file()) if sample_dir.exists() else 0,'sample_wall_seconds_including_monitoring_and_gaps':time.perf_counter()-sample_start})
            if sample_dir.exists():shutil.rmtree(sample_dir)
            assert not sample_dir.exists()
            print(f'CLEANED {sample}',flush=True)
        state['status'] = 'completed' if len(results)==9 and all(r['status']=='completed' for r in results) else 'finished_with_failures'
    except BaseException:
        state['status']='interrupted_or_failed'
        (args.record/'controller_error.log').write_text(traceback.format_exc())
        raise
    finally:
        if child is not None and child.returncode is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if scratch.exists():shutil.rmtree(scratch)
        after = source_snapshot()
        source_check = {'metadata_unchanged':before==after,'inputs_sha256_unchanged':all(file_hash(Path(p))==h for p,h in hashes.items()),'scratch_deleted':not scratch.exists()}
        save(args.record/'source_integrity_and_cleanup.json',source_check)
        state['finished_at']=datetime.now(timezone.utc).isoformat();state.update(source_check)
        save(args.record/'status.json',state)
        report(args.record,results)
        print(json.dumps(state),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--sample');parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--scratch',type=Path);parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:main(args)
