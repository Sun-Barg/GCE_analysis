# =====================================================================
# phase2_fitting.py — likelihood 피팅 전용 (clean rewrite)
# =====================================================================

import os
import sys
import time
import json
import signal
import tempfile
import subprocess
import multiprocessing as mp
import psutil

NUM_CORES       = 3
FIT_TIMEOUT_MIN = 120
MEM_WARN_THRESHOLD = 90.0

FITTING_WORKER_CODE = """
import sys, os, json, signal

roman_name  = sys.argv[1]
result_file = sys.argv[2]

_like      = None
_logL_init = None
_out_xml   = None

def _save_and_exit(signum, frame):
    result = {
        'roman_name' : roman_name,
        'logL_init'  : _logL_init,
        'logL_final' : None,
        'fit_quality': -2,
        'status'     : 'timeout_sigterm',
        'xml_written': False,
    }
    if _like is not None:
        try:
            result['logL_final'] = _like.logLike.value()
        except Exception:
            pass
        if _out_xml:
            try:
                _like.logLike.writeXml(_out_xml)
                result['xml_written'] = True
            except Exception:
                pass
    try:
        with open(result_file, 'w') as f:
            json.dump(result, f)
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _save_and_exit)

try:
    import BinnedAnalysis as ba

    srcmap_file = os.path.join('Source_Maps_Perfect',
                  'GCE_17yr_srcmap_Model_{}.fits'.format(roman_name))
    xml_file    = os.path.join('XML_models',
                  'GCE_17yr_FL16Y_Model_{}.xml'.format(roman_name))
    _out_xml    = os.path.join('Fitted_XML_models_Smart',
                  'GCE_17yr_fitted_Model_{}.xml'.format(roman_name))

    os.makedirs('Fitted_XML_models_Smart', exist_ok=True)

    obs    = ba.BinnedObs(srcMaps=srcmap_file,
                          expCube='GCE_17yr_ltcube.fits',
                          binnedExpMap='GCE_17yr_expcube_large.fits',
                          irfs='P8R3_ULTRACLEANVETO_V3')
    _like  = ba.BinnedAnalysis(obs, srcModel=xml_file, optimizer='NewMinuit')
    _logL_init = _like.logLike.value()

    try:
        fit_quality = _like.fit(covar=False, tol=1e-2, verbosity=0)
    except RuntimeError as e:
        err = str(e).lower()
        if any(k in err for k in ('bounds', 'outside', 'convergence', 'matrix')):
            fit_quality = -1
        else:
            raise

    logL_final = _like.logLike.value()
    _like.logLike.writeXml(_out_xml)

    if fit_quality == 3:
        status = 'success'
    elif fit_quality == -1:
        status = 'bounds_error'
    else:
        status = 'quality_{}'.format(fit_quality)

    result = {
        'roman_name' : roman_name,
        'logL_init'  : _logL_init,
        'logL_final' : logL_final,
        'fit_quality': fit_quality,
        'status'     : status,
        'xml_written': True,
    }

except Exception as e:
    import traceback
    result = {
        'roman_name' : roman_name,
        'logL_init'  : _logL_init,
        'logL_final' : None,
        'fit_quality': None,
        'status'     : 'error: {}'.format(e),
        'traceback'  : traceback.format_exc(),
        'xml_written': False,
    }

with open(result_file, 'w') as f:
    json.dump(result, f)
"""


def fit_model(roman_name):
    out_xml = os.path.join('Fitted_XML_models_Smart',
               f'GCE_17yr_fitted_Model_{roman_name}.xml')
    if os.path.exists(out_xml):
        return {'roman_name': roman_name, 'status': 'already_fitted',
                'logL_init': None, 'logL_final': None, 'fit_quality': None,
                'xml_written': True}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as tf:
        result_file = tf.name

    timed_out = False
    try:
        proc = subprocess.Popen(
            [sys.executable, '-c', FITTING_WORKER_CODE, roman_name, result_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
        )
        try:
            proc.wait(timeout=FIT_TIMEOUT_MIN * 60)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            timed_out = True

        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            try:
                with open(result_file) as f:
                    result = json.load(f)
                if timed_out and result.get('status') not in ('timeout_sigterm',):
                    result['status']      = f'timeout_{FIT_TIMEOUT_MIN}min'
                    result['fit_quality'] = -2
            except Exception:
                result = {'roman_name': roman_name, 'logL_init': None,
                          'logL_final': None, 'fit_quality': -2 if timed_out else None,
                          'status': f'timeout_{FIT_TIMEOUT_MIN}min' if timed_out else 'no_result',
                          'xml_written': False}
        else:
            result = {'roman_name': roman_name, 'logL_init': None,
                      'logL_final': None, 'fit_quality': -2 if timed_out else None,
                      'status': f'timeout_{FIT_TIMEOUT_MIN}min' if timed_out else 'no_result',
                      'xml_written': False}

    except Exception as e:
        result = {'roman_name': roman_name, 'logL_init': None,
                  'logL_final': None, 'fit_quality': None,
                  'status': f'error: {e}', 'xml_written': False}
    finally:
        if os.path.exists(result_file):
            os.remove(result_file)

    result['roman_name'] = roman_name
    return result


def memory_monitor(threshold_pct, stop_event):
    log_path = 'fitting_memory.log'
    with open(log_path, 'w') as f:
        f.write("timestamp,used_pct,avail_gb\n")
    while not stop_event.is_set():
        vm = psutil.virtual_memory()
        ts = time.strftime('%H:%M:%S')
        with open(log_path, 'a') as f:
            f.write(f"{ts},{vm.percent:.1f},{vm.available/1024**3:.2f}\n")
        if vm.percent >= threshold_pct:
            print(f"\n  ⚠️  [메모리 경고 {ts}] {vm.percent:.1f}%", flush=True)
        time.sleep(10)


if __name__ == '__main__':
    start = time.time()
    print("=" * 60)
    print(f"Phase 2: likelihood 피팅  [코어:{NUM_CORES} / timeout:{FIT_TIMEOUT_MIN}분]")
    print("=" * 60)

    vm = psutil.virtual_memory()
    print(f"  메모리: {vm.used/1024**3:.1f} GB / {vm.total/1024**3:.1f} GB "
          f"(가용 {vm.available/1024**3:.1f} GB)\n")

    import glob
    if len(glob.glob('Source_Maps/*.fits')) == 0:
        print("  ❌ srcmap 없음. phase1_srcmap.py를 먼저 실행하세요.")
        sys.exit(1)

    models = []
    with open('NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                models.append(parts[0])

    models_to_fit = [m for m in models if os.path.exists(
        f'Source_Maps/GCE_17yr_srcmap_Model_{m}.fits')]
    print(f"  피팅 대상: {len(models_to_fit)}개\n")

    os.makedirs('Fitted_XML_models_Smart', exist_ok=True)
    os.makedirs('Error_Logs', exist_ok=True)

    results_file = 'Likelihood_Results_Final.csv'
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("Model,LogLikelihood_init,LogLikelihood_final,FitStatus,FitQuality,XML_written\n")

    stop_event   = mp.Event()
    monitor_proc = mp.Process(target=memory_monitor,
                              args=(MEM_WARN_THRESHOLD, stop_event), daemon=True)
    monitor_proc.start()

    summary = {'success': 0, 'timeout': 0, 'skip': 0, 'bounds': 0, 'error': 0}

    with mp.Pool(processes=NUM_CORES, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(fit_model, models_to_fit, chunksize=1):
            name    = result['roman_name']
            logL_i  = result.get('logL_init')
            logL_f  = result.get('logL_final')
            status  = result.get('status', 'unknown')
            quality = result.get('fit_quality')
            xml_ok  = result.get('xml_written', False)
            mem_pct = psutil.virtual_memory().percent

            li_str = f"{logL_i:.2f}" if logL_i is not None else "NaN"
            lf_str = f"{logL_f:.2f}" if logL_f is not None else "NaN"

            if status == 'success':
                print(f"  ✅ {name:10s} | -logL={lf_str} | 시스템 {mem_pct:.1f}%", flush=True)
                summary['success'] += 1
            elif status == 'already_fitted':
                print(f"  ⏩ {name:10s} (스킵)", flush=True)
                summary['skip'] += 1
            elif 'timeout' in status:
                xml_mark = "XML✅" if xml_ok else "XML❌"
                print(f"  ⏰ {name:10s} | {status} | init={li_str} best={lf_str} "
                      f"| {xml_mark} | 시스템 {mem_pct:.1f}%", flush=True)
                summary['timeout'] += 1
            elif 'bounds' in status:
                print(f"  ⚠️  {name:10s} | bounds_error | -logL={lf_str}", flush=True)
                summary['bounds'] += 1
            else:
                print(f"  ❌ {name:10s} | {status}", flush=True)
                summary['error'] += 1
                if 'traceback' in result:
                    with open(f"Error_Logs/error_fit_{name}.log", 'w') as ef:
                        ef.write(result['traceback'])

            with open(results_file, 'a') as f:
                f.write(f"{name},{li_str},{lf_str},{status},{quality},{xml_ok}\n")

    stop_event.set()
    monitor_proc.join(timeout=5)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"Phase 2 완료 — {elapsed:.1f}분")
    print(f"  성공    : {summary['success']}개")
    print(f"  timeout : {summary['timeout']}개 (best-fit logL + XML 저장됨)")
    print(f"  bounds  : {summary['bounds']}개")
    print(f"  스킵    : {summary['skip']}개")
    print(f"  실패    : {summary['error']}개")
    print(f"  결과    : {results_file}")
    print(f"{'='*60}")
