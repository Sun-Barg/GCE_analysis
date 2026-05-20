#!/usr/bin/env python3
"""run_one_model_wrapper.py — pipeline phase 분리 wrapper.

launcher는 'python run_one_model_wrapper.py MODEL' 형식으로 이 wrapper를 호출.
wrapper는 동일 인터페이스로 run_one_model.py를 두 subprocess로 분리 호출:

  Phase 1 (prepare): XML build + gtsrcmaps × 2 + gtmodel × 12
                     → fermitools state 남기고 종료
  Phase 2 (mcmc)   : Likelihood + emcee × 14 bin + save .dat
                     → fresh process (Job 6 패턴)

Root cause (2026-05-14 확정):
  Job 4 (IX, srcmap skip + Pool 제거):  67.4 min 완주
  Job 5 (X,  srcmap build + Pool 제거): MCMC 진입 직전 SIGKILL
  Job 6 (X,  srcmap skip + Pool 제거):  68.3 min 완주
  → fermitools 실행 후 같은 process에서 MCMC = SIGKILL trigger
  → 12yr lesson #10 'Fermi tools fork 이슈' 확정

interface:
  python run_one_model_wrapper.py MODEL

종료 코드:
  prepare 실패: 그 rc 반환, mcmc 안 실행
  mcmc 실패  : 그 rc 반환
  모두 성공 : 0
환경변수 (예: DIAG_SAVE_CHAIN)는 두 subprocess에 그대로 전달.
"""
import os
import sys
import time
import subprocess


def _run_phase(runner, model, phase):
    env = os.environ.copy()
    env['RUN_PHASE'] = phase
    print(f'\n---- phase: {phase} ----', flush=True)
    t = time.time()
    rc = subprocess.call(
        [sys.executable, '-u', runner, model],
        env=env,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    dt = (time.time() - t) / 60
    if rc != 0:
        print(f'---- phase {phase} FAIL  rc={rc}  elapsed={dt:.1f} min ----',
              flush=True)
    else:
        print(f'---- phase {phase} done  elapsed={dt:.1f} min ----', flush=True)
    return rc


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_one_model_wrapper.py MODEL', file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1].strip()

    runner = './run_one_model.py'
    if not os.path.exists(runner):
        runner = 'run_one_model.py'
    if not os.path.exists(runner):
        print(f'[FATAL] run_one_model.py not found in cwd ({os.getcwd()})',
              file=sys.stderr)
        sys.exit(2)

    print(f'==== wrapper start  model={model}  pid={os.getpid()}  '
          f'runner={runner} ====', flush=True)
    t0 = time.time()

    rc = _run_phase(runner, model, 'prepare')
    if rc != 0:
        print(f'==== wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    rc = _run_phase(runner, model, 'mcmc')
    if rc != 0:
        print(f'==== wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    print(f'==== wrapper done  model={model}  '
          f'total={(time.time()-t0)/60:.1f} min ====', flush=True)


if __name__ == '__main__':
    main()
