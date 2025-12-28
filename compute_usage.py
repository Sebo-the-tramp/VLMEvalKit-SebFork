#!/usr/bin/env python3
import os, time, subprocess, pathlib, re, math
from threading import Event, Thread
from datetime import datetime, timedelta

# config
# DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'

# this should run on 8 A100-40GBs
# GPUS = list(range(8))

# This should run on 2 5090s
GPUS = list(range(2))                    # physical GPU indices to use
DATASET = '/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output'
RUN_NAME = 'run_08_general'

GPUS = GPUS[1:]
# jobs: model, g = number of GPUs
# optional: uv = ['pkg==ver', ...], extra = ['--flag','value', ...]

# Initial check for answers and questions

# call another python program
# check_program = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/utils/check_questions_have_answers.py"
# question_path = os.path.join(DATASET, RUN_NAME, f"test_{RUN_NAME}.json")
# answer_path = os.path.join(DATASET, RUN_NAME, f"val_answer_{RUN_NAME}.json")

# # Run the check program and get the result
# result = subprocess.run(['python', check_program, "--question-path", question_path, "--answer-path", answer_path], capture_output=True, text=True)

# # If the check fails (non-zero return code), stop execution
# if result.returncode != 0:
#     print(f"Check failed: {result.stderr}")
#     print("Stopping execution.")
#     exit(1)

# print("Check passed. Continuing with job execution...")

JOBS_ALL = [
    # {'model':"InternVL2-8B-MPO-CoT"},
    {'model': "InternVL2.5-78B"},
    {'model': "Qwen2-VL-72B"},
    {'model': "InternVL2.5-38B"},
    {'model': "InternVL2.5-26B"},
    {'model': "Ovis1.6-Gemma2-27B"},
    {'model': "InternVL2-Llama3-76B"},
    {'model': "InternVL2-40B"},
    {'model': "Ovis1.6-Gemma2-9B"},
    {'model': "InternVL2.5-8B"},
    {'model': "LLaVA-OneVision-72B"},
    {'model': "Qwen2-VL-7B"},
    {'model': "InternVL2-26B"},
    {'model': "MiniCPM-V-2.6"},
    {'model': "InternVL2.5-4B"},
    {'model': "InternVL2-8B-MPO-CoT"},
    {'model': "InternVL2-8B-MPO"},
    {'model': "InternVL2-8B"},
    {'model': "Ovis1.5-Gemma2-9B"},
    {'model': "Aria"},
    {'model': "Llama-3.2-90B-Vision-Instruct"},
    {'model': "MMAlaya2"},
    {'model': "Ovis1.5-Llama3-8B"},
    {'model': "InternVL-Chat-V1.5"},
    {'model': "Ovis1.6-Llama3.2-3B"},
    {'model': "InternLM-XComposer2.5"},
    {'model': "Pixtral-12B"},
    {'model': "InternVL2-4B"},
    {'model': "LLaVA-OneVision-7B"},
    {'model': "InternVL2.5-2B"},
    {'model': "Aquila-VL-2B"},
    {'model': "MiniCPM-Llama3-V2.5"},
    {'model': "VILA1.5-40B"},
    {'model': "Llama-3.2-11B-Vision-Instruct"},
    {'model': "Qwen2-VL-2B"},
    {'model': "Idefics3-8B-Llama3"},
    {'model': "Mini-InternVL-Chat-4B-V1.5"},
    {'model': "XinYuan-VL-2B-Instruct"},
    {'model': "Eagle-X5-34B-Chat"},
    {'model': "LLaVA-Next-Yi-34B"},
    {'model': "LLaVA-CoT"},
    {'model': "InternVL2.5-1B"},
    {'model': "InternVL2-2B"},
    {'model': "H2OVL-2B"},
    {'model': "MiniMonkey"},
    {'model': "Phi-3.5-Vision"},
    {'model': "Vintern-3B-beta"},
    {'model': "LLaVA-Next-Qwen-32B"},
    {'model': "XGen-MM-Instruct-Interleave-v1.5"},
    {'model': "Eagle-X5-13B"},
    {'model': "XGen-MM-Instruct-DPO-v1.5"},
    {'model': "Mini-InternVL-Chat-2B-V1.5"},
    {'model': "LLaVA-Next-Llama3"},
    {'model': "Yes"},
    {'model': "Mantis-8B-Idefics2"},
    {'model': "LLaVA-Next-Interleave-7B-DPO"},
    {'model': "Monkey-Chat"},
    {'model': "InternVL2-1B"},
    {'model': "LLaVA-Next-Vicuna-13B"},
    {'model': "LLaVA-Next-Interleave-7B"},
    {'model': "Eagle-X5-7B"},
    {'model': "SmolVLM-Synthetic"},
    {'model': "SmolVLM-Instruct"},
    {'model': "DeepSeek-VL-7B"},
    {'model': "LLaVA-Next-Mistral-7B"},
    {'model': "LLaVA-LLaMA-3-8B"},
    {'model': "Qwen-VL-Chat"},
    {'model': "LLaVA-Next-Vicuna-7B"},
    {'model': "Slime-8B"},
    {'model': "Llama-3-VILA1.5-8B"},
    {'model': "Monkey"},
    {'model': "H2OVL-800M"},
    {'model': "Mantis-8B-siglip-llama3"},
    {'model': "LLaVA-OneVision-0.5B"},
    {'model': "Vintern-1B-v2"},
    {'model': "VILA1.5-3B"},
    {'model': "Parrot-7B"},
    {'model': "Janus-1.3B"},
    {'model': "DeepSeek-VL-1.3B"},
    {'model': "Emu2_chat"},
    {'model': "Mantis-8B-clip-llama3"},
    {'model': "Mantis-8B-Fuyu"},
    {'model': "IDEFICS-80B-Instruct"},
    {'model': "Qwen-VL"},
    {'model': "IDEFICS-9B-Instruct"},
    {'model': "Chameleon-30B"},
    {'model': "OpenFlamingo v2"},
    {'model': "Chameleon-7B"},
    {'model': "Kosmos2"},
    {'model': "LLaVA-Video-7B-Qwen2"},
    {'model': "LLaVA-Video-72B-Qwen2"},
    {'model': "Video-LLaVA"},
    {'model': "VideoChat2-HD"},
    {'model': "Chat-UniVi-7B"},
    {'model': "Chat-UniVi-7B-v1.5"},
    {'model': "LLaMA-VID-7B"},
    {'model': "Video-ChatGPT"},
    {'model': "PLLaVA-7B"},
    {'model': "PLLaVA-13B"},
    {'model': "PLLaVA-34B"},
    {'model': "SAIL-VL-2B"},
    {'model': "VARCO-VISION-14B"},
    {'model': "Valley-Eagle"},
    {'model': "InternVL2.5-78B-MPO"},
    {'model': "InternVL2.5-38B-MPO"},
    {'model': "InternVL2.5-26B-MPO"},
    {'model': "InternVL2.5-8B-MPO"},
    {'model': "InternVL2.5-4B-MPO"},
    {'model': "InternVL2.5-2B-MPO"},
    {'model': "InternVL2.5-1B-MPO"},
    {'model': "QVQ-72B-Preview"},
    {'model': "VITA1.5"},
    {'model': "VITA"},
    {'model': "DeepSeek-VL2-Tiny"},
    {'model': "DeepSeek-VL2-Small"},
    {'model': "DeepSeek-VL2"},
    {'model': "Ross-Qwen2-7B"},
    {'model': "MiniCPM-o-2.6"},
    {'model': "Emu3_chat"},
    {'model': "SmolVLM-256M"},
    {'model': "SmolVLM-500M"},
    {'model': "SenseNova"},
    {'model': "Taiyi"},
    {'model': "Step-1.5V"},
    {'model': "TeleMM"},
    {'model': "Gemini-1.5-Pro-002"},
    {'model': "GPT-4o-20241120"},
    {'model': "GPT-4o-20240806"},
    {'model': "BailingMM-Lite-1203"},
    {'model': "GLM-4v-Plus"},
    {'model': "Claude 3.5 Sonnet 20241022"},
    {'model': "GPT-4o-20240513"},
    {'model': "abab7-preview"},
    {'model': "Gemini-1.5-Flash-002"},
    {'model': "JT-VL-Chat-V3.0"},
    {'model': "Claude 3.5 Sonnet 20240620"},
    {'model': "BlueLM-V-3B"},
    {'model': "Qwen-VL-Max-0809"},
    {'model': "Qwen-VL-Plus-0809"},
    {'model': "CongRong"},
    {'model': "Step-1.5V-mini"},
    {'model': "Gemini-1.5-Pro"},
    {'model': "GPT-4o-mini-20240718"},
    {'model': "Yi-Vision"},
    {'model': "GPT-4v-20240409"},
    {'model': "Gemini-1.5-Flash"},
    {'model': "GLM-4v"},
    {'model': "Qwen-VL-Max"},
    {'model': "GPT-4v-20231106"},
    {'model': "Gemini-1.0-Pro"},
    {'model': "Claude3-Opus"},
    {'model': "Claude3-Sonnet"},
    {'model': "Qwen-VL-Plus"},
    {'model': "Claude3-Haiku"},
    {'model': "abab6.5s"},
    {'model': "Hunyuan-Vision"},
    {'model': "HunYuan-Standard-Vision"},
    {'model': "BailingMM-Pro-0120"},
    {'model': "Doubao-1.5-Pro"},
    {'model': "GLM-4v-Plus-20250111"}
]

def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+','_', name)


def query_gpu_memory(gpu_ids):
    """Return dict of {gpu_id: memory_used_mib}."""
    if not gpu_ids:
        return {}
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits',
        '-i', ','.join(str(i) for i in gpu_ids)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if res.returncode != 0:
        return None
    usage = {}
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',') if p.strip()]
        if len(parts) != 2:
            continue
        try:
            idx, mem = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        usage[idx] = mem
    return usage


def start_vram_monitor(job_name, run_name, gpu_ids, log_dir, interval=1.0):
    """Start a monitoring thread that samples VRAM usage for specific GPUs."""
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fname = f'{timestamp}_{safe(run_name or "run")}_{safe(job_name)}.csv'
    log_path = log_dir / fname
    stop_event = Event()
    stats = {'max': {gid: 0 for gid in gpu_ids}, 'error': None}

    def monitor_loop():
        header = 'timestamp,' + ','.join(f'gpu{gid}_mb' for gid in gpu_ids)
        try:
            with open(log_path, 'w') as fh:
                fh.write(header + '\n')
                fh.flush()
                while not stop_event.is_set():
                    usage = query_gpu_memory(gpu_ids)
                    now = datetime.utcnow().isoformat()
                    if usage is None:
                        stats['error'] = 'nvidia-smi unavailable'
                        fh.write(f"{now},error\n")
                        fh.flush()
                        break
                    row = [str(usage.get(gid, '')) for gid in gpu_ids]
                    fh.write(f"{now}," + ','.join(row) + '\n')
                    fh.flush()
                    for gid in gpu_ids:
                        try:
                            stats['max'][gid] = max(stats['max'][gid], int(usage.get(gid, 0)))
                        except (TypeError, ValueError):
                            continue
                    if stop_event.wait(interval):
                        break
        except OSError:
            stats['error'] = 'Failed to write VRAM log'

    thread = Thread(target=monitor_loop, daemon=True)
    thread.start()
    return {
        'thread': thread,
        'stop_event': stop_event,
        'log_path': log_path,
        'gpu_ids': gpu_ids,
        'stats': stats,
    }


def stop_vram_monitor(monitor):
    if not monitor:
        return
    monitor['stop_event'].set()
    monitor['thread'].join(timeout=5)


def append_vram_summary(summary_path, job_name, run_name, monitor):
    summary_path.parent.mkdir(exist_ok=True)
    exists = summary_path.exists()
    with open(summary_path, 'a') as fh:
        if not exists:
            fh.write('timestamp,run_name,model,gpus,max_usage_mb,recommended_mb,log_path,status\n')
        timestamp = datetime.utcnow().isoformat()
        gpu_str = ';'.join(str(g) for g in monitor['gpu_ids'])
        max_values = [str(monitor['stats']['max'].get(g, 0)) for g in monitor['gpu_ids']]
        recommended = []
        for val in max_values:
            try:
                recommended.append(str(int(math.ceil(int(val) * 1.1))))
            except ValueError:
                recommended.append('')
        status = monitor['stats']['error'] or 'ok'
        fh.write(
            f"{timestamp},{run_name},{job_name},{gpu_str},{'|'.join(max_values)},{'|'.join(recommended)},{monitor['log_path']},{status}\n"
        )

def make_cmd(job, run_name):
    if run_name is None:
        raise ValueError("run_name must be provided to make_cmd")

    base = [
        'python', 'run.py',
        '--data', 'NewtPhys_SingleImage_100',
        '--model', job['model'],
    ]
    print("run_name in make_cmd:", base)
    if job.get('extra'):
        base += job['extra']
    if job.get('uv'):
        pref = ['uv','run']
        for p in job['uv']:
            pref += ['--with', p]
        return pref + base
    return base

def run_one_experiment(run_name='default_run'):

    print("Starting experiment with run name:", run_name)
    print()

    logs = pathlib.Path('logs'); logs.mkdir(exist_ok=True)
    vram_logs = pathlib.Path('vram_logs'); vram_logs.mkdir(exist_ok=True)
    vram_summary_path = vram_logs / 'summary.csv'
    completed_jobs = []
    overall_start_time = datetime.now()

    JOBS = JOBS_ALL.copy()
    print("Total jobs to run:", len(JOBS))

    for job in JOBS:
        job = job.copy()
        model_name = job['model']
        g = job.get('g', 1)
        if g < 1:
            raise ValueError(f"Job {model_name} requested invalid GPU count: {g}")
        if g > len(GPUS):
            raise ValueError(f"Not enough GPUs ({len(GPUS)}) for job {model_name} requesting {g}")

        devs = GPUS[:g]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in devs)

        cmd = make_cmd(job, run_name=run_name)
        print("running the command:", ' '.join(cmd))
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_path = logs / f'{ts}_{safe(model_name)}_g{g}.log'
        logf = open(log_path, 'w')

        start_time = datetime.now()
        print(f'Starting: {model_name} at {start_time.strftime("%H:%M:%S")} on GPUs {env["CUDA_VISIBLE_DEVICES"]}')

        monitor = start_vram_monitor(model_name, run_name, devs, vram_logs)
        return_code = None

        try:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            return_code = p.wait()
        finally:
            stop_vram_monitor(monitor)
            append_vram_summary(vram_summary_path, model_name, run_name, monitor)
            logf.close()

        end_time = datetime.now()
        duration = end_time - start_time
        completed_jobs.append({
            'model': model_name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'return_code': return_code
        })

        status = "✓" if return_code == 0 else "✗"
        print(f'{status} Completed: {model_name} in {duration} (return code: {return_code}) {len(JOBS) - len(completed_jobs)} jobs remaining.')

    # Print final summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time

    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)

    # Sort by duration for summary
    completed_jobs.sort(key=lambda x: x['duration'], reverse=True)

    successful_jobs = [j for j in completed_jobs if j['return_code'] == 0]
    failed_jobs = [j for j in completed_jobs if j['return_code'] != 0]

    print(f"\nSUCCESSFUL MODELS ({len(successful_jobs)}):")
    print("-" * 50)
    for job in successful_jobs:
        hours, remainder = divmod(job['duration'].total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
        print(f"✓ {job['model']:<40} {duration_str}")

    if failed_jobs:
        print(f"\nFAILED MODELS ({len(failed_jobs)}):")
        print("-" * 50)
        for job in failed_jobs:
            hours, remainder = divmod(job['duration'].total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
            else:
                duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
            print(f"✗ {job['model']:<40} {duration_str} (code: {job['return_code']})")

    # Overall statistics
    total_model_time = sum(job['duration'].total_seconds() for job in completed_jobs)
    avg_time = total_model_time / len(completed_jobs) if completed_jobs else timedelta(0)

    print(f"\nOVERALL STATISTICS:")
    print("-" * 50)

    def to_seconds(x):
        if isinstance(x, timedelta):
            return x.total_seconds()
        elif isinstance(x, (int, float)):
            return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def format_duration(td):
        total_seconds = int(to_seconds(td))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            return f"{int(minutes):02d}m {int(seconds):02d}s"

    print(f"Total models completed: {len(completed_jobs)}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Overall wall time: {format_duration(overall_duration)}")
    print(f"Total model time: {format_duration(total_model_time)}")
    print(f"Average time per model: {format_duration(avg_time)}")
    if overall_duration.total_seconds() > 0:
        parallelization_factor = total_model_time / overall_duration.total_seconds()
        print(f"Parallelization factor: {parallelization_factor:.2f}x")

    print("="*80)


def main():

    # we have a list of experiments with different run names

    runs_config = {
        # new runs for analysis
        "10K_general":{
            "run_name": "run_09_general",
            "quantity": "10K"
        }
    }

    karo = ""
    if os.path.exists("/home/it4i-thvu/seb_dev/computational_physics/PhysBench-Sebfork"):
        karo="karo_"

    for run_name, config in runs_config.items():
        # config["run_name"] = f"run_{str(GENERAL_RUN_COUNT).zfill(2)}_{run_name}"
        print(f"Starting experiment: {config['run_name']}")
        run_name_and_path = f"{config['run_name']}/test_{config['run_name']}_{karo}{config['quantity']}"
        run_one_experiment(run_name=run_name_and_path)

if __name__ == '__main__':
    main()
