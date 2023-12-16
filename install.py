import platform
import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback
import re

main_path = os.path.dirname(__file__)
sys.path.append(main_path)

windows_not_install = ['mmcv_full']

def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)

def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[Impact Pack] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

# ---
pip_list = None

def get_installed_packages():
    global pip_list
    if pip_list is None:
        try:
            result = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], universal_newlines=True)
            pip_list = set([line.split()[0].lower() for line in result.split('\n') if line.strip()])
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-Manager] Failed to retrieve the information of installed pip packages.")
            return set()

    return pip_list

def is_installed(name):
    name = name.strip()
    pattern = r'([^<>!=]+)([<>!=]=?)'
    match = re.search(pattern, name)

    if match:
        name = match.group(1)

    result = name.lower() in get_installed_packages()
    return result

def check_and_install_requirements(file_path):
    print(f"req_path: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not is_installed(line):
                    if platform.system() == "Windows" and line in windows_not_install:
                        continue
                    else:
                        process_wrap(pip_install + [line], cwd=main_path)
            return False
    return True

try:
    import platform

    print("### ComfyUI-Impact-Pack: Check dependencies")
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
        mim_install = [sys.executable, '-s', '-m', 'mim', 'install']
    else:
        pip_install = [sys.executable, '-m', 'pip', 'install']
        mim_install = [sys.executable, '-m', 'mim', 'install']

    subpack_req = os.path.join(main_path, "requirements.txt")
    check_and_install_requirements(subpack_req)

    if platform.system() != "Windows":
        process_wrap([sys.executable, '-s', '-m', 'pip', 'install', '-q', 'mmcv_full'])

    if sys.argv[0] == 'install.py':
        sys.path.append('.')  # for portable version

except Exception as e:
    print("[ERROR] ComfyUI-Impact-Pack: Dependency installation has failed. Please install manually.")
    traceback.print_exc()
