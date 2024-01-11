import os
import sys
import subprocess
import threading
import locale
import traceback
import re

from .config import root_path

plugin_name = os.path.basename(root_path)
windows_not_install = ['mmcv_full\n']

def log(msg, end=None, file=None):
    print(f'{plugin_name} :', msg, end=end, file=file)

def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            log(msg, end="", file=sys.stdout)
        else:
            log(msg, end="", file=sys.stderr)

def process_wrap(cmd_str, cwd=None, handler=None):
    log(f"EXECUTE: {cmd_str} in '{cwd}'")
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
            log(f"Failed to retrieve the information of installed pip packages.")
            return set()

    return pip_list

def mmcv_install():
    process_wrap(pip_install + ['-U', 'openmim'], cwd=root_path)
    process_wrap(mim_install + ['mmcv-full'], cwd=root_path)
    pass

def is_installed(name):
    name = name.strip()
    pattern = r'([^<>!=]+)([<>!=]=?)'
    match = re.search(pattern, name)

    if match:
        name = match.group(1)

    result = name.lower() in get_installed_packages()
    return result

def check_and_install_requirements(file_path):
    log(file_path)
    version = sys.version_info[:2]
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                log(line)
                if not is_installed(line):
                    if platform.system() == "Windows" and version[1] == 11 and 'insightface' in line:
                        process_wrap(pip_install + ['insightface-0.7.3-cp311-cp311-win_amd64.whl'], cwd=root_path)
                        continue
                    if platform.system() == "Windows" and line in windows_not_install:
                        log(f"windows skip {line}")
                        continue
                    log(f"install {line}")
                    process_wrap(pip_install + [line], cwd=root_path)
            return False
    return True

try:
    import platform

    log("### : Check dependencies")
    if "python_embed" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, '-s', '-m', 'pip', 'install', '-q']
        mim_install = [sys.executable, '-s', '-m', 'mim', 'install', '-q']
    else:
        pip_install = [sys.executable, '-m', 'pip', 'install', '-q']
        mim_install = [sys.executable, '-m', 'mim', 'install', '-q']

    subpack_req = os.path.join(root_path, "requirements.txt")
    # mmcv_install()
    check_and_install_requirements(subpack_req)
    if sys.argv[0] == 'install.py':
        sys.path.append('..')  # for portable version

except Exception as e:
    log("Dependency installation has failed. Please install manually.")
    traceback.print_exc()
