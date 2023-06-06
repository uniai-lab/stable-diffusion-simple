# this scripts installs necessary requirements and launches main program in webui.py
import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import json

from modules import cmd_args
from modules.paths_internal import script_path, extensions_dir

commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)

args, _ = cmd_args.parser.parse_known_args()

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
stored_git_tag = None
dir_repos = "repositories"

if 'GRADIO_ANALYTICS_ENABLED' not in os.environ:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def git_tag():
    global stored_git_tag

    if stored_git_tag is not None:
        return stored_git_tag

    try:
        stored_git_tag = run(f"{git} describe --tags").strip()
    except Exception:
        stored_git_tag = "<none>"

    return stored_git_tag


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(".")

        print(run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env))
    except Exception as e:
        print(e, file=sys.stderr)


def list_extensions(settings_file):
    settings = {}

    try:
        if os.path.isfile(settings_file):
            with open(settings_file, "r", encoding="utf8") as file:
                settings = json.load(file)
    except Exception as e:
        print(e, file=sys.stderr)

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none':
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    for dirname_extension in list_extensions(settings_file):
        run_extension_installer(os.path.join(extensions_dir, dirname_extension))

def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()


if __name__ == "__main__":
    start()
