import subprocess


def get_version():
    try:
        with open('VERSION', 'r') as f:
            version = f.read()
    except FileNotFoundError:
        last_tag = subprocess.check_output(
            'git describe --tags `git rev-list --tags --max-count=1`',
            shell=True).decode().strip()
        version = f'{last_tag}-dev'

    return version
