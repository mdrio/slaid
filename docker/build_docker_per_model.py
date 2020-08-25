#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import subprocess
import pkg_resources

DIR = os.path.dirname(os.path.realpath(__file__))


def main(lib_version='', docker_build_dir='../docker-build'):
    if not os.path.exists(docker_build_dir):
        os.mkdir(docker_build_dir)
    dockerfiles = glob.glob(f'{os.path.join(DIR,"Dockerfile.*")}')
    for dockerfile in dockerfiles:
        model_type = os.path.splitext(dockerfile)[1][1:]
        models = glob.glob(f'../slaid/models/{model_type}*')
        for model in models:
            model = os.path.basename(model)
            model_name = os.path.splitext(model)[0]
            #  model = pkg_resources.resource_filename('slaid', f'models/{model}')
            command = f'docker build {docker_build_dir} -f {dockerfile} -t slaid:{lib_version + "-" if lib_version else ""}{model_name} --build-arg MODEL={model}'
            print(command)
            print(command.split())
            subprocess.run(command.split(' '), check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='lib_version', default='')
    parser.add_argument('-d',
                        dest='docker_build_dir',
                        default='../docker-build')

    args = parser.parse_args()
    main(args.lib_version, args.docker_build_dir)
