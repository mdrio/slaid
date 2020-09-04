#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import subprocess

DIR = os.path.dirname(os.path.realpath(__file__))


def get_models():
    dockerfiles = glob.glob(f'{os.path.join(DIR,"Dockerfile.*")}')
    for dockerfile in dockerfiles:
        model_type = os.path.splitext(dockerfile)[1][1:]
        feature = model_type.split('_')[1]
        models = glob.glob(f'../slaid/models/{model_type}*')
        for model in models:
            model_path = os.path.basename(model)
            model_name = os.path.splitext(model_path)[0]
            yield model_path, model_name, feature, dockerfile


def docker_build(
    lib_version='',
    docker_build_dir='../docker-build',
):
    if not os.path.exists(docker_build_dir):
        os.mkdir(docker_build_dir)
    for model_path, model_name, feature, dockerfile in get_models():
        command = f'docker build {docker_build_dir} -f {dockerfile} -t slaid:{lib_version + "-" if lib_version else ""}{model_name} --build-arg MODEL={model_path} --build-arg FEATURE={feature}'
        subprocess.run(command.split(' '), check=True)


def docker_push(repo, lib_version=''):
    for model_path, model_name, _, _ in get_models():
        img = f'slaid:{lib_version + "-" if lib_version else ""}{model_name}'
        print(img)
        commands = []
        commands.append(f'docker tag {img} {repo}/{img}')
        commands.append(f'docker push {repo}/{img}')

    for command in commands:
        subprocess.run(command.split(' '), check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='lib_version', default='')
    subparsers = parser.add_subparsers()
    build_parser = subparsers.add_parser('build')
    build_parser.set_defaults(func=docker_build)

    build_parser.add_argument('-d',
                              dest='docker_build_dir',
                              default='../docker-build')

    push_parser = subparsers.add_parser('push')
    push_parser.add_argument('-r', dest='repo')
    push_parser.set_defaults(func=docker_push)
    args = parser.parse_args()
    args_dct = vars(args)
    func = args_dct.pop('func')
    func(**args_dct)
