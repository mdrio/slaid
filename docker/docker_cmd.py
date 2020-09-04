#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import logging
import os
import subprocess

logging.basicConfig(level=logging.DEBUG)

DIR = os.path.dirname(os.path.realpath(__file__))


def build(
    image='slaid',
    lib_version='',
    docker_build_dir='../docker-build',
):
    if not os.path.exists(docker_build_dir):
        os.mkdir(docker_build_dir)

    kwargs_list = [
        dict(
            dockerfile='Dockerfile',
            build_dir=docker_build_dir,
            image=image,
            tag=f'{lib_version}',
        )
    ]

    for model_path, model_name, feature, dockerfile in get_models():
        kwargs_list.append(
            dict(dockerfile=dockerfile,
                 build_dir=docker_build_dir,
                 image=image,
                 tag=f'{lib_version + "-" if lib_version else ""}{model_name}',
                 build_args=[f'MODEL={model_path}', f'FEATURE={feature}']))

    for kwargs in kwargs_list:
        docker_build(**kwargs)


def docker_build(dockerfile, build_dir, image, tag='', build_args=None):
    build_args = build_args or []
    build_args = f'{" ".join(["--build-arg " + arg for arg in build_args])}'\
        if build_args else ''
    command = (f'docker build -f {dockerfile} '
               f'-t {image}{":" + tag if tag else ""} '
               f'{build_dir} '
               f'{build_args}')

    logging.debug(command)
    subprocess.run(command, shell=True, check=True)


def push(image='slaid', lib_version='', repo=''):
    kwargs_list = [dict(image=image, tag=f'{lib_version}', repo=repo)]

    for model_path, model_name, feature, dockerfile in get_models():
        kwargs_list.append(
            dict(image=image,
                 tag=f'{lib_version + "-" if lib_version else ""}{model_name}',
                 repo=repo))

    for kwargs in kwargs_list:
        docker_push(**kwargs)


def docker_push(image, tag, repo):
    command = (f'docker push {repo + "/" if repo else ""}{image}'
               f'{":" + tag if tag else ""}')
    logging.debug(command)
    subprocess.run(command, shell=True, check=True)


def tag(repo, image='slaid', lib_version=''):
    kwargs_list = [dict(repo=repo, image=image, tag=f'{lib_version}')]

    for _, model_name, _, _ in get_models():
        kwargs_list.append(
            dict(
                repo=repo,
                image=image,
                tag=f'{lib_version + "-" if lib_version else ""}{model_name}',
            ))

    for kwargs in kwargs_list:
        docker_tag(**kwargs)


def docker_tag(repo, image, tag):
    tag = ":" + tag if tag else ""
    command = f'docker tag {image}{tag} {repo}/{image}{tag}'
    logging.debug(command)
    subprocess.run(command, shell=True, check=True)


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='lib_version', default='')
    subparsers = parser.add_subparsers()

    build_parser = subparsers.add_parser('build')
    build_parser.set_defaults(func=build)
    build_parser.add_argument('-d',
                              dest='docker_build_dir',
                              default='../docker-build')

    push_parser = subparsers.add_parser('push')
    push_parser.add_argument('-r', dest='repo')
    push_parser.set_defaults(func=push)

    push_parser = subparsers.add_parser('tag')
    push_parser.add_argument('-r', dest='repo')
    push_parser.set_defaults(func=tag)

    args = parser.parse_args()
    args_dct = vars(args)
    func = args_dct.pop('func')
    func(**args_dct)
