import os
import tarfile
import argparse

SOURCES = {
    'art-faces-256': 'https://www.dropbox.com/s/8shgau6vvj8mfcw/256-art-faces.tar',
    'art-landscapes-art-256': 'https://www.dropbox.com/s/wc2jr1c0kzgxgzf/256-art-landscapes.tar',
    'mountains-256': 'https://www.dropbox.com/s/7e3y2xkn08mthct/256-mountains.tar',
}


def download(source, destination):
    tmp_tar = os.path.join(destination, '.tmp.tar')
    # urllib has troubles with dropbox
    os.system(f'wget {source} -O {tmp_tar}')
    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(destination)

    os.remove(tmp_tar)


def main():
    parser = argparse.ArgumentParser(description='Pretrained models loader')
    parser.add_argument('--models', nargs='+', type=str,
                        choices=list(SOURCES.keys()) + ['all'], default=['all'])
    parser.add_argument('--out', type=str, help='root out dir')

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    models = args.models
    if 'all' in models:
        models = list(SOURCES.keys())

    for model in set(models):
        source = SOURCES[model]
        print(f'downloading {model}\nfrom {source}')
        download(source, args.out)


if __name__ == '__main__':
    main()