"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']

import os
import zipfile
import logging
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('3f5b78b0c982f8bdf3a2c3a27e6136d4d2680e96', 'mobilenetv2_yolov3.param'),
    ('0705b0f8fe5a77718561b9b7d6ed4f33fcd3d455', 'mobilenetv2_yolov3.bin'),
]}

apache_repo_url = 'https://github.com/nihui/ncnn-assets/raw/master/models/'
_url_format = '{repo_url}{file_name}'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join('~', '.ncnn', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.ncnn/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if 'NCNN_HOME' in os.environ:
        root = os.path.join(os.environ['NCNN_HOME'], 'models')

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}'.format(name=name)
    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name)
    lockfile = os.path.join(root, file_name + '.lock')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root)

    with portalocker.Lock(lockfile, timeout=int(os.environ.get('GLUON_MODEL_LOCK_TIMEOUT', 300))):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning("Hash mismatch in the content of model file '%s' detected. "
                                "Downloading again.", params_path)
        else:
            logging.info('Model file not found. Downloading.')

        zip_file_path = os.path.join(root, file_name)
        repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        if zip_file_path.endswith(".zip"):
            with zipfile.ZipFile(zip_file_path) as zf:
                zf.extractall(root)
            os.remove(zip_file_path)
        # Make sure we write the model file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')


def purge(root=os.path.join('~', '.mxnet', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))
