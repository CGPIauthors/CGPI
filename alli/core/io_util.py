import gzip
import lz4.frame

class LZ4FileWrapper(object):
    @classmethod
    def open(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, filename, mode='rb', *args, **kwargs):
        self._open_args = dict(
            filename=filename,
            mode=mode,
            args=args,
            kwargs=kwargs,
        )
        self._file = lz4.frame.open(filename, mode, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._file, name)

    def flush(self):
        self._file.flush()
        self._file.close()
        filename = self._open_args['filename']
        mode = self._open_args['mode'].replace('w', 'a')
        args = self._open_args['args']
        kwargs = self._open_args['kwargs']
        self._file = lz4.frame.open(filename, mode, *args, **kwargs)


class GzipFileWrapper(object):
    @classmethod
    def open(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, filename, mode='rb', *args, **kwargs):
        self._open_args = dict(
            filename=filename,
            mode=mode,
            args=args,
            kwargs=kwargs,
        )
        self._file = gzip.open(filename, mode, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._file, name)

    def flush(self):
        self._file.flush()
        self._file.close()
        filename = self._open_args['filename']
        mode = self._open_args['mode'].replace('w', 'a')
        args = self._open_args['args']
        kwargs = self._open_args['kwargs']
        self._file = gzip.open(filename, mode, *args, **kwargs)


