from fvcore.common.config import CfgNode as _CfgNode
from utils.logger import setup_logger

_C = _CfgNode()
_C.VERSION = 2
_C.DETECTOR_MODEL_NAME = ""
_C.FACE_ID_MODEL_NAME = ""
_C.CAMERA_LIST = []
_C.GPU_ASSIGNED_LIST = []
_C.SERVER_URL = ""
_C.OUTPUT_DIM = 0
_C.INPUT_W = 0
_C.INPUT_H = 0
_C.MODEL_C = 0
_C.MODEL_H = 0
_C.MODEL_W = 0
_C.OUTPUT_DIM = 0
_C.CHANNELS = 0
_C.CONF_THRESH = 0.
_C.IOU_THRESHOLD = 0.
_C.OUTPUT_MEDIA_FILENAME = ""
_C.INPUT_TENSOR_NAME = ""
_C.OUTPUT_TENSOR_NAME = ""


def get_cfg(filename: str, allow_unsafe: bool = False):
    cfg = _C.clone()
    cfg.merge_from_file(filename, allow_unsafe=allow_unsafe)
    return cfg


logger = setup_logger(name='_')


def get_logger():
    return logger


import functools
import inspect
from utils.logger import setup_logger
from fvcore.common.config import CfgNode as _CfgNode

from fvcore.common.file_io import PathManager


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:
    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.
    """

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode

        latest_ver = _C.VERSION
        assert (
                latest_ver == self.VERSION
        ), "CfgNode.merge_from_file is only allowed on a config object of latest version!"

        # logger = logging.getLogger(__name__)
        logger = setup_logger(name='person_track')
        loaded_ver = loaded_cfg.get("VERSION", None)
        logger.debug(f"loaded_ver is: {loaded_ver}")
        assert loaded_ver == self.VERSION, "Cannot merge a v{} config into a v{} config.".format(
            loaded_ver, self.VERSION
        )

        if loaded_ver == self.VERSION:
            self.merge_from_other_cfg(loaded_cfg)

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


# global_cfg = CfgNode()


# def set_global_cfg(cfg: CfgNode) -> None:
#     """
#     Let the global config point to the given cfg.
#     Assume that the given "cfg" has the key "KEY", after calling
#     `set_global_cfg(cfg)`, the key can be accessed by:
#     ::
#         from detectron2.config import global_cfg
#         print(global_cfg.KEY)
#     By using a hacky global config, you can access these configs anywhere,
#     without having to pass the config object or the values deep into the code.
#     This is a hacky feature introduced for quick prototyping / research exploration.
#     """
#     global global_cfg
#     global_cfg.clear()
#     global_cfg.update(cfg)


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    def check_docstring(func):
        if func.__module__.startswith("detectron2."):
            assert (
                    func.__doc__ is not None and "experimental" in func.__doc__.lower()
            ), f"configurable {func} should be marked experimental"

    if init_func is not None:
        assert (
                inspect.isfunction(init_func)
                and from_config is None
                and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."
        check_docstring(init_func)

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            check_docstring(orig_func)

            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    if len(args) and isinstance(args[0], _CfgNode):
        return True
    if isinstance(kwargs.pop("cfg", None), _CfgNode):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False