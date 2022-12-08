#!/usr/bin/env python3

import argparse
import ast
import importlib
import inspect
import os
import pprint
import subprocess
import sys
import types
import megfile
from loguru import logger

from basedet.configs import BaseConfig, ConfigDict

_OTHER_DEF = []
_IMPORT_SOURCE = None
_IMPORT_ALIAS = []


def make_parser():
    parser = argparse.ArgumentParser(description="A script that auto convert config")
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="path config file"
    )
    parser.add_argument(
        "-o", "--output", default="new_config.py", type=str, help="name of output file"
    )
    parser.add_argument(
        "-p", "--path", default=None, type=str, help="output path, default to config file path"
    )
    return parser


def get_func_def(func):
    func_source = inspect.getsource(func)
    return func_source.split(":" + os.linesep, maxsplit=1)[0] + ":" + os.linesep


def get_func_content(func):
    return inspect.getsource(func).split(":", maxsplit=1)[-1]


def get_property_source(obj_prop):
    prop_func_name = ("fget", "fset", "fdel")
    prop_source_list = []
    for f in prop_func_name:
        func = getattr(obj_prop, f, None)
        if func:
            prop_source_list.append(inspect.getsource(func))
    return prop_source_list


def function2source(func):
    global _OTHER_DEF, _IMPORT_ALIAS
    if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
        # lambda function
        func_source = inspect.getsource(func)
        if func_source.strip().startswith("self."):
            return "lambda" + func_source.rsplit("lambda", maxsplit=1)[-1]
        else:
            _OTHER_DEF.append(func_source)
            lambda_name = func_source.rsplit("=", maxsplit=1)[0].strip()
            return lambda_name
    else:
        # non-lambda function
        func_name = func.__name__
        if func_name not in _IMPORT_ALIAS:
            _OTHER_DEF.append(inspect.getsource(func))
        return func_name


def contains_inf(v):
    if isinstance(v, (tuple, list)):
        return any(float("inf") == x or contains_inf(x) for x in v)


def pretty_dict_code(d, depth=1):
    dict_string = "dict(" + os.linesep
    python_indent_string = " " * 4
    code_indent = python_indent_string * depth
    for k, v in d.items():
        if isinstance(v, dict):
            pretty_dict = pretty_dict_code(v, depth=depth + 1)
            format_code = "{}={}".format(k, pretty_dict)
        elif v == float("inf"):
            format_code = f"{k}=float('inf')"
        elif contains_inf(v):
            replace_inf_str = pprint.pformat(v).replace("inf", "float('inf')")
            format_code = f"{k}={replace_inf_str}"
        else:
            format_code = "{}={}".format(k, pprint.pformat(v))
        dict_string += code_indent + format_code + "," + os.linesep

    dict_string += python_indent_string * (depth - 1) + ")" + os.linesep
    return dict_string


def values2source(values):
    if isinstance(values, ConfigDict):
        return pretty_dict_code(values.to_dict())
    elif isinstance(values, types.FunctionType):
        return function2source(values)
    else:
        value_name = getattr(values, "__name__", None)
        if value_name is None:
            return pprint.pformat(values)
        elif value_name in _IMPORT_ALIAS:
            return value_name
        else:
            logger.warning(f"unsupport type {value_name}, use repr to cast")
            return repr(values)


def add_code_indent(source, indent_string=" " * 4):
    return os.linesep.join([indent_string + s for s in source.splitlines()])


def get_indent_length(func):
    """get indent number of function"""
    source = inspect.getsource(func)
    def_line = source.splitlines()[0]
    def_line_indent = len(def_line) - len(def_line.strip())
    for first_indent_line in source.split(":")[1].splitlines():
        if first_indent_line:
            break
    first_line_indent = len(first_indent_line) - len(first_indent_line.strip())
    indent_len = first_line_indent - def_line_indent
    assert indent_len > 0
    return indent_len


def lint_source(filename):
    try:
        subprocess.call(["black", filename])
    except FileNotFoundError:
        logger.warning("Please install black to lint source code")
    try:
        subprocess.call(["isort", "-rc", filename])
    except Exception:
        pass


def generate_import_alias(source):
    alias_list = []
    module = ast.parse(source)
    for body in module.body:
        for name in body.names:
            if name.asname is not None:
                alias_list.append(name.asname)
            else:
                assert name.name is not None
                alias_list.append(name.name)

    return alias_list


def check_cfg_diff(cfg1, cfg2):
    diff1 = cfg1.diff(cfg2)
    diff2 = cfg2.diff(cfg1)
    assert set(diff1.keys()) == set(diff2.keys())
    for k, v in diff1.items():
        if not isinstance(v, types.FunctionType):
            assert not v
    logger.info("All check passed...")


class CodeReader:

    def __init__(self, pyobj, root_type, skipped_funcname=None):
        self.pyobj = pyobj
        self.obj_class = pyobj.__class__
        self.obj_module = inspect.getmodule(self.pyobj)
        self.root_type = root_type
        if skipped_funcname is None:
            skipped_funcname = ["__module__", "__doc__"]
        self.skipped_funcname = skipped_funcname

    def parse(self, dump_filename="new_config.py"):
        import_string = self.generate_import()
        obj_string = self.generate_obj()
        # other definition should be dumped at the end of time
        other_def_string = self.generate_other_def()
        write_str = os.linesep.join([import_string, other_def_string, obj_string])
        with megfile.smart_open(dump_filename, "w") as f:
            f.write(write_str)
        logger.info(f"config is dumped into file {dump_filename}")

    def generate_obj(self):
        def_string = self.generate_class_def()
        static_string = self.generate_static_attr()
        init_string = self.generate_obj_init()
        func_string = self.generate_obj_function()
        return os.linesep.join([def_string, static_string, init_string, func_string])

    def generate_import(self):
        module_source = inspect.getsource(self.obj_module)
        source_in_lines = module_source.splitlines()

        def is_import_line(s):
            return s.strip().startswith("import") or s.strip().startswith("from")

        def is_empty_lines(s):
            return not s.strip()

        def get_start_index(sources):
            for idx, s in enumerate(sources):
                if is_import_line(s):
                    return idx

        def get_end_index(sources, skip_lines=0):
            for idx, s in enumerate(sources):
                if idx < skip_lines:
                    continue
                elif not (is_import_line(s) or is_empty_lines(s) or s.startswith(" ")):
                    return idx

        import_source = ["from basedet.configs import BaseConfig"]
        start_index = get_start_index(source_in_lines)
        end_index = get_end_index(source_in_lines, start_index)
        import_source.extend(source_in_lines[start_index:end_index])
        import_source = os.linesep.join(import_source)
        global _IMPORT_SOURCE, _IMPORT_ALIAS
        _IMPORT_SOURCE = import_source
        _IMPORT_ALIAS = generate_import_alias(import_source)
        return import_source

    def generate_other_def(self):
        global _OTHER_DEF
        return (os.linesep * 2).join(_OTHER_DEF)

    def generate_static_attr(self):
        attr_string = []
        for k, v in vars(self.pyobj.__class__).items():
            if k.startswith("__") and k.endswith("__"):
                continue
            attr = inspect.getattr_static(self.pyobj, k, None)
            if attr:
                if "method" in repr(attr):
                    continue
                if callable(attr):
                    if f".{k}" in repr(attr):
                        # skip normal callable method
                        continue
                    if inspect.getsourcefile(v) == inspect.getsourcefile(self.obj_module):
                        global _OTHER_DEF
                        _OTHER_DEF.append(inspect.getsource(v))
                    attr_string.append(f"{k} = {v.__name__}")
                else:
                    attr_string.append(f"{k} = {attr}")
        source = os.linesep.join(attr_string)
        indent_length = get_indent_length(self.pyobj.__init__)
        return add_code_indent(source, " " * indent_length)

    def generate_class_def(self):
        return "class {}({}):".format(self.obj_class.__name__, self.root_type.__name__)

    def generate_obj_function(self, skipped_method=("__init__",)):
        # `__init__` function is generate in `generate_obj_init` method.
        func_string_list = []
        for k, v in vars(self.pyobj.__class__).items():
            if k in skipped_method:
                continue
            if callable(v):
                if f".{k}" in repr(v):
                    func_string_list.append(inspect.getsource(v))
            elif isinstance(v, property):
                func_string_list.extend(get_property_source(v))
            else:
                attr = getattr(self.pyobj, k)
                if callable(attr):
                    # static and classmethod
                    func_string_list.extend(inspect.getsource(attr))

        return "".join(func_string_list)

    def generate_obj_init(self):
        func_def = get_func_def(self.pyobj.__init__)
        init_attr_list = []
        for k, v in vars(self.pyobj).items():
            init_attr_list.append("self.{} = {}".format(k, values2source(v)))
        func_content = os.linesep.join(init_attr_list)
        indent_length = get_indent_length(self.pyobj.__init__)
        func_content = add_code_indent(func_content, " " * indent_length * 2)
        return func_def + func_content


@logger.catch
def main():
    args = make_parser().parse_args()
    sys.path.append(os.path.dirname(args.file))
    module = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg = module.Cfg()
    assert isinstance(cfg, ConfigDict), f"cfg in {args.file} is not a ConfigDict"

    reader = CodeReader(cfg, root_type=BaseConfig)
    dirname = os.path.dirname(args.file) if args.path is None else args.path
    output_file = os.path.join(dirname, args.output)
    reader.parse(output_file)
    lint_source(output_file)

    module = importlib.import_module(os.path.basename(args.output).split(".")[0])
    new_cfg = module.Cfg()
    check_cfg_diff(cfg, new_cfg)


if __name__ == "__main__":
    main()
