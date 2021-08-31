import dataclasses
import re
import sys
import copy
import json
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union, Dict

from transformers.hf_argparser import HfArgumentParser as ArgumentParser


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def lambda_field(default, **kwargs):
    return field(default_factory=lambda: copy.copy(default))


class HfArgumentParser(ArgumentParser):
    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            arg_name = dtype.__mro__[-2].__name__
            inputs = {k: v for k, v in data[arg_name].items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_yaml_file(self, yaml_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a yaml file and populating the
        dataclass types.
        """
        data = yaml.load(Path(yaml_file).read_text(), Loader=yaml.Loader)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            arg_name = dtype.__mro__[-2].__name__
            inputs = {k: v for k, v in data[arg_name].items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
