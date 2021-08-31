from typing import Dict, Any


def flatten_dict(args: Dict[str, Any]):
    data = {}
    if isinstance(args, dict):
        for k, v in args.items():
            if isinstance(v, dict):
                data.update(flatten_dict(v))
            elif isinstance(v, list):
                contain_dict = False
                for i in v:
                    res_val = flatten_dict(i)
                    if res_val:
                        data.update(res_val)
                        contain_dict = True
                if not contain_dict:
                    data.update({k: v})
            else:
                data.update({k: v})
    elif isinstance(args, list):
        for v in args:
            data.update(flatten_dict(v))
    else:
        return {}
    return data
