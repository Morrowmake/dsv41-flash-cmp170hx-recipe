#!/usr/bin/env python3
"""Validate the number of filesystem layers in a release image."""
import argparse
import json
import os
import sys

LIMIT = 123


def check(count, enabled=None):
    flag = os.getenv('IMAGE_LAYER_COUNT_CHECK', '1') if enabled is None else enabled
    if flag not in ('0', '1'):
        raise ValueError('IMAGE_LAYER_COUNT_CHECK must be 0 or 1')
    print(f'[image-layers] count={count} limit={LIMIT} check={flag}', flush=True)
    if flag == '1' and count > LIMIT:
        raise ValueError(f'image has {count} layers; maximum is {LIMIT}')
    return count


def from_inspect(value):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError('expected one Docker image inspection')
    layers = value[0]['RootFS']['Layers']
    if not isinstance(layers, list):
        raise ValueError('invalid filesystem layer list')
    return len(layers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        check(from_inspect(json.load(sys.stdin)))
    except (ValueError, KeyError, TypeError) as error:
        print('[image-layers] FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
