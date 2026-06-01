#!/usr/bin/env python3
"""Compatibility wrapper for the repeatable corpus/vocab pipeline."""

import sys

from data_gen.prepare_corpus import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("build")
    main()
