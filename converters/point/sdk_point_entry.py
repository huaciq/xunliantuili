"""SDK launcher: efficient, numerically identical quantization-table I/O.

JSON is a YAML subset. The vendor's round-trip YAML serializer is extremely
slow for large per-channel tables under QEMU. Only .quantize documents use
this adapter; official SDK files and all other YAML handling are unchanged.
"""
import json
import os
from pathlib import Path
import runpy
import re
import sys


def _name(stream):
    return str(getattr(stream, "name", stream))


def _number(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported quantization parameter type: " + type(value).__name__)


def quantization_json(data, indent=2):
    text = json.dumps(data, default=_number, allow_nan=False, indent=indent)
    # YAML 1.1 readers require a decimal point in scientific notation.
    # Match complete JSON strings first, so strings/keys are never rewritten.
    def exponent(match):
        token = match.group(0)
        if token.startswith('"'):
            return token
        mantissa, power = token.lower().split("e")
        return mantissa + ("" if "." in mantissa else ".0") + "e" + power
    return re.sub(r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?[eE][+-]?\d+', exponent, text) + "\n"


def install_quantization_io(parser):
    original_dump, original_load = parser.dump, parser.load

    def dump(data, stream, indent=2, comments=None):
        if not _name(stream).endswith(".quantize"):
            return original_dump(data, stream, indent=indent, comments=comments)
        text = quantization_json(data, indent)
        # Quantization comments do not carry numeric/structural data.
        if hasattr(stream, "write"):
            stream.write(text)
        else:
            Path(stream).write_text(text, encoding="utf-8")
        print("SDK quantization table: JSON/YAML numeric I/O", flush=True)

    def load(stream):
        if not _name(stream).endswith(".quantize"):
            return original_load(stream)
        if hasattr(stream, "read"):
            position = stream.tell()
            prefix = stream.read(1)
            stream.seek(position)
            if prefix == "{":
                return json.load(stream)
        else:
            with open(stream, encoding="utf-8") as file:
                if file.read(1) == "{":
                    file.seek(0)
                    return json.load(file)
        return original_load(stream)

    parser.dump, parser.load = dump, load


def main():
    sdk = sys.argv.pop(1)
    os.chdir(Path(__file__).resolve().parent)
    sys.argv[0] = "pegasus.py"
    import acuitylib.yaml_parser as parser
    install_quantization_io(parser)
    runpy.run_path(sdk, run_name="__main__")


if __name__ == "__main__":
    main()
