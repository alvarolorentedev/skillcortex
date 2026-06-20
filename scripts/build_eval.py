#!/usr/bin/env python3
"""Build the deterministic 150-case execution benchmark."""

import json
import sys
from pathlib import Path


# name, arguments, description, correct expression, mutant expression, assertions
SPECS = [
    (
        "absolute_gap",
        "a, b",
        "returns the absolute difference between two numbers",
        "abs(a - b)",
        "a - b",
        ["{fn}(3, 8) == 5", "{fn}(8, 3) == 5"],
    ),
    (
        "count_truthy",
        "values",
        "counts truthy values",
        "sum(bool(value) for value in values)",
        "len(values)",
        ["{fn}([0, 1, '', 'x']) == 2", "{fn}([]) == 0"],
    ),
    (
        "first_or",
        "values, default=None",
        "returns the first item or a default",
        "values[0] if values else default",
        "values[-1] if values else default",
        ["{fn}([4, 5], 0) == 4", "{fn}([], 9) == 9"],
    ),
    (
        "last_or",
        "values, default=None",
        "returns the last item or a default",
        "values[-1] if values else default",
        "values[0] if values else default",
        ["{fn}([4, 5], 0) == 5", "{fn}([], 9) == 9"],
    ),
    (
        "is_non_decreasing",
        "values",
        "checks whether values are sorted ascending",
        "all(left <= right for left, right in zip(values, values[1:]))",
        "all(left < right for left, right in zip(values, values[1:]))",
        ["{fn}([1, 1, 3])", "not {fn}([2, 1])"],
    ),
    (
        "ordered_unique",
        "values",
        "removes duplicates while preserving order",
        "list(dict.fromkeys(values))",
        "list(set(values))",
        ["{fn}([3, 1, 3, 2]) == [3, 1, 2]", "{fn}([]) == []"],
    ),
    (
        "reverse_words",
        "text",
        "reverses the order of whitespace-separated words",
        "' '.join(reversed(text.split()))",
        "text[::-1]",
        ["{fn}('one two three') == 'three two one'", "{fn}('') == ''"],
    ),
    (
        "vowel_total",
        "text",
        "counts vowels case-insensitively",
        "sum(character.lower() in 'aeiou' for character in text)",
        "sum(character in 'aeiou' for character in text)",
        ["{fn}('Education') == 5", "{fn}('rhythm') == 0"],
    ),
    (
        "shared_sorted",
        "left, right",
        "returns sorted unique values shared by two iterables",
        "sorted(set(left) & set(right))",
        "sorted(set(left) | set(right))",
        ["{fn}([3, 1, 2], [2, 3, 4]) == [2, 3]", "{fn}([], [1]) == []"],
    ),
    (
        "overlay",
        "base, updates",
        "returns a new dictionary with updates applied",
        "{**base, **updates}",
        "{**updates, **base}",
        ["{fn}({'a': 1}, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}", "{fn}({}, {}) == {}"],
    ),
    (
        "divide_or",
        "a, b, default=None",
        "divides a by b or returns a default for zero",
        "default if b == 0 else a / b",
        "default if a == 0 else a / b",
        ["{fn}(8, 2) == 4", "{fn}(8, 0, 'x') == 'x'"],
    ),
    (
        "center_item",
        "values",
        "returns the center item of an odd-length sequence",
        "values[len(values) // 2]",
        "values[(len(values) - 1) // 2 - 1]",
        ["{fn}([1, 2, 3]) == 2", "{fn}(['a']) == 'a'"],
    ),
    (
        "single_spaces",
        "text",
        "collapses all whitespace runs to single spaces",
        "' '.join(text.split())",
        "text.strip()",
        ["{fn}('  a   b ') == 'a b'", "{fn}('\\ta\\nb') == 'a b'"],
    ),
    (
        "name_initials",
        "name",
        "returns uppercase initials from a name",
        "''.join(part[0].upper() for part in name.split())",
        "''.join(part[-1].upper() for part in name.split())",
        ["{fn}('Ada Lovelace') == 'AL'", "{fn}('  grace   hopper ') == 'GH'"],
    ),
    (
        "left_rotate",
        "values, amount",
        "rotates a sequence left by an amount",
        "values[amount % len(values):] + values[:amount % len(values)] if values else values[:]",
        "values[-amount:] + values[:-amount]",
        ["{fn}([1, 2, 3, 4], 1) == [2, 3, 4, 1]", "{fn}([], 3) == []"],
    ),
    (
        "alternating",
        "values",
        "returns items at even indexes",
        "values[::2]",
        "values[1::2]",
        ["{fn}([0, 1, 2, 3]) == [0, 2]", "{fn}([7]) == [7]"],
    ),
    (
        "positive_sum",
        "values",
        "sums only positive numbers",
        "sum(value for value in values if value > 0)",
        "sum(values)",
        ["{fn}([-2, 0, 3, 4]) == 7", "{fn}([-1]) == 0"],
    ),
    (
        "all_same",
        "values",
        "checks whether all values are equal",
        "len(set(values)) <= 1",
        "len(set(values)) == 1",
        ["{fn}([])", "{fn}([2, 2, 2])", "not {fn}([2, 3])"],
    ),
    (
        "position_or_none",
        "values, needle",
        "returns an item's index or None",
        "values.index(needle) if needle in values else None",
        "values.index(needle)",
        ["{fn}(['a', 'b'], 'b') == 1", "{fn}(['a'], 'x') is None"],
    ),
    (
        "adjacent_sums",
        "values",
        "returns sums of adjacent pairs",
        "[left + right for left, right in zip(values, values[1:])]",
        "[left + right for left, right in zip(values[::2], values[1::2])]",
        ["{fn}([1, 2, 3]) == [3, 5]", "{fn}([1]) == []"],
    ),
    (
        "without_none",
        "values",
        "removes None while retaining other falsey values",
        "[value for value in values if value is not None]",
        "[value for value in values if value]",
        ["{fn}([0, None, False, 2]) == [0, False, 2]", "{fn}([None]) == []"],
    ),
    (
        "headline",
        "text",
        "title-cases each word",
        "text.title()",
        "text.capitalize()",
        ["{fn}('hello WORLD') == 'Hello World'", "{fn}('') == ''"],
    ),
    (
        "decimal_digit_sum",
        "number",
        "sums decimal digits and ignores sign",
        "sum(int(digit) for digit in str(abs(number)))",
        "sum(int(digit) for digit in str(number))",
        ["{fn}(407) == 11", "{fn}(-12) == 3"],
    ),
    (
        "integer_or",
        "text, default=0",
        "parses a signed decimal integer or returns a default",
        "int(text) if text.strip().lstrip('+-').isdigit() else default",
        "int(text) if text.isdigit() else default",
        ["{fn}(' -12 ') == -12", "{fn}('x', 7) == 7"],
    ),
    (
        "word_plural",
        "count, singular",
        "returns a count and a simply pluralized word",
        "f'{count} {singular if count == 1 else singular + \"s\"}'",
        "f'{count} {singular}'",
        ["{fn}(1, 'file') == '1 file'", "{fn}(2, 'file') == '2 files'"],
    ),
    (
        "file_extension",
        "filename",
        "returns the final filename extension without a dot",
        "filename.rsplit('.', 1)[1] if '.' in filename.rsplit('/', 1)[-1] else ''",
        "filename.split('.', 1)[1] if '.' in filename else ''",
        ["{fn}('archive.tar.gz') == 'gz'", "{fn}('/tmp/file') == ''"],
    ),
    (
        "capped_total",
        "values, cap",
        "sums values but caps the result",
        "min(sum(values), cap)",
        "max(sum(values), cap)",
        ["{fn}([2, 3], 10) == 5", "{fn}([8, 7], 10) == 10"],
    ),
    (
        "item_counts",
        "values",
        "returns a frequency dictionary",
        "{value: values.count(value) for value in values}",
        "{value: 1 for value in values}",
        ["{fn}(['a', 'b', 'a']) == {'a': 2, 'b': 1}", "{fn}([]) == {}"],
    ),
    (
        "swap_pairs",
        "pairs",
        "swaps left and right in every pair",
        "[(right, left) for left, right in pairs]",
        "[(left, right) for left, right in pairs]",
        ["{fn}([(1, 'a'), (2, 'b')]) == [('a', 1), ('b', 2)]", "{fn}([]) == []"],
    ),
    (
        "longest_or_none",
        "values",
        "returns the longest string or None",
        "max(values, key=len) if values else None",
        "max(values) if values else None",
        ["{fn}(['zz', 'alphabet', 'tree']) == 'alphabet'", "{fn}([]) is None"],
    ),
    (
        "set_minus",
        "left, right",
        "returns members in the left set only",
        "left - right",
        "right - left",
        ["{fn}({1, 2, 3}, {2, 4}) == {1, 3}", "{fn}(set(), {1}) == set()"],
    ),
    (
        "has_suffix",
        "text, suffix",
        "checks whether text ends with a suffix",
        "text.endswith(suffix)",
        "text.startswith(suffix)",
        ["{fn}('report.py', '.py')", "not {fn}('python', 'py')"],
    ),
    (
        "largest_first",
        "values",
        "returns a descending sorted copy",
        "sorted(values, reverse=True)",
        "sorted(values)",
        ["{fn}([2, 1, 3]) == [3, 2, 1]", "{fn}([]) == []"],
    ),
    (
        "second_item",
        "values",
        "returns the second item",
        "values[1]",
        "values(1)",
        ["{fn}([4, 9, 2]) == 9", "{fn}(('a', 'b')) == 'b'"],
    ),
    (
        "clean_strings",
        "values",
        "strips strings and removes empty results",
        "[value.strip() for value in values if value.strip()]",
        "[value for value in values if value]",
        ["{fn}([' a ', ' ', 'b']) == ['a', 'b']", "{fn}([]) == []"],
    ),
    (
        "ascii_digits",
        "text",
        "checks whether text contains only ASCII digits and is non-empty",
        "bool(text) and all('0' <= character <= '9' for character in text)",
        "text.isdigit()",
        ["{fn}('2048')", "not {fn}('²')", "not {fn}('')"],
    ),
    (
        "minimum_or_none",
        "values",
        "returns the minimum value or None",
        "min(values) if values else None",
        "max(values) if values else None",
        ["{fn}([8, 2, 5]) == 2", "{fn}([]) is None"],
    ),
    (
        "all_but_first",
        "values",
        "returns all items except the first",
        "values[1:]",
        "values[:-1]",
        ["{fn}([1, 2, 3]) == [2, 3]", "{fn}([]) == []"],
    ),
    (
        "inclusive_range",
        "value, low, high",
        "checks inclusive numeric bounds",
        "low <= value <= high",
        "low < value < high",
        ["{fn}(3, 1, 3)", "not {fn}(4, 1, 3)"],
    ),
    (
        "multiply_values",
        "values",
        "multiplies all values with an empty product of one",
        "__import__('math').prod(values)",
        "sum(values)",
        ["{fn}([2, 3, 4]) == 24", "{fn}([]) == 1"],
    ),
    (
        "url_slug",
        "text",
        "lowercases words and joins them with hyphens",
        "'-'.join(text.lower().split())",
        "text.lower().replace(' ', '-')",
        ["{fn}('  Hello   World ') == 'hello-world'", "{fn}('One') == 'one'"],
    ),
    (
        "number_sign",
        "value",
        "returns negative one, zero, or positive one",
        "(value > 0) - (value < 0)",
        "1 if value >= 0 else -1",
        ["{fn}(9) == 1", "{fn}(-2) == -1", "{fn}(0) == 0"],
    ),
    (
        "parity_counts",
        "values",
        "returns counts of even and odd integers",
        "(sum(value % 2 == 0 for value in values), sum(value % 2 != 0 for value in values))",
        "(sum(value % 2 != 0 for value in values), sum(value % 2 == 0 for value in values))",
        ["{fn}([1, 2, 4]) == (2, 1)", "{fn}([]) == (0, 0)"],
    ),
    (
        "keys_for_value",
        "mapping, expected",
        "returns sorted keys having a value",
        "sorted(key for key, value in mapping.items() if value == expected)",
        "sorted(value for key, value in mapping.items() if value == expected)",
        ["{fn}({'a': 1, 'b': 2, 'c': 1}, 1) == ['a', 'c']", "{fn}({}, 1) == []"],
    ),
    (
        "flip_mapping",
        "mapping",
        "swaps keys and values",
        "{value: key for key, value in mapping.items()}",
        "{key: value for key, value in mapping.items()}",
        ["{fn}({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}", "{fn}({}) == {}"],
    ),
    (
        "largest_magnitude",
        "values",
        "returns the value with largest absolute magnitude or None",
        "max(values, key=abs) if values else None",
        "max(values) if values else None",
        ["{fn}([-9, 4, 8]) == -9", "{fn}([]) is None"],
    ),
    (
        "closest_to",
        "values, target",
        "returns the value closest to a target or None",
        "min(values, key=lambda value: (abs(value - target), value)) if values else None",
        "min(values) if values else None",
        ["{fn}([1, 4, 8], 6) == 4", "{fn}([], 2) is None"],
    ),
    (
        "substring_total",
        "text, fragment",
        "counts non-overlapping substring occurrences",
        "text.count(fragment)",
        "text.count(fragment) + 1",
        ["{fn}('banana', 'na') == 2", "{fn}('abc', 'x') == 0"],
    ),
    (
        "mask_address",
        "email",
        "masks an email username except its first character",
        "email.split('@', 1)[0][:1] + '*' * max(0, len(email.split('@', 1)[0]) - 1) + '@' + email.split('@', 1)[1]",
        "email",
        [
            "{fn}('ada@example.com') == 'a**@example.com'",
            "{fn}('x@y.com') == 'x@y.com'",
        ],
    ),
    (
        "trim_prefix",
        "text, prefix",
        "removes one prefix when present",
        "text[len(prefix):] if text.startswith(prefix) else text",
        "text.replace(prefix, '')",
        ["{fn}('foobarfoo', 'foo') == 'barfoo'", "{fn}('bar', 'foo') == 'bar'"],
    ),
]

VERIFY_TESTS = """\
import os
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(__file__).parent
environment = {{**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}}
correct = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "test_generated.py"],
    cwd=root,
    env=environment,
).returncode
(root / "solution.py").write_text({mutant!r})
shutil.rmtree(root / "__pycache__", ignore_errors=True)
mutant = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "test_generated.py"],
    cwd=root,
    env=environment,
).returncode
raise SystemExit(0 if correct == 0 and mutant != 0 else 1)
"""


def function_source(name, arguments, expression):
    return f"def {name}({arguments}):\n    return {expression}"


def tests_source(name, checks):
    assertions = "\n".join(
        f"    assert {check.replace('{fn}', name)}" for check in checks
    )
    return f"from solution import {name}\n\n\ndef test_{name}():\n{assertions}"


def execution_for_function(name, checks):
    return {
        "files": {"test_solution.py": tests_source(name, checks)},
        "command": ["python", "-m", "pytest", "-q"],
        "timeout_seconds": 10,
    }


def execution_for_tests(correct, mutant):
    return {
        "files": {
            "solution.py": correct + "\n",
            "verify_tests.py": VERIFY_TESTS.format(mutant=mutant + "\n"),
        },
        "command": ["python", "verify_tests.py"],
        "timeout_seconds": 10,
    }


def build():
    rows = []
    for index, (stem, arguments, description, correct, mutant, checks) in enumerate(
        SPECS, 1
    ):
        generation_name = f"generated_{stem}"
        generation_target = function_source(generation_name, arguments, correct)
        rows.append(
            {
                "id": f"benchmark-python-{index:03}",
                "group": stem,
                "task_type": "python_generation",
                "skills": ["python_skill"],
                "prompt": f"Write a Python function named {generation_name} that {description}. Return code only.",
                "target": generation_target,
                "execution": execution_for_function(generation_name, checks),
            }
        )

        debugging_name = f"repair_{stem}"
        debugging_target = function_source(debugging_name, arguments, correct)
        buggy = function_source(debugging_name, arguments, mutant)
        rows.append(
            {
                "id": f"benchmark-debug-{index:03}",
                "group": stem,
                "task_type": "debugging",
                "skills": ["python_skill", "debugging_skill"],
                "prompt": f"Fix this Python function so it {description}. Return code only.\n\n{buggy}",
                "target": debugging_target,
                "execution": execution_for_function(debugging_name, checks),
            }
        )

        tested_name = f"subject_{stem}"
        correct_source = function_source(tested_name, arguments, correct)
        mutant_source = function_source(tested_name, arguments, mutant)
        rows.append(
            {
                "id": f"benchmark-test-{index:03}",
                "group": stem,
                "task_type": "test_generation",
                "skills": ["python_skill", "test_generation_skill"],
                "prompt": f"Generate pytest tests for `{tested_name}({arguments})`, which {description}. Import it from solution. Return tests only.",
                "target": tests_source(tested_name, checks),
                "execution": execution_for_tests(correct_source, mutant_source),
            }
        )
    return sorted(rows, key=lambda row: (row["task_type"], row["id"]))


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in build())
    )


if __name__ == "__main__":
    main()
