#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-check the board tokenizer against the existing C++ implementation."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from clip_tokenizer import (
    BOS_TOKEN_ID,
    CONTEXT_LENGTH,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    CLIPTokenizer,
    tokenize_prompts,
)
from extract_clip_merges import extract_bytes, validate_merges


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SOURCE_CPP_TOKENIZER_DIR = REPO_ROOT / "examples/yolo_world/cpp/tokenizer"
BUNDLED_CPP_TOKENIZER_DIR = SCRIPT_DIR.parent / "tests/tokenizer"
CPP_TOKENIZER_DIR = (
    SOURCE_CPP_TOKENIZER_DIR
    if SOURCE_CPP_TOKENIZER_DIR.is_dir()
    else BUNDLED_CPP_TOKENIZER_DIR
)
MERGES_PATH = SCRIPT_DIR / "bpe_simple_vocab_16e6.txt"
HEADER_PATH = CPP_TOKENIZER_DIR / "clip_vocab.h"
EXPECTED_ASSET_SHA256 = (
    "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
)
PROMPTS = ("cup", "cat", "bottle", "person", "红色杯子")
CPP_SAFE_PROMPTS = PROMPTS[:4]

# Token-for-token reference generated with transformers 4.55.2
# CLIPTokenizerFast using the OpenAI CLIP vocab derived from this merges file.
EXPECTED_ROWS = {
    "cup": [49406, 1937, *([49407] * 18)],
    "cat": [49406, 2368, *([49407] * 18)],
    "bottle": [49406, 5392, *([49407] * 18)],
    "person": [49406, 2533, *([49407] * 18)],
    "红色杯子": [
        49406,
        163,
        118,
        95,
        164,
        231,
        110,
        27667,
        107,
        35751,
        494,
        *([49407] * 9),
    ],
}


class CLIPTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = CLIPTokenizer(MERGES_PATH)
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="clip-tokenizer-test-")
        cls.reference_binary = Path(cls.temp_dir.name) / "cpp_tokenizer_reference"
        compile_command = [
            "g++",
            "-std=c++17",
            "-O0",
            "-I",
            str(CPP_TOKENIZER_DIR),
            str(SCRIPT_DIR / "cpp_tokenizer_reference.cc"),
            str(CPP_TOKENIZER_DIR / "clip_tokenizer.cpp"),
            "-o",
            str(cls.reference_binary),
        ]
        subprocess.run(compile_command, check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_generated_asset_matches_embedded_cpp_bytes(self) -> None:
        embedded = extract_bytes(HEADER_PATH)
        validate_merges(embedded)
        generated = MERGES_PATH.read_bytes()
        self.assertEqual(generated, embedded)
        self.assertEqual(hashlib.sha256(generated).hexdigest(), EXPECTED_ASSET_SHA256)

    def test_special_token_ids_and_context_contract(self) -> None:
        self.assertEqual(self.tokenizer.encoder["<|startoftext|>"], BOS_TOKEN_ID)
        self.assertEqual(self.tokenizer.encoder["<|endoftext|>"], EOS_TOKEN_ID)
        for prompt in PROMPTS:
            row = self.tokenizer.tokenize(prompt)
            self.assertEqual(len(row), CONTEXT_LENGTH)
            self.assertEqual(row[0], BOS_TOKEN_ID)
            self.assertEqual(row[-1], PAD_TOKEN_ID)
            self.assertIn(EOS_TOKEN_ID, row[1:])

    def test_required_prompts_match_pinned_transformers_rows(self) -> None:
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.tokenizer.tokenize(prompt),
                    EXPECTED_ROWS[prompt],
                )

    def test_ascii_prompts_match_cpp_tokenizer_token_for_token(self) -> None:
        result = subprocess.run(
            [str(self.reference_binary), *CPP_SAFE_PROMPTS],
            check=True,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
        )
        cpp_rows = [
            [int(token) for token in line.split(",")]
            for line in result.stdout.splitlines()
        ]
        python_rows = [self.tokenizer.tokenize(prompt) for prompt in CPP_SAFE_PROMPTS]
        self.assertEqual(len(cpp_rows), len(CPP_SAFE_PROMPTS))
        for prompt, python_row, cpp_row in zip(
            CPP_SAFE_PROMPTS,
            python_rows,
            cpp_rows,
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(python_row, cpp_row)

    def test_batch_api_has_no_numpy_requirement_by_default(self) -> None:
        rows = tokenize_prompts(PROMPTS, merges_path=MERGES_PATH)
        self.assertIsInstance(rows, list)
        self.assertTrue(all(isinstance(row, list) for row in rows))
        self.assertTrue(all(len(row) == CONTEXT_LENGTH for row in rows))

    def test_long_prompt_truncates_and_terminates(self) -> None:
        row = self.tokenizer.tokenize(" ".join(["person"] * 100))
        self.assertEqual(len(row), CONTEXT_LENGTH)
        self.assertEqual(row[0], BOS_TOKEN_ID)
        self.assertEqual(row[-1], EOS_TOKEN_ID)
        with self.assertRaises(ValueError):
            self.tokenizer.tokenize(
                " ".join(["person"] * 100),
                truncate=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
