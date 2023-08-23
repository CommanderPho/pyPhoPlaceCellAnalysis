import unittest
from pathlib import Path
import pandas as pd
from attrs import define, field, Factory
import tables as tb
from typing import List, Dict

from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import H5FileReference, H5ExternalLinkBuilder

#TODO 2023-08-23 10:41: - [ ] Skeleton of tests written by ChatGPT, write tests


# Test for H5FileReference
class TestH5FileReference(unittest.TestCase):
    def test_init(self):
        short_name = "example"
        path = Path("/path/to/file")
        ref = H5FileReference(short_name=short_name, path=path)
        self.assertEqual(ref.short_name, short_name)
        self.assertEqual(ref.path, path)

# Test for H5ExternalLinkBuilder
class TestH5ExternalLinkBuilder(unittest.TestCase):
    def test_init(self):
        file_reference_list = [H5FileReference(short_name="example", path=Path("/path/to/file"))]
        table_key_list = ["table_key"]
        loader = H5ExternalLinkBuilder(file_reference_list=file_reference_list, table_key_list=table_key_list)
        self.assertEqual(loader.file_reference_list, file_reference_list)
        self.assertEqual(loader.table_key_list, table_key_list)

    def test_init_from_file_lists(self):
        file_list = [Path("/path/to/file1"), Path("/path/to/file2")]
        table_key_list = ["table_key1", "table_key2"]
        short_name_list = ["example1", "example2"]
        loader = H5ExternalLinkBuilder.init_from_file_lists(file_list, table_key_list, short_name_list)
        self.assertEqual(len(loader.file_reference_list), 2)
        self.assertEqual(loader.file_short_name, short_name_list)

    # Further tests can be added for other methods such as `load_and_consolidate` and `build_linking_results`
    # These may require mocking the interaction with .h5 files or using temporary files for testing

if __name__ == "__main__":
    unittest.main()
