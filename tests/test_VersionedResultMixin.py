import unittest
import datetime

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult, VersionedResultMixin

class TestVersionedResultMixin(unittest.TestCase):
    def test_parsing_version_string(self):
        mixer = VersionedResultMixin()
        version_str = "2023.12.23_1"
        parsed = mixer._VersionedResultMixin_parse_result_version_string(version_str)
        self.assertEqual(parsed, (datetime.datetime(2023, 12, 23), 1))

    def test_compare_result_version_strings(self):
        mixer = VersionedResultMixin()
        version0 = "2023.12.23_1"
        version1 = "2024.11.22_2"
        is_earlier = mixer._VersionedResultMixin_compare_result_version_strings(version0, version1)
        self.assertEqual(is_earlier, True)

    def test_earlier_than(self):
        mixer = VersionedResultMixin()
        # Set result version
        mixer.result_version = "2023.12.23_1"
        version1_str = "2024.11.22_2"
        is_earlier = mixer.is_result_version_earlier_than(version1_str)
        self.assertEqual(is_earlier, True)
        
    def test_newer_than(self):
        mixer = VersionedResultMixin()
        # Set result version
        mixer.result_version = "2024.11.22_2"
        version1_str = "2023.12.23_1"
        is_newer = mixer.is_result_version_newer_than(version1_str)
        self.assertEqual(is_newer, True)
        
    def test_equals(self):
        mixer = VersionedResultMixin()
        # Set result version
        mixer.result_version = "2024.11.22_2"
        version1_str = "2024.11.22_2"
        is_equal = mixer.is_result_version_equal_to(version1_str)
        self.assertEqual(is_equal, True)

if __name__ == '__main__':
    unittest.main()