import unittest
from bat_internship.site_selection import select_diverse_sites


def meta(site, bat, rois, stimuli=17, onset=1):
    return dict(site_index=site, bat_id=bat, site_number=site + 1,
                shape_time_repeat_stimulus_roi=[3, 2, stimuli, rois],
                onset_frame_index=onset, categories=["Social"] * stimuli)


class SelectionTests(unittest.TestCase):
    def test_animals_over_extra_sites_and_smallest_equivalent(self):
        rows = [meta(0, 1, 100), meta(1, 1, 10), meta(2, 2, 20)]
        self.assertEqual(select_diverse_sites(rows)["site_indices"], [1, 2])

    def test_protocol_diversity_before_size(self):
        rows = [meta(0, 1, 10, 12), meta(1, 2, 1, 12), meta(2, 2, 10, 17)]
        self.assertEqual(select_diverse_sites(rows)["site_indices"], [0, 2])

    def test_budget_and_incompatible_grid(self):
        rows = [meta(0, 1, 10), meta(1, 2, 20), meta(2, 3, 1, onset=2)]
        selection = select_diverse_sites(rows, response_budget_mib=0.01)
        self.assertEqual(selection["site_indices"], [2])
        self.assertLessEqual(selection["response_bytes"], 0.01 * 1024 ** 2)

    def test_deterministic_ties_and_max_sites(self):
        rows = [meta(2, 2, 10), meta(1, 1, 10), meta(0, 1, 10)]
        self.assertEqual(select_diverse_sites(rows, max_sites=1)["site_indices"], [0])

    def test_invalid_or_impossible(self):
        for kwargs in (dict(max_sites=0), dict(response_budget_mib=float("nan")),
                       dict(response_budget_mib=0.000001)):
            with self.assertRaises(ValueError):
                select_diverse_sites([meta(0, 1, 10)], **kwargs)
        with self.assertRaises(ValueError):
            select_diverse_sites([meta(0, 1, 10), meta(0, 1, 10)])
