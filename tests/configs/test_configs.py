#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pickle
import unittest

from basedet.configs import BaseConfig, ConfigDict


class ConfigTest(unittest.TestCase):

    def setUp(self):
        self.dict1 = {"MODEL": {"NAME": "RetinaNet", "FPN": True}}
        self.dict2 = {"MODEL": {"NAME": "FCOS"}}
        self.list1 = ["A.B.C", (1, 2), "A.B.D", 3]
        self.list2 = ["A.B.C", {"A": 1, "B": 2}, "A.B.D", 3]

    def tearDown(self):
        pass

    def test_generate_object(self):
        try:
            cfg = BaseConfig(self.dict1)
            cfg2 = BaseConfig(cfg)
            cfg2 = BaseConfig(self.dict1, TEST=True)
        except Exception:
            self.fail("raise Exception unexceptedly!")
        self.assertTrue(cfg2.TEST)

    def test_diff_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg2 = BaseConfig(self.dict2)

        diff1 = cfg1.diff(cfg2)
        diff2 = cfg2.diff(cfg1)
        self.assertNotEqual(diff1, diff2)

    def test_update_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.update(self.dict2)
        self.assertEqual(cfg1, BaseConfig(["MODEL.NAME", "FCOS"]))

    def test_merge_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.merge(self.dict2)
        cfg1.merge(self.list1)
        updated_cfg = {"MODEL": {"FPN": True, "NAME": "FCOS"}, "A": {"B": {"C": (1, 2), "D": 3}}}
        self.assertEqual(cfg1, BaseConfig(updated_cfg))

    def test_find_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.merge(self.list1)
        find_res = cfg1.find("B", show=False)
        self.assertEqual(find_res, ConfigDict(self.list1))

    def test_uion_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.merge(self.list1)
        cfg2 = BaseConfig(self.dict1)
        cfg2.MODEL.FPN = False
        un = cfg1.union(cfg2)
        self.assertEqual(un, ConfigDict(["MODEL.NAME", "RetinaNet"]))

    def test_remove_func(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.merge(self.list1)
        cfg2 = BaseConfig(self.dict1)
        cfg1.remove(cfg2)
        self.assertEqual(cfg1, BaseConfig(self.list1))

    def test_pickle(self):
        cfg1 = BaseConfig(self.dict1)
        filename = "temp.pkl"
        with open(filename, "wb") as f:
            pickle.dump(cfg1, f)

        with open(filename, "rb") as f:
            cfg2 = pickle.load(f)
        os.remove(filename)
        x = cfg1.diff(cfg2)
        assert len(x.keys()) == 0

    def test_hash(self):
        cfg1 = BaseConfig(self.list1)
        cfg2 = BaseConfig(self.list1)
        cfg3 = BaseConfig(self.list2)
        assert hash(cfg1) == hash(cfg2)
        assert hash(cfg1) != hash(cfg3)

    def test_yaml_load_and_save(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        x = BaseConfig(values_or_file=os.path.join(dirname, "test.yaml"))
        temp_file = os.path.join(dirname, "temp.yaml")
        x.dump_to_file(temp_file)
        y = BaseConfig(temp_file)
        assert len(x.diff(y).keys()) == 0
        os.remove(temp_file)

    def test_link_log(self):
        cfg1 = BaseConfig(self.dict1)
        cfg1.GLOBAL = dict(OUTPUT_DIR="/tmp")
        cfg1.link_log_dir()
        cfg1.link_log_dir()  # ensure second call link works


if __name__ == '__main__':
    unittest.main()
