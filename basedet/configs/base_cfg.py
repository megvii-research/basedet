#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import getpass
import os

from basecore.config import ConfigDict

from basedet.utils import ensure_dir


class BaseConfig(ConfigDict):
    # TODO: check user in config, change from user to USER
    user = getpass.getuser()

    def link_log_dir(self, link_name="log"):
        """
        create soft link to output dir.

        Args:
            link_name (str): name of soft link.
        """
        output_dir = self.get("output_dir", None)
        if not output_dir:
            output_dir = self.GLOBAL.OUTPUT_DIR

        if not output_dir:
            raise ValueError("output dir is not specified")
        ensure_dir(output_dir)

        if os.path.islink(link_name) and os.readlink(link_name) != output_dir:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(output_dir, link_name)
            os.system(cmd)
