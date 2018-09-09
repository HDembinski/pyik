import unittest
import re
import os
import doctest
import importlib
import sys

root_dir = os.path.join(os.path.dirname(__file__), "..")
module_paths = [root_dir + "/pyik/" + x
                for x in os.listdir(root_dir + "/pyik")
                if x.endswith(".py") and not x.endswith("__init__.py")]
assert module_paths

class TestReadme(unittest.TestCase):

    def test_readme(self):
        txt = open(root_dir + "/README.md").read()
        readme_modules = set(re.findall("\* \*([^*]+)\*:", txt))
        modules = {os.path.basename(x)[:-3] for x in module_paths}
        self.assertEqual(readme_modules, modules)

def load_tests(loader, tests, ignore):
    try:
        import nlopt
        have_nlopt = True
    except:
        have_nlopt = False

    try:
        import ROOT
        have_root = True
    except:
        have_root = False

    sys.path = [root_dir] + sys.path
    for path in module_paths:
        name = os.path.basename(path)[:-3]
        if name == "fit" and not have_nlopt:
            continue
        if name == "rootext" and not have_root:
            continue
        m = importlib.import_module("pyik." + name)
        suite = doctest.DocTestSuite(module=m)
        tests.addTests(suite)
    return tests

if __name__ == '__main__':
    unittest.main()
