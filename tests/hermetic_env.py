"""Imported first by the tests that touch execution code: removes every
PIASTQ_* variable (real backend name, token, budget limit, resume flag, jobs in
flight) so the unit tests behave identically on a laptop, in `run_all.py`'s
test stage and on the cluster, and can never reach the real service. Tests that
need such a variable set it explicitly with mock.patch.dict."""

import os

for _name in [n for n in os.environ if n.startswith("PIASTQ_")]:
    del os.environ[_name]
