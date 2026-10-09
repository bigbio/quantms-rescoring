"""Feature worker pools must not start children where Python forbids them."""

from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

import pytest

from quantmsrescore import utils


@pytest.mark.parametrize("daemon,parent", [(True, None), (False, object())])
def test_worker_pool_uses_one_thread_in_daemons_and_pool_workers(monkeypatch, daemon, parent):
    monkeypatch.setattr(
        utils.multiprocessing, "current_process", lambda: SimpleNamespace(daemon=daemon)
    )
    monkeypatch.setattr(utils.multiprocessing, "parent_process", lambda: parent)

    with utils.worker_pool(4) as pool:
        assert isinstance(pool, ThreadPool)
        assert list(pool.imap(abs, [-1, 2])) == [1, 2]
