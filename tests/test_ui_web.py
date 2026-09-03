"""GUI reset hook (Issue 2) - module-level bind/bound, no flask required.

``har.ui.web`` only imports flask lazily inside ``create_app``, so the
``bind_reset`` / ``_bound_reset`` add-on used by the manual Restart button can
be exercised in a bare interpreter.
"""

import unittest

from har.ui import web


class ResetHookTests(unittest.TestCase):
    def tearDown(self) -> None:
        web.bind_reset(None)

    def test_bind_reset_stores_and_returns_the_handler(self):
        captured = []

        def handler():
            captured.append("hit")
            return {"ok": True}

        web.bind_reset(handler)
        self.assertIs(handler, web._bound_reset())
        result = web._bound_reset()()
        self.assertEqual({"ok": True}, result)
        self.assertEqual(["hit"], captured)

    def test_unbound_reset_is_none(self):
        web.bind_reset(None)
        self.assertIsNone(web._bound_reset())

    def test_reset_hooks_are_independent_per_binding(self):
        first = lambda: {"ok": True, "who": "first"}  # noqa: E731
        web.bind_reset(first)
        self.assertIs(first, web._bound_reset())
        web.bind_reset(None)
        self.assertIsNone(web._bound_reset())


if __name__ == "__main__":
    unittest.main()
