"""La dependencia del entorno rechaza destinos que el navegador interpreta como otro host."""

import pytest

web = pytest.importorskip("tornado.web")
testing = pytest.importorskip("tornado.testing")


class TestRedirectBoundary(testing.AsyncHTTPTestCase):
    def get_app(self):
        return web.Application([(r"/redirect/(.*)", web.RedirectHandler, {"url": "{0}"})])

    def test_relative_destination_remains_usable(self):
        response = self.fetch("/redirect/local/path", follow_redirects=False)
        assert response.code == 301
        assert response.headers["Location"] == "local/path"

    def test_encoded_triple_slash_cannot_change_the_host(self):
        response = self.fetch("/redirect/%2F%2F%2Fexample.test/path", follow_redirects=False)
        assert response.code == 400
        assert "Location" not in response.headers
