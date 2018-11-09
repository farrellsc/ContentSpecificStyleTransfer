from unittest import TestCase


class BaseTestCase(TestCase):
    def setUp(self):
        raise NotImplementedError
