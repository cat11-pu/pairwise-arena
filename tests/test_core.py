"""Behaviour tests for the arena allocator."""

import unittest

from arena.core import Arena
from arena.core import ArenaError


def span(pointer, size):
    """The half open byte range of one allocation."""
    return (pointer, pointer + size)


class ArenaTestCase(unittest.TestCase):
    def assertDisjoint(self, ranges):
        """Fail when two of the live ranges share a byte."""
        for index, first in enumerate(ranges):
            for second in ranges[index + 1:]:
                overlap = first[0] < second[1] and second[0] < first[1]
                self.assertFalse(overlap, "ranges %r and %r overlap" % (first, second))


class TestAllocation(ArenaTestCase):
    def test_alloc_returns_aligned_addresses_without_overlap(self):
        arena = Arena(128)
        small = arena.alloc(1)
        self.assertEqual(small, 0)
        aligned = arena.alloc(50, align=64)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned % 64, 0)
        filler = arena.alloc(60)
        self.assertIsNotNone(filler)
        self.assertDisjoint([span(small, 1), span(aligned, 50), span(filler, 60)])

    def test_aligned_request_that_does_not_fit_is_refused(self):
        arena = Arena(256)
        head = arena.alloc(48)
        middle = arena.alloc(24)
        tail = arena.alloc(184)
        self.assertEqual([head, middle, tail], [0, 48, 72])
        self.assertTrue(arena.free(middle))
        self.assertIsNone(arena.alloc(16, align=64))
        self.assertEqual(arena.stats()["free_bytes"], 24)
        self.assertDisjoint([span(head, 48), span(tail, 184)])

    def test_zero_length_requests_are_distinct_allocations(self):
        arena = Arena(64)
        first = arena.alloc(0)
        second = arena.alloc(0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertTrue(arena.free(first))
        self.assertTrue(arena.free(second))
        self.assertEqual(arena.stats()["used_blocks"], 0)
        self.assertEqual(arena.stats()["free_bytes"], 64)


class TestDataAccess(ArenaTestCase):
    def test_read_and_write_stay_inside_the_allocation(self):
        arena = Arena(512)
        head = arena.alloc(100)
        middle = arena.alloc(150)
        tail = arena.alloc(100)
        self.assertEqual([head, middle, tail], [0, 100, 250])
        arena.write(middle, b"m" * 150)
        arena.write(tail, b"t" * 100)
        with self.assertRaises(ArenaError):
            arena.write(middle, b"m" * 151)
        with self.assertRaises(ArenaError):
            arena.read(middle, 151)
        self.assertEqual(arena.read(middle, 150), b"m" * 150)
        self.assertEqual(arena.read(tail, 100), b"t" * 100)


class TestPlacement(ArenaTestCase):
    def test_strategies_pick_the_expected_free_blocks(self):
        arena = Arena(256)
        big = arena.alloc(128)
        arena.alloc(16)
        arena.alloc(48)
        last = arena.alloc(64)
        self.assertEqual([big, last], [0, 192])
        self.assertTrue(arena.free(big))
        self.assertTrue(arena.free(last))
        self.assertEqual(arena.alloc(24, strategy="first"), 0)
        self.assertEqual(arena.alloc(24, strategy="best"), 192)


class TestReclaiming(ArenaTestCase):
    def test_a_block_between_two_free_blocks_joins_both(self):
        arena = Arena(256)
        first = arena.alloc(32)
        middle = arena.alloc(32)
        last = arena.alloc(32)
        arena.alloc(160)
        self.assertEqual([first, middle, last], [0, 32, 64])
        self.assertTrue(arena.free(first))
        self.assertTrue(arena.free(last))
        self.assertTrue(arena.free(middle))
        self.assertEqual(arena.stats()["free_blocks"], 1)
        self.assertEqual(arena.stats()["largest_free"], 96)
        self.assertEqual(arena.alloc(96), 0)

    def test_freed_run_is_reused_by_a_larger_request(self):
        arena = Arena(256)
        first = arena.alloc(64)
        second = arena.alloc(64)
        third = arena.alloc(128)
        self.assertEqual([first, second, third], [0, 64, 128])
        arena.write(first, b"a" * 64)
        arena.write(third, b"c" * 128)
        self.assertTrue(arena.free(first))
        self.assertTrue(arena.free(second))
        reused = arena.alloc(96)
        self.assertEqual(reused, 0)
        self.assertDisjoint([span(reused, 96), span(third, 128)])
        self.assertEqual(arena.read(third, 128), b"c" * 128)
        self.assertEqual(
            arena.stats()["used_bytes"] + arena.stats()["free_bytes"],
            arena.capacity,
        )

    def test_repeated_cycles_return_every_byte(self):
        arena = Arena(256)
        for _ in range(12):
            pointers = [arena.alloc(24) for _ in range(4)]
            for pointer in pointers:
                self.assertIsNotNone(pointer)
            self.assertEqual(arena.stats()["used_blocks"], 4)
            for pointer in pointers:
                self.assertTrue(arena.free(pointer))
            self.assertEqual(arena.stats()["free_blocks"], 1)
            self.assertEqual(arena.stats()["largest_free"], 256)
            self.assertEqual(arena.stats()["free_bytes"], 256)


class TestAccounting(ArenaTestCase):
    def test_every_byte_stays_accounted_and_bad_calls_are_rejected(self):
        arena = Arena(256)
        empty = arena.alloc(0)
        first = arena.alloc(24)
        second = arena.alloc(24)
        self.assertEqual(arena.stats()["used_blocks"], 3)
        self.assertEqual(
            arena.stats()["used_bytes"] + arena.stats()["free_bytes"],
            arena.capacity,
        )
        self.assertTrue(arena.free(first))
        self.assertEqual(arena.stats()["used_blocks"], 2)
        self.assertEqual(
            arena.stats()["used_bytes"] + arena.stats()["free_bytes"],
            arena.capacity,
        )
        with self.assertRaises(ArenaError):
            arena.free(first)
        with self.assertRaises(ArenaError):
            arena.free(200)
        with self.assertRaises(ValueError):
            arena.alloc(-8)
        with self.assertRaises(ValueError):
            arena.alloc(16, align=6)
        with self.assertRaises(TypeError):
            arena.alloc("16")
        self.assertTrue(arena.free(empty))
        self.assertTrue(arena.free(second))
        self.assertEqual(arena.stats()["used_blocks"], 0)
        self.assertEqual(arena.stats()["free_blocks"], 1)
        self.assertEqual(arena.stats()["free_bytes"], 256)
        self.assertEqual(arena.stats()["largest_free"], 256)
        huge = arena.alloc(240)
        self.assertEqual(huge, 0)
        small = arena.alloc(16)
        self.assertIsNotNone(small)
        self.assertDisjoint([span(huge, 240), span(small, 16)])
        self.assertEqual(arena.stats()["free_bytes"], 0)
        self.assertEqual(
            arena.stats()["used_bytes"] + arena.stats()["free_bytes"],
            arena.capacity,
        )
