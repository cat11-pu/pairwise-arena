"""Arena: a small byte range allocator built on one fixed buffer.

Callers work with integer offsets instead of objects.  The arena keeps every
live allocation in a ledger and the ranges that can still be handed out in a
free list.
"""

MIN_BLOCK = 16
STRATEGIES = ("first", "best")


class ArenaError(Exception):
    """Raised when an operation cannot be carried out on the arena."""


def _align_up(value, align):
    """Round value up to the next multiple of align."""
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


def _check_size(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an int")
    if size < 0:
        raise ValueError("size must not be negative")
    return size


def _check_align(align):
    if isinstance(align, bool) or not isinstance(align, int):
        raise TypeError("align must be an int")
    if align < 1 or align & (align - 1):
        raise ValueError("align must be a power of two")
    return align


class _FreeRange:
    __slots__ = ("offset", "size", "next")

    def __init__(self, offset, size, nxt=None):
        self.offset = offset
        self.size = size
        self.next = nxt


class _LiveRange:
    __slots__ = ("pointer", "start", "size", "usable")

    def __init__(self, pointer, start, size, usable):
        self.pointer = pointer
        self.start = start
        self.size = size
        self.usable = usable


class Arena:
    """A fixed size arena served by a first fit or best fit free list."""

    def __init__(self, capacity, strategy="first"):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if strategy not in STRATEGIES:
            raise ValueError("unknown strategy: %r" % (strategy,))
        self.capacity = capacity
        self.strategy = strategy
        self._buffer = bytearray(capacity)
        self._live = {}
        self._head = _FreeRange(0, capacity)

    def alloc(self, size, align=1, strategy=None):
        """Hand out a range of size bytes that starts on an align boundary."""
        size = _check_size(size)
        align = _check_align(align)
        if strategy is None:
            strategy = self.strategy
        if strategy not in STRATEGIES:
            raise ValueError("unknown strategy: %r" % (strategy,))
        # A zero length request still needs a unique anchor byte so that two
        # such allocations never share an address.
        span = size if size > 0 else 1
        prev, node = self._search(span, align, strategy)
        if node is None:
            return None
        return self._take(prev, node, span, align, size)

    def free(self, pointer):
        """Release an address that alloc returned."""
        record = self._lookup(pointer)
        del self._live[pointer]
        self._insert_free(record.start, record.size)
        return True

    def write(self, pointer, data):
        """Copy data into the allocation at pointer."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        record = self._lookup(pointer)
        if len(data) > record.usable:
            raise ArenaError("write past the end of the allocation at %d" % (pointer,))
        end = pointer + len(data)
        self._buffer[pointer:end] = data
        return len(data)

    def read(self, pointer, size):
        """Return size bytes of the allocation at pointer."""
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be an int")
        if size < 0:
            raise ValueError("size must not be negative")
        record = self._lookup(pointer)
        if size > record.usable:
            raise ArenaError("read past the end of the allocation at %d" % (pointer,))
        end = pointer + size
        return bytes(self._buffer[pointer:end])

    def stats(self):
        """The current arena accounting."""
        free_bytes = 0
        free_blocks = 0
        largest = 0
        node = self._head
        while node is not None:
            free_bytes += node.size
            free_blocks += 1
            if node.size > largest:
                largest = node.size
            node = node.next
        used_bytes = 0
        for record in self._live.values():
            used_bytes += record.size
        return {
            "capacity": self.capacity,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,
            "used_blocks": len(self._live),
            "free_blocks": free_blocks,
            "largest_free": largest,
        }

    # -- internals -----------------------------------------------------
    def _lookup(self, pointer):
        if isinstance(pointer, bool) or not isinstance(pointer, int):
            raise TypeError("pointer must be an int")
        record = self._live.get(pointer)
        if record is None:
            raise ArenaError("no live allocation at %d" % (pointer,))
        return record

    def _pointer_for(self, node, align):
        """The first offset inside node that satisfies align."""
        return _align_up(node.offset, align)

    def _fits(self, node, span, align):
        """The aligned start, padding and fit status for one free node."""
        pointer = self._pointer_for(node, align)
        pad = pointer - node.offset
        if pad + span > node.size:
            return None
        return pointer, pad

    def _search(self, need, align, strategy):
        prev = None
        node = self._head
        best = None
        best_prev = None
        best_waste = None
        while node is not None:
            fit = self._fits(node, need, align)
            if fit is not None:
                pointer, pad = fit
                if strategy == "first":
                    return prev, node
                waste = node.size - pad - need
                if best is None or waste < best_waste:
                    best = node
                    best_prev = prev
                    best_waste = waste
            prev = node
            node = node.next
        return best_prev, best

    def _take(self, prev, node, span, align, requested):
        block_start = node.offset
        block_size = node.size
        pointer, pad = self._fits(node, span, align)
        if prev is None:
            self._head = node.next
        else:
            prev.next = node.next
        tail = block_size - pad - span
        if pad >= MIN_BLOCK:
            self._insert_free(block_start, pad)
            start = pointer
        else:
            start = block_start
        if tail >= MIN_BLOCK:
            self._insert_free(pointer + span, tail)
            end = pointer + span
        else:
            end = block_start + block_size
        size = end - start
        self._live[pointer] = _LiveRange(pointer, start, size, requested)
        return pointer

    def _locate(self, offset):
        """The free ranges around offset: the closest below, the first above."""
        prev = None
        node = self._head
        while node is not None and node.offset < offset:
            prev = node
            node = node.next
        return prev, node

    def _insert_free(self, offset, size):
        """Return the range to the free list, joining it to its neighbours."""
        prev, node = self._locate(offset)
        if prev is not None and prev.offset + prev.size == offset:
            prev.size += size
            if node is not None and prev.offset + prev.size == node.offset:
                prev.size += node.size
                prev.next = node.next
            return
        merged = _FreeRange(offset, size, node)
        if prev is None:
            self._head = merged
        else:
            prev.next = merged
        if node is not None and merged.offset + merged.size == node.offset:
            merged.size += node.size
            merged.next = node.next
