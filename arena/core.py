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
    return (value // align) * align


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
        need = size
        prev, node = self._search(need, align, strategy)
        if node is None:
            return None
        return self._take(prev, node, need, align, size)

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
        end = pointer + len(data)
        if end > self.capacity:
            raise ArenaError("write past the end of the arena at %d" % (pointer,))
        self._buffer[pointer:end] = data
        return len(data)

    def read(self, pointer, size):
        """Return size bytes of the allocation at pointer."""
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be an int")
        if size < 0:
            raise ValueError("size must not be negative")
        record = self._lookup(pointer)
        end = pointer + size
        if end > self.capacity:
            raise ArenaError("read past the end of the arena at %d" % (pointer,))
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
        pointer = _align_up(node.offset, align)
        if pointer < node.offset:
            pointer = node.offset
        return pointer

    def _search(self, need, align, strategy):
        prev = None
        node = self._head
        best = None
        best_prev = None
        while node is not None:
            if node.size >= need:
                if strategy == "first":
                    return prev, node
                if best is None or node.size < best.size:
                    best = node
                    best_prev = prev
            prev = node
            node = node.next
        return best_prev, best

    def _take(self, prev, node, need, align, requested):
        block_start = node.offset
        block_size = node.size
        pointer = self._pointer_for(node, align)
        pad = pointer - block_start
        tail = block_size - pad - need
        front = pad if pad >= MIN_BLOCK else 0
        back = tail if tail > MIN_BLOCK else 0
        if prev is None:
            self._head = node.next
        else:
            prev.next = node.next
        if front:
            self._insert_free(pointer, front)
        if back:
            self._insert_free(pointer + need, back)
        start = block_start + front
        size = block_size - front - back
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
        merged = _FreeRange(offset, size, self._head)
        self._head = merged
        if prev is not None and prev.offset + prev.size == offset:
            prev.size += size
            return
        if merged.next is not None and merged.offset + merged.size == merged.next.offset:
            merged.size += merged.next.size
            merged.next = merged.next.next
