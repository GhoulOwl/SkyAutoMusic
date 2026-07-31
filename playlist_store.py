import json
import os
import tempfile
from dataclasses import dataclass


def _normalize_filename(filename):
    if not isinstance(filename, str):
        return None
    filename = filename.strip()
    if (
        not filename
        or not filename.lower().endswith(".json")
        or os.path.basename(filename) != filename
    ):
        return None
    return filename


class PlaylistStore:
    """Ordered, de-duplicated playlist persisted as a JSON filename array."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.items = []

    def load(self, available_files=None):
        existed = os.path.exists(self.path)
        raw_items = []
        needs_rewrite = False
        if existed:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    raw_items = payload
                else:
                    needs_rewrite = True
            except (OSError, ValueError, TypeError):
                raw_items = []
                needs_rewrite = True

        available = None if available_files is None else set(available_files)
        cleaned = []
        seen = set()
        for value in raw_items:
            filename = _normalize_filename(value)
            if (
                filename is None
                or filename in seen
                or (available is not None and filename not in available)
            ):
                continue
            seen.add(filename)
            cleaned.append(filename)

        changed = needs_rewrite or cleaned != raw_items
        self.items = cleaned
        if existed and changed:
            try:
                self.save()
            except OSError:
                # Read-only install locations may prevent cleanup; the in-memory
                # playlist is still safe and usable for this run.
                pass
        return list(self.items)

    def save(self):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".playlist-",
            suffix=".tmp",
            dir=directory,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def add(self, filename):
        filename = _normalize_filename(filename)
        if filename is None or filename in self.items:
            return False
        previous = list(self.items)
        self.items.append(filename)
        try:
            self.save()
        except OSError:
            self.items = previous
            raise
        return True

    def remove(self, filename):
        if filename not in self.items:
            return False
        previous = list(self.items)
        self.items.remove(filename)
        try:
            self.save()
        except OSError:
            self.items = previous
            raise
        return True

    def move(self, filename, offset):
        if filename not in self.items:
            return False
        old_index = self.items.index(filename)
        new_index = old_index + int(offset)
        if new_index < 0 or new_index >= len(self.items):
            return False
        previous = list(self.items)
        self.items.insert(new_index, self.items.pop(old_index))
        try:
            self.save()
        except OSError:
            self.items = previous
            raise
        return True

    def refresh(self, available_files):
        available = set(available_files)
        cleaned = [item for item in self.items if item in available]
        if cleaned == self.items:
            return False
        previous = list(self.items)
        self.items = cleaned
        try:
            self.save()
        except OSError:
            self.items = previous
            raise
        return True


@dataclass
class PlaybackSession:
    """Stable source snapshot used by manual and automatic track changes."""

    items: tuple
    index: int
    mode: str
    auto_advance: bool
    source_tab: str

    @classmethod
    def create(cls, items, current, mode, auto_advance=False, source_tab=""):
        items = tuple(items)
        if current not in items:
            raise ValueError("Current track is not part of the playback sequence.")
        return cls(
            items=items,
            index=items.index(current),
            mode=mode,
            auto_advance=bool(auto_advance),
            source_tab=source_tab,
        )

    @property
    def current(self):
        return self.items[self.index]

    def candidate_indexes(self, offset):
        step = 1 if int(offset) > 0 else -1
        index = self.index + step
        while 0 <= index < len(self.items):
            yield index
            index += step

    def can_move(self, offset):
        target = self.index + (1 if int(offset) > 0 else -1)
        return 0 <= target < len(self.items)
