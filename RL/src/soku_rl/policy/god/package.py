"""Resolve original script bytes and fingerprint the complete external package."""
import hashlib
from pathlib import Path


CHARACTERS = (
    "reimu", "marisa", "sakuya", "alice", "patchouli", "youmu", "remilia", "yuyuko",
    "yukari", "suica", "udonge", "aya", "komachi", "iku", "tenshi", "sanae", "chirno",
    "meirin", "utsuho", "suwako",
)
# Complete missing definitions only for the exact published variant. Distance
# bounds reproduce the six corresponding branches in 01_marisa_main.ai lines
# 2231-2265. Both A entries have the same interval because the variant ORs them.
# dis is an integer, so >300, >600, >400 become >=301, >=601, >=401.
# These are documented repairs, not recovered definitions from the variant.
REPAIRS = {
    "01_marisa_main_新版厨远A.ai": (
        "33bd3d0faf57a2a96f2b3806dbf9f4abf69e1748445cdf629991899f2fa9ec4c",
        b"wait_frame = 1;\n"
        b"pogi_line = {[300]=160,[301]=160,[305]=300,[402]=300,[400]=500,[411]=math.huge,[410]=math.huge};\n"
        b"nega_line = {[300]=0,[301]=0,[305]=200,[402]=150,[400]=301,[411]=601,[410]=401};\n",
    ),
}


class ScriptPackage:
    def __init__(self, directory, api_source):
        self.root = Path(directory).resolve(strict=True)
        self.api = Path(api_source).read_bytes()
        self.files, self.aliases = {}, {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            self.files[relative] = path.read_bytes()
            for encoding in ("utf-8", "cp932", "gbk"):
                try:
                    alias = relative.encode(encoding)
                except UnicodeEncodeError:
                    continue
                if alias in self.aliases and self.aliases[alias] != relative:
                    raise ValueError("ambiguous original script filename encoding")
                self.aliases[alias] = relative
        if not self.files or b"function api_main" not in self.api:
            raise ValueError("the complete script directory and original api.ai are required")
        digest = hashlib.sha256(self.api)
        for name, data in self.files.items():
            digest.update(name.encode("utf-8") + b"\0" + data)
        self.original_fingerprint = digest.hexdigest()
        self.repairs = {}
        for name, (expected, definition) in REPAIRS.items():
            if name in self.files and hashlib.sha256(self.files[name]).hexdigest() == expected:
                self.repairs[name] = definition
                digest.update(name.encode("utf-8") + b"\0" + definition)
        self.fingerprint = digest.hexdigest()

    def resolve_name(self, name):
        if isinstance(name, str):
            name = name.encode("utf-8")
        name = name.replace(b"\\", b"/")
        if name not in self.aliases:
            raise FileNotFoundError(f"script is absent from the source package: {name!r}")
        return name

    def source(self, name):
        relative = self.aliases[self.resolve_name(name)]
        if relative in self.repairs:
            return self.repairs[relative] + self.files[relative]
        return self.files[relative]

    def character_script(self, character):
        if type(character) is not int or not 0 <= character < len(CHARACTERS):
            raise ValueError("character must be an integer from zero to nineteen")
        name = f"{character:02}_{CHARACTERS[character]}_main.ai"
        self.source(name)
        return name
