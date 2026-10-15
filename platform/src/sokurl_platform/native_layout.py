"""Typed th123 1.10a fields read by the published th123_ai script interface."""

# Names match Lua globals. Offsets and signedness match ObjBase and Character.
OBJECT_FIELDS = {
    "x": (0xEC, "f"), "y": (0xF0, "f"), "xspeed": (0xF4, "f"),
    "yspeed": (0xF8, "f"), "dir": (0x104, "b"), "act": (0x13C, "H"),
    "act_block": (0x13E, "H"), "frame": (0x144, "I"),
    "hp": (0x184, "H"), "hitstop": (0x196, "H"),
}
FIGHTER_FIELDS = {
    "air": (0x49B, "b"), "rei": (0x49E, "H"), "rmax": (0x4A0, "H"),
    "rei_stop": (0x4A2, "H"), "timestop": (0x4A8, "h"),
    "correction": (0x4AD, "b"), "rate": (0x4B0, "f"),
    "combo": (0x4B4, "H"), "dam": (0x4B6, "H"), "limit": (0x4B8, "H"),
    "speed_power": (0x4D0, "f"), "attack_power": (0x530, "f"),
    "defense_power": (0x534, "f"), "win_count": (0x573, "B"),
}
# index: (character, offset, type); -1 means all playable characters.
SPECIAL_FIELDS = {
    0: (2, 0x890, "h"), 1: (2, 0x892, "h"), 2: (0, 0x8B4, "h"),
    3: (5, 0x8D6, "h"), 4: (-1, 0x528, "h"), 5: (1, 0x892, "h"),
    6: (4, 0x890, "h"), 8: (0, 0x8B6, "h"), 9: (-1, 0x526, "h"),
    10: (14, 0x924, "h"), 11: (10, 0x898, "h"), 12: (10, 0x89A, "h"),
    15: (9, 0x892, "h"), 17: (-1, 0x844, "f"), 18: (-1, 0x840, "f"),
    19: (-1, 0x834, "h"), 20: (-1, 0x560, "h"), 21: (-1, 0x850, "h"),
    22: (-1, 0x852, "H"), 23: (-1, 0x84E, "H"), 24: (15, 0x89C, "i"),
    25: (15, 0x8A0, "i"), 26: (10, 0x8B2, "h"), 27: (4, 0x8A0, "h"),
}
MAX_OBJECTS = 1024
MAX_BOXES = 16
OBJECT_NAMES = (*OBJECT_FIELDS, "img", "fflags", "aflags", "address", "hitarea_n", "attackarea_n")
FIGHTER_NAMES = (*OBJECT_NAMES, *FIGHTER_FIELDS, "char", "spell", "card", "obj_n", "is_card_use")
WORLD_NAMES = ("frame", "battle_time", "weather", "weather2", "weather_time", "stage_number", "bgm_number")
OBJECT_WIDTH = len(OBJECT_NAMES) + MAX_BOXES * 8
FIGHTER_WIDTH = len(FIGHTER_NAMES) + MAX_BOXES * 8 + 10 + 16 + 28 + 10 + 20
PLAYER_WIDTH = FIGHTER_WIDTH + MAX_OBJECTS * OBJECT_WIDTH
RAW_WIDTH = len(WORLD_NAMES) + 2 * PLAYER_WIDTH
PRIVILEGED_FEATURES = RAW_WIDTH * 2
