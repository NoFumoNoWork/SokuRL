"""Implement the original script's read and key functions over one common observation."""
import math
import struct


def float_difference(left, right):
    """The original operands and their subtraction use C++ float precision."""
    return struct.unpack("<f", struct.pack("<f", left - right))[0]


class ScriptAPI:
    def __init__(self, lua, seed):
        self.lua, self.globals = lua, lua.globals()
        self.random_state = seed
        self.requested, self.applied = [0] * 10, [0] * 10
        self.events = []
        self.frame = 0
        self.data_delay = 0
        self.delayed_actions, self.delayed_blocks, self.delayed_frames = [], [], []
        self.delays = {b"key_delay": 0, b"data_delay": 0, b"weather_delay": 300}
        self.globals[b"key_delay"] = 0
        self.globals[b"data_delay"] = 0
        self.globals[b"weather_delay"] = 300
        self.read_delays = self.lua.eval(b"function() return key_delay,data_delay,weather_delay end")
        for name in ("key_on", "key_off", "key_reset", "get_key_stat", "get_key_map",
                     "get_skill_lv", "get_card_id", "get_card_cost", "get_card_cost2",
                     "get_obj_data", "get_obj_attackarea", "get_obj_hitarea", "get_attackarea",
                     "get_hitarea", "get_special_data", "get_fflags", "get_aflags",
                     "get_correction", "get_key_stat2", "get_key_map2", "get_opt_xy", "get_deck_list"):
            self.globals[name.encode()] = getattr(self, name)
        self.globals[b"_set_delay"] = self.set_delay
        self.lua.execute(b"""
          function set_key_delay(n) _set_delay('key_delay',n); key_delay=n end
          function set_data_delay(n) _set_delay('data_delay',n); data_delay=n end
          function set_weather_delay(n) _set_delay('weather_delay',n); weather_delay=n end
        """)
        self.globals[b"get_version"] = lambda: b"ver0.93"
        self.globals[b"get_real_key_stat"] = self.no_external_keys
        self.globals[b"_random_unit"] = self.random
        self.lua.execute(b"""
          function math.random(...)
            local r, n = _random_unit(), select('#', ...)
            local l, u = ...
            if n == 0 then return r end
            if n == 1 then u=l; l=1 end
            if n > 2 then error('wrong number of arguments') end
            if type(l) ~= 'number' or type(u) ~= 'number' then error('number expected') end
            l = l < 0 and math.ceil(l) or math.floor(l)
            u = u < 0 and math.ceil(u) or math.floor(u)
            if l > u then error('interval is empty') end
            return math.floor(r*(u-l+1))+l
          end
        """)
        self.globals[b"math"][b"randomseed"] = self.randomseed
        self.lua.execute(b"function getfunc(f) if type(f)=='string' then return _G[f] else return f end end")

    def no_external_keys(self, key):
        raise RuntimeError("external keyboard queries are not part of an offline game episode")

    def set_delay(self, key, value):
        if int(value) != value or not 0 <= value <= 10000:
            raise ValueError("script delay must be an integer from zero to 10000")
        self.delays[key] = int(value)

    def randomseed(self, seed):
        self.random_state = int(seed) & 0xFFFFFFFF

    def random(self, *limits):
        self.random_state = (214013 * self.random_state + 2531011) & 0xFFFFFFFF
        value = (((self.random_state >> 16) & 32767) % 32767) / 32767.
        if not limits:
            return value
        if len(limits) not in (1, 2):
            raise ValueError("random accepts zero, one or two bounds")
        low, high = (1, int(limits[0])) if len(limits) == 1 else tuple(map(int, limits))
        if low > high:
            raise ValueError("empty random interval")
        return math.floor(value * (high - low + 1)) + low

    def _change(self, key, state):
        if self.requested[key] != state:
            self.requested[key] = state
            self.events.append((self.frame + self.delays[b"key_delay"], key, state))

    def key_on(self, key):
        key = int(key)
        if not 0 <= key < 14:
            raise ValueError("invalid original key index")
        if key >= 10:
            for part in ((1, 2), (1, 3), (0, 2), (0, 3))[key - 10]:
                self.key_on(part)
        elif not self.requested[key]:
            if key < 4:
                self._change(key ^ 1, 0)
            self._change(key, 1)

    def key_off(self, key):
        key = int(key)
        if not 0 <= key < 14:
            raise ValueError("invalid original key index")
        if key >= 10:
            # Preserve key_manager.cpp's diagonal-release semantics.
            self.key_off(1); self.key_off(2)
        else:
            self._change(key, 0)
            if key < 4:
                self._change(key ^ 1, 0)

    def key_reset(self):
        for key in range(10):
            self.key_off(key)

    def get_key_stat(self, key):
        return self.requested[int(key)]

    def get_key_map(self):
        return tuple(self.requested)

    def inputs(self):
        pending = []
        for at, key, state in self.events:
            if at <= self.frame:
                self.applied[key] = state
            else:
                pending.append((at, key, state))
        self.events = pending
        keys = self.applied
        self.frame += 1
        return (keys[3] - keys[2], keys[1] - keys[0], *keys[4:])

    def player(self, player):
        if player not in (0, 1):
            raise ValueError("invalid original player index")
        return self.observation.players[int(player)]

    def get_skill_lv(self, player, index):
        return self.player(player)["skills"][int(index)] if 0 <= index <= 14 else -1

    def get_card_id(self, player, index):
        return self.player(player)["cards"][2 * int(index)] if 0 <= index < 5 else -1

    def get_card_cost(self, player, index):
        return self.player(player)["cards"][2 * int(index) + 1] if 0 <= index < 5 else -1

    def get_card_cost2(self, player, index):
        cost = self.get_card_cost(player, index)
        return cost - int(self.observation.world["weather"] == 2 and cost > 1)

    def get_special_data(self, player, index):
        if not 0 <= index < 28:
            return -1
        return self.player(1 - player if index == 9 else player)["special"][int(index)]

    def get_fflags(self, player, flag):
        return bool(int(self.player(player)["fflags"]) & int(flag))

    def get_aflags(self, player, flag):
        return bool(int(self.player(player)["aflags"]) & int(flag))

    def get_correction(self, player, flag):
        return bool(int(self.player(player)["correction"]) & int(flag))

    def get_key_stat2(self, player, key):
        return self.player(player)["keys"][int(key)] if 0 <= key < 10 else -1

    def get_key_map2(self, player):
        return self.player(player)["keys"]

    def get_deck_list(self, player):
        return self.player(0)["deck"]

    def get_obj_data(self, player, index):
        objects = self.player(player)["objects"]
        if not 0 <= index < len(objects):
            return ()
        obj = objects[int(index)]
        return tuple(obj[key] for key in ("act", "x", "y", "hp", "img", "frame", "xspeed",
                                         "yspeed", "address", "attackarea_n", "hitarea_n"))

    def box(self, entity, kind, index):
        boxes = entity[kind]
        return boxes[int(index)] if 0 <= index < len(boxes) else ()

    def get_hitarea(self, player, index):
        return self.box(self.player(player), "hitarea", index)

    def get_attackarea(self, player, index):
        return self.box(self.player(player), "attackarea", index)

    def object_box(self, player, obj, index, kind):
        objects = self.player(player)["objects"]
        return self.box(objects[int(obj)], kind, index) if 0 <= obj < len(objects) else ()

    def get_obj_attackarea(self, player, obj, index):
        return self.object_box(player, obj, index, "attackarea")

    def get_obj_hitarea(self, player, obj, index):
        return self.object_box(player, obj, index, "hitarea")

    def get_opt_xy(self, player, index):
        entity = self.player(player)
        character = entity["char"]
        if character not in (3, 5, 12) or index < 0 or (character == 5 and index != 0):
            return ()
        for obj in entity["objects"]:
            matches = ((character == 3 and obj["attackarea_n"] == 0 and obj["act"] == 805 and obj["img"] != 221)
                       or (character == 5 and obj["act"] == 899)
                       or (character == 12 and obj["act"] in (801, 852, 855)))
            if matches:
                if index == 0:
                    return obj["x"], obj["y"]
                index -= 1
        return ()

    def observe(self, observation):
        for key, value in zip(self.delays, self.read_delays(), strict=True):
            self.delays[key] = int(value)
        self.observation = observation
        world = observation.world
        for name, value in world.items():
            self.globals[name.encode()] = value
        for prefix, player in zip(("my_", "enemy_"), observation.players, strict=True):
            for name, value in player.items():
                if isinstance(value, (int, float)):
                    self.globals[(prefix + name).encode()] = value
        own, enemy = observation.players
        self.globals[b"x"], self.globals[b"y"] = own["x"], own["y"]
        self.globals[b"ex"], self.globals[b"ey"] = enemy["x"], enemy["y"]
        right = own["x"] < enemy["x"] or (own["x"] == enemy["x"] and own["dir"] == 1)
        for name, value in {"front": 3 if right else 2, "back": 2 if right else 3,
            "d_front": 11 if right else 10, "d_back": 10 if right else 11,
            "u_front": 13 if right else 12, "u_back": 12 if right else 13,
            "is_dir_front": own["dir"] == (1 if right else -1),
            "is_card_use": bool(own["is_card_use"]), "is_th105": False, "is_th123": True}.items():
            self.globals[name.encode()] = value
        dx = int(abs(float_difference(own["x"], enemy["x"])))
        dy = int(abs(float_difference(own["y"], enemy["y"])))
        for name, value in (("dis", dx), ("dis_x", dx), ("dis_y", dy), ("dis2", int(math.hypot(dx, dy)))):
            self.globals[name.encode()] = value
        distance, centre = 10000, 10000
        for obj in enemy["objects"]:
            centre = min(centre, int(abs(float_difference(own["x"], obj["x"]))))
            if len(obj["attackarea"]) >= 16:
                continue
            for left, _, right, _ in obj["attackarea"]:
                # is_bullethit assigns 1 directly for an enclosing box. A
                # preceding zero distance must not survive that assignment.
                distance = (1 if left < own["x"] < right else
                            min(distance, int(min(abs(float_difference(left, own["x"])),
                                                  abs(float_difference(right, own["x"]))))))
        self.globals[b"obj_dis"], self.globals[b"obj_dis2"] = distance, centre
        if world["weather2"] == 19 and 1000 - self.delays[b"weather_delay"] < world["weather_time"]:
            self.globals[b"weather"] = world["weather2"]
        delay = self.delays[b"data_delay"]
        if delay != self.data_delay:
            self.delayed_actions = [0] * delay
            self.delayed_blocks = [0] * delay
            self.delayed_frames = [0] * delay
            self.data_delay = delay
        if delay:
            previous = (self.delayed_actions[0], self.delayed_blocks[0], self.delayed_frames[0])
            for name, value in zip((b"enemy_act", b"enemy_act_block", b"enemy_frame"), previous, strict=True):
                self.globals[name] = value
            # Preserve main.cpp's second memmove destination (act, not
            # act_block). It overwrites the shifted action buffer with blocks.
            self.delayed_actions[:-1] = self.delayed_blocks[1:]
            self.delayed_frames[:-1] = self.delayed_frames[1:]
            self.delayed_actions[-1] = (int(enemy["act"]) + 32768) % 65536 - 32768
            self.delayed_blocks[-1] = (int(enemy["act_block"]) + 32768) % 65536 - 32768
            self.delayed_frames[-1] = (int(enemy["frame"]) + 2**31) % 2**32 - 2**31
