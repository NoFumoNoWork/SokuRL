"""Resolve one public play configuration into owned network clients."""
import ipaddress


def client_plan(settings):
    connection = settings["connection"]
    if connection not in {"local", "host", "join"}:
        raise ValueError("connection must be local, host or join")
    human, ai, network = settings["human"], settings["ai"], settings["network"]
    if type(human["seat"]) is not int or human["seat"] not in (1, 2):
        raise ValueError("human.seat must be 1 or 2")
    ipaddress.IPv4Address(network["address"])
    if type(network["port"]) is not int or not 1 <= network["port"] <= 65535:
        raise ValueError("network.port must be in [1, 65535]")
    for player in (human, ai):
        if type(player["character"]) is not int or not 0 <= player["character"] < 20:
            raise ValueError("character must be a playable ID from 0 to 19")
        for key in ("render", "mute_audio"):
            if type(player[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
    if type(human["automate_menu"]) is not bool:
        raise ValueError("human.automate_menu must be a boolean")
    seat = 2-human["seat"] if connection == "local" else ("host", "join").index(connection)
    clients = [{"name": "ai", "seat": seat, "role": ("host", "join")[seat],
                "automate_menu": True, "realtime": True, **ai}]
    if connection == "local":
        clients.append({**human, "name": "human", "seat": 1-seat,
                        "role": ("host", "join")[1-seat], "realtime": False})
    return tuple(sorted(clients, key=lambda client: client["seat"]))
