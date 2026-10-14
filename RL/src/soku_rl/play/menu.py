"""Select a network role and opponent from the same Hydra play catalog."""
import json


CHARACTERS = ("博丽灵梦", "雾雨魔理沙", "十六夜咲夜", "爱丽丝", "帕秋莉", "魂魄妖梦",
              "蕾米莉亚", "西行寺幽幽子", "八云紫", "伊吹萃香", "铃仙", "射命丸文",
              "小野塚小町", "永江衣玖", "比那名居天子", "东风谷早苗", "琪露诺", "红美铃",
              "灵乌路空", "洩矢诹访子")


def play_menu(catalog, settings, read, write):
    def choose(title, choices):
        write(title)
        for index, (_, label) in enumerate(choices, 1):
            write(f"  {index}. {label}")
        selected = read("输入编号：").strip()
        if not selected.isdecimal() or not 1 <= int(selected) <= len(choices):
            raise ValueError("请输入列表中的编号")
        return choices[int(selected)-1][0]

    connection = choose("连接方式", (("local", "本机人机对战"), ("host", "AI 建房，等待网络玩家"),
                                      ("join", "AI 加入网络玩家的房间")))
    track = choose("AI 使用的信息和反应设置", (("human", "拟人模式"), ("superhuman", "超人模式（包含全部神 AI）")))
    options = [(name, f"{opponent.label}（{'、'.join(CHARACTERS[c] for c in opponent.characters)}）")
               for name, opponent in catalog.items() if track in opponent.tracks]
    name = choose("选择对手", options)
    opponent = catalog[name]
    character = (opponent.characters[0] if len(opponent.characters) == 1 else
                 choose("AI 使用的角色", [(c, CHARACTERS[c]) for c in opponent.characters]))
    overrides = ["operation=play", f"track={track}", "opponent="+json.dumps(name, ensure_ascii=False),
                 f"play.connection={connection}", f"play.ai.character={character}"]
    if connection == "local":
        seat = choose("玩家座位", ((1, "1P（玩家建房）"), (2, "2P（AI 建房）")))
        overrides.append(f"play.human.seat={seat}")
        write("进入 SokuRL - Player 窗口后，自行选择角色和卡组。")
    else:
        if connection == "join":
            address = read(f"房主的 IPv4 地址（回车使用 {settings['network']['address']}）：").strip()
            if address:
                overrides.append("play.network.address="+json.dumps(address))
        port = read(f"房间端口（回车使用 {settings['network']['port']}）：").strip()
        if port:
            if not port.isdecimal():
                raise ValueError("端口必须是整数")
            overrides.append(f"play.network.port={int(port)}")
    return overrides
