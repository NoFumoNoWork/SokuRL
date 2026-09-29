# 安装与验收

## 依赖

使用 Windows、Python 3.11 x64、MSVC x86 和 CMake。游戏为《东方非想天则》1.10a，`th123.exe` 的 MD5 必须为 `DF35D1FBC7B583317ADABE8CD9F53B2E`。

游戏本体单独提供。不要替换或修改原始 `th123.exe`、`th123a.dat`、`th123b.dat`、`th123c.dat`。游戏放在仓库的 `th123_jp`，也可以使用本地目录联接指向已有安装。已有 `th105` 时，检查 `configex123.ini` 中的路径；修改前备份配置。

## Python 安装

以下命令使用标准 Windows 虚拟环境目录结构。先用环境管理器创建仓库内的 Python 3.11 环境，再执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe scripts\00_check_game.py
```

安装包含启动器必需的 `psutil`。当前控制路径不依赖训练框架；不必为了验证游戏接口安装深度学习软件。

## 原生源码版本

此次构建使用以下固定提交。不能把 SokuLib 换成最新分支：提交 `0794becf1f578b32329604483be2112fe0afd4ee` 的输入管理器类型与现有桥接代码不兼容。

| 源码 | 提交 | 相对路径 |
| --- | --- | --- |
| [SokuMods](https://github.com/SokuDev/SokuMods) | `eb0574cea1eda70484b736eb192964ef87e50a67` | `third_party/SokuMods` |
| [SokuLib](https://github.com/SokuDev/SokuLib) | `e96ff6941373703adc35a10831014cbcc6bac866` | `third_party/SokuMods/SokuLib` |
| [SkipIntro](https://github.com/SokuDev/SkipIntro) | `fd945ff5c1c5b7de0ff7d03b988e9219a3674213` | `third_party/SokuMods/modules/SkipIntro/Soku-SkipIntro` |
| [Detours](https://github.com/microsoft/Detours) | `64ec135a509884aa60ac6c19b59564f1da9cb2fa` | `third_party/SokuMods/detours` |

可递归检出 SokuMods 的固定提交；也可下载该提交的源码归档，并按表格补齐三个子模块。归档文件不包含子模块内容。

`native/RuntimeModules` 构建加载器和四个社区模块：WindowResizer、SkipIntro、MemoryPatch、ReplayDnD。SokuRLBridge 由自己的构建目录生成。这样无需为了运行本项目构建 SokuMods 中其余插件。

## 编译

在 MSVC 的 x86 开发者命令行中运行：

```text
cmake -S native/SokuRLBridge -B native/SokuRLBridge/build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build native/SokuRLBridge/build --target SokuRLBridge
cmake -S native/RuntimeModules -B native/RuntimeModules/build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build native/RuntimeModules/build
```

这组命令使用开发者命令行所选的 x86 编译器。不要给 NMake 生成器加 `-A Win32`；使用 Visual Studio 生成器时才用该参数。不要在同一构建目录混用生成器。

社区源码按 UTF-8 编译，避免机器默认字符集把注释误读为代码。构建输出应为 x86 DLL。

## 部署清单

游戏退出后部署。若目标文件已存在，先备份并逐项说明覆盖内容。

| 构建结果或源配置 | 游戏目录内目标 |
| --- | --- |
| `native/RuntimeModules/build/d3d9.dll` | `d3d9.dll` |
| `native/SokuRLBridge/build/SokuRLBridge.dll` | `modules/SokuRLBridge/SokuRLBridge.dll` |
| `native/RuntimeModules/build/WindowResizer.dll` | `modules/WindowResizer/WindowResizer.dll` |
| `native/RuntimeModules/build/SkipIntro.dll` | `modules/SkipIntro/SkipIntro.dll` |
| `native/RuntimeModules/build/MemoryPatch.dll` | `modules/MemoryPatch/MemoryPatch.dll` |
| `native/RuntimeModules/build/ReplayDnD.dll` | `modules/ReplayDnD/ReplayDnD.dll` |

复制上述社区模块目录中的 INI 配置。`SWRSToys.ini` 使用以下模块列表：

```ini
[Module]
WindowResizer=modules/WindowResizer/WindowResizer.dll
SokuRLBridge=modules/SokuRLBridge/SokuRLBridge.dll
SkipIntro=modules/SkipIntro/SkipIntro.dll
MemoryPatch=modules/MemoryPatch/MemoryPatch.dll
ReplayDnD=modules/ReplayDnD/ReplayDnD.dll
```

为配合现有验收程序，SkipIntro 的 GLOBAL 设置为 `scene_id=3`、`type=8`、`subtype=0`；P1 使用角色 1，P2 使用角色 0，双方卡组设为 0。MemoryPatch 只开启 `AllowMultiInstance`。VS 启动器会临时调整 SkipIntro 的场景配置并恢复原始内容。

## 真实游戏验收

先由人类确认画面、移动、跳跃和攻击，再运行程序控制：

```powershell
.\.venv\Scripts\python.exe tools\sokurl.py practice
.\.venv\Scripts\python.exe tools\sokurl.py list
.\.venv\Scripts\python.exe tools\sokurl.py shutdown --pid <本次启动的进程编号>
.\.venv\Scripts\python.exe tools\vsplayer_validation.py
```

VS 验收报告要求双方攻击、对象生成、伤害和伤害持续均通过。程序会关闭它自己启动的游戏实例。报告位于 `logs/validation`。

需要完整验证加速时，运行 README 中的正常、无渲染与无限速一致性验证。协议单元测试成功不代表真实游戏验收成功。
