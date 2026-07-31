> 以下内容由AI生成，我懒得写README

# SkyAutoMusic 自动弹琴

## 项目简介
SkyAutoMusic 是一款用于自动演奏《Sky光遇》等游戏内乐器的Python工具。支持多乐谱选择、按键映射、现代美观的GUI界面、全局热键控制，并可自动检测并置顶游戏窗口。

## 主要功能
- 支持多份JSON格式乐谱，自动识别并选择
- 支持自定义音符-按键映射
- 现代美观的图形界面（Tkinter）
- 支持多键同时按下，节奏精准
- 全局热键控制（可自定义/重置）
- 自动检测并置顶Sky/光遇游戏窗口
- 收藏曲谱、持久播放列表与分页切换（全部/收藏/播放列表）
- 乐谱信息悬停显示与走马灯效果
- 选中曲谱后立即展示歌名、作者、制谱人、文件名、总时长、BPM 与音符数
- 曲谱列表支持纵向和横向滚动、搜索结果计数与选中状态保持
- JSON 曲谱可使用内置 Sky 钢琴音色本地试听，不会向游戏发送按键
- 按乐谱 `time` 绝对时间戳播放，长曲节奏更稳定
- **虚拟 HID / 驱动级键盘**：基于 Interception 内核驱动在驱动层注入按键，可绕过部分游戏对 SendInput 的屏蔽；未安装驱动时自动回退到常规键盘
- 半透明乐谱覆盖层：显示播放进度、音符密度与当前按键，F10 可解锁拖动位置
- 诊断页：查看热键、游戏窗口、前台窗口、管理员权限、键盘输入方式与最近按键日志
- 窗口大小和位置自动保存，下次启动自动恢复
- 音频智能扒谱：离线分离人声、鼓点、钢琴/键盘、贝斯、吉他与完整伴奏，按“人声主旋律优先”融合为15键光遇琴谱
- 鼓点只校准 BPM、重拍和量化网格；专用乐器轨补充和声、低音，完整伴奏仅智能补漏
- 扒谱草稿可查看15键时间线、调整调性/八度/量化/复音数和六轨开关；最终琴谱统一使用光遇钢琴音色试听
- 网易云在线扒谱：支持关键词搜索、分页结果、加密 Cookie、yt-dlp 下载及自动生成可审核草稿
- 适配Windows平台

## 安装依赖
建议使用Python 3.10及以上版本（驱动级键盘依赖 `interception-python` 需要 Python>=3.10）。

```bash
pip install pyautogui keyboard psutil pywin32 interception-python
```

完整安装（包含扒谱）请使用 Python 3.10：

```bash
pip install -r requirements.txt
python scripts/fetch_separation_model.py
```

发布版固定使用 Python 3.10，并通过 `requirements-lock.txt` 安装锁定依赖，使
Basic Pitch 使用 ONNX Runtime、Demucs 使用 CPU 版 PyTorch/Torchaudio；不打包
TensorFlow 或 CUDA。完整 ZIP 已包含模型，只有源码运行需要执行上述模型获取脚本。

### 启用虚拟 HID / 驱动级键盘（可选，推荐）
默认输入方式为"自动"，会优先尝试驱动级键盘；若未安装 Interception 驱动则自动回退到常规键盘（`keyboard` / `pyautogui`），不影响使用。

若要启用驱动级键盘以获得更好的兼容性（绕过部分游戏对 SendInput 的屏蔽），需安装 Interception 内核驱动：

1. 下载 Interception 驱动安装包：<https://github.com/oblitum/Interception/releases>（`Interception.zip`）。
2. 解压后，**以管理员身份**打开命令行，执行：
   ```cmd
   install-interception.exe /install
   ```
3. **重启电脑**使驱动生效。
4. 重新运行本程序，诊断页"键盘输入方式"会显示"Interception 驱动已就绪"。
5. 若按键未打到游戏里，可点击"校准驱动级键盘"，按提示按一次键以识别键盘设备。

> 注意：少数带有强反作弊的游戏（如 Vanguard、部分 EAC）检测到该驱动会拒绝启动，请按需取舍。卸载驱动：`install-interception.exe /uninstall` 后重启。

## 使用方法
1. 将乐谱（JSON格式，结构见示例）放入 `Sheet Music` 文件夹。
2. 运行 `play_music_gui.py`：
   ```bash
   python play_music_gui.py
   ```
3. 在界面中选择乐谱，点击"游戏演奏"将按键发送到游戏，或点击"本地试听"直接预览 JSON 曲谱。
   - F5：开始游戏演奏；F6/F8：上一首/下一首；F7：停止当前演奏或试听。
   - F11：暂停/继续当前演奏或试听。
   - F10：切换乐谱覆盖层的点击穿透锁定；解锁后可拖动覆盖层位置。
4. 可在"说明"页查看作者主页、交流群等信息。
5. 右键曲谱可收藏/取消收藏，分页切换显示全部或收藏曲谱。
6. 程序会自动检测Sky/光遇窗口并置顶，未检测到会提示。
7. 窗口大小和位置、收藏数据等会自动保存，无需手动配置。
8. 点击“生成乐谱”进入扒谱窗口：
   - “本地文件”页可选择音频或 MIDI；音频默认使用智能六轨融合，MIDI 会自动绕过分离。
   - 六轨面板可独立控制人声、钢琴/键盘、贝斯、吉他和完整伴奏是否参与琴谱；鼓点开关只影响节奏，不产生琴键音符。
   - “单轨原声试听”播放分离出的真实 WAV；“光遇音色试听”播放融合后的15键琴谱，两者互斥。
   - 可选择“人声优先 / 键盘优先 / 平衡”融合预设。轨道、调性、八度、量化、复音数和重复音设置变化只重新融合缓存结果。
   - “网易云在线”页可搜索歌曲并翻页；选择结果后点击“生成在线草稿”，程序会通过 yt-dlp 下载并用内置 FFmpeg 转成临时 WAV。
   - 生成结果先保存在内存草稿中；试听、调参后点击“保存当前”或“保存全部”才会写入乐谱文件夹。
   - 会员或登录歌曲可点击 Cookie 区的“编辑”，粘贴 Netscape 格式 `cookies.txt` 完整内容。程序仅保留网易云域名 Cookie，并使用 Windows DPAPI 按当前用户加密保存到 `netease_auth.json`。

## 构建与发布（完整 ZIP 目录）

智能分轨需要模型、运行库和许可证，因此发布版采用 PyInstaller `onedir`，并将完整目录压缩为
`SkyAutoMusic-windows-x64.zip`。**不要只复制其中的 EXE**；乐谱与个人配置仍不打包。

### 方式一：GitHub Actions 自动构建（推荐）
1. 进入仓库的 **Actions** 页面，选择 `Build EXE & Release` 工作流。
2. 点击 **Run workflow**，可填写可选的 Release 名称，确认后即开始构建。
3. 构建完成后，自动在 **Releases** 中生成 `build-<序号>` 版本，下载并完整解压 `SkyAutoMusic-windows-x64.zip`。

> 构建在 GitHub 云端 Windows 环境中完成（依赖 Windows API），无需本地环境。

### 方式二：本地用 PyInstaller 构建
```bash
pip install -r requirements.txt pyinstaller==6.21.0
python scripts/fetch_separation_model.py
pyinstaller --noconfirm --onedir --windowed --name SkyAutoMusic ^
  --hidden-import keyboard --hidden-import win32timezone ^
  --hidden-import interception --collect-all interception ^
  --add-data "assets/audio/sky/Piano;assets/audio/sky/Piano" ^
  --add-data "assets/models/htdemucs_6s;assets/models/htdemucs_6s" ^
  --add-data "THIRD_PARTY_LICENSES;THIRD_PARTY_LICENSES" ^
  --collect-all basic_pitch --collect-all onnxruntime ^
  --collect-all demucs --collect-all sphn --collect-all torchaudio ^
  --collect-all yt_dlp --collect-all imageio_ffmpeg ^
  play_music_gui.py
```
生成的 `dist/SkyAutoMusic/` 是完整运行目录，请整体压缩或分发。

### 运行 exe 前的准备
- 将 ZIP 完整解压到一个**有写入权限**的目录（如桌面或专门文件夹），不要移动或删除 `_internal`、模型和音色资源。
- 在 `SkyAutoMusic.exe` **同级目录**放入 `Sheet Music/` 文件夹，并存放你的乐谱 JSON 文件。
- 程序会按需在同目录生成 `config.json`、`favorites.json`、`playlist.json` 等配置文件，设置、收藏与播放列表可持久化保存。

## 乐谱文件格式说明
- 乐谱为JSON文件，需包含`songNotes`字段。
- 示例结构：
```json
[
  {
    "name": "Army Dreamers (json)",
    ...,
    "songNotes": [
      {"time": 948, "key": "1Key0"},
      {"time": 948, "key": "1Key2"},
      ...
    ]
  }
]
```
- 同一time下的多个key表示同时按下。
- 支持多种乐谱结构，自动兼容解析。

## 特色功能说明
- **收藏与播放列表**：右键曲谱可收藏或加入播放列表；播放列表支持排序、移除和顺序自动续播。
- **本地试听**：使用 `assets/audio/sky/Piano/0.mp3` 至 `14.mp3` 预览 JSON 曲谱；`1KeyN` 与 `2KeyN` 均映射到 `N.mp3`，缺失音色会从 Sky Music 自动下载并缓存。
- **智能分轨**：使用 `htdemucs_6s` 离线分离六轨；模型或校验清单缺失、损坏时可切换为原有复音/单旋律模式继续扒谱。
- **乐谱信息展示**：选中曲谱时立即解析并显示歌名、作者、制谱人、文件名、播放总时长、BPM 和有效音符数。
- **稳定节奏播放**：播放器按乐谱 `time` 毫秒时间戳进行绝对时间调度，不按 BPM 重算节奏；BPM 字段主要作为乐谱元信息保留。
- **虚拟 HID / 驱动级键盘**：在诊断页"键盘输入方式"可选择"自动 / 虚拟HID驱动级键盘 / 常规键盘"。驱动级模式通过 Interception 内核驱动在驱动层注入按键，兼容性更好；点击"校准驱动级键盘"可重新识别键盘设备。
- **乐谱覆盖层**：开始演奏后显示半透明置顶窗口，默认点击穿透，不影响游戏操作；按 F10 解锁后可拖动到合适位置。
- **诊断页**：可查看热键注册结果、游戏窗口识别、当前前台窗口、管理员权限、键盘输入方式与最近按键日志，方便排查"按键没有打到游戏里"的问题。
- **窗口与配置**：窗口大小、位置、收藏、输入方式等均自动保存，无需手动配置。
- **资源路径适配**：所有资源文件（config.json、favorites.json、playlist.json、Sheet Music）均自动适配开发和打包环境，无需修改路径。
- **在线音频兼容**：发布版内置 `imageio-ffmpeg`，会增加约 31 MB 依赖体积，但无需用户另行安装 FFmpeg。

## 常见问题
- **找不到乐谱/收藏/配置文件？**
  - 请确保 `Sheet Music` 位于程序目录；其余配置文件缺失时会按需创建。
  - 程序已自动适配路径，无需手动调整。
- **窗口大小和位置未保存？**
  - 程序关闭时会自动保存窗口配置到config.json，重新打开会自动恢复。
- **热键无效？**
  - 请以管理员身份运行程序，或更换为未被系统占用的热键。
- **按键没有打到游戏里？**
  - 优先尝试启用"虚拟 HID / 驱动级键盘"：安装 Interception 驱动后，在诊断页选择"虚拟HID/驱动级键盘 (Interception)"，必要时点击"校准驱动级键盘"。
  - 仍不行时检查诊断页"前台窗口"是否为游戏窗口，并确认已以管理员身份运行。
- **诊断页显示"Interception 驱动未安装"？**
  - 未安装驱动时会自动回退到常规键盘。如需驱动级输入，请按 README 安装 Interception 驱动并重启。
- **按键映射不符？**
  - 请在代码中修改`note_to_key`字典。
- **网易云 Cookie 一直显示无效？**
  - 确认已在浏览器登录网易云，并导出 Netscape `cookies.txt`；文件首行应为 `# Netscape HTTP Cookie File`，且包含未过期的 `MUSIC_U`。
  - Cookie 由 Windows DPAPI 绑定到当前 Windows 用户，复制到另一台电脑或另一个系统账号后需要重新保存。
- **网易云歌曲无法扒谱？**
  - 无 Cookie 时只能获取公开可播放歌曲；Cookie 只能使用账号自身已有权限，不能绕过会员、版权、下架或地区限制。
  - 网易云接口和 yt-dlp 提取器可能随网站更新而变化，请先升级到项目锁定或更新后的 yt-dlp 版本。
- **其它问题**
  - 如遇异常可反馈至作者主页或交流群。

## 免责声明
本工具仅供学习与娱乐，请勿用于破坏游戏公平性。在线扒谱仅应处理你有权访问和使用的音频；本程序不会绕过会员、版权或地区限制。

---

如有新功能或需求，README会实时更新。 
