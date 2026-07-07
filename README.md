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
- 收藏曲谱、分页切换（全部/收藏）
- 乐谱信息悬停显示与走马灯效果
- 右侧主控区展示详细乐谱信息（歌名、作者、制谱人、文件名）
- 自动读取BPM，节奏自适应
- 窗口大小和位置自动保存，下次启动自动恢复
- 适配Windows平台

## 安装依赖
建议使用Python 3.8及以上版本。

```bash
pip install pyautogui keyboard psutil pywin32
```

## 使用方法
1. 将乐谱（JSON格式，结构见示例）放入 `Sheet Music` 文件夹。
2. 运行 `play_music_gui.py`：
   ```bash
   python play_music_gui.py
   ```
3. 在界面中选择乐谱，点击"开始演奏"或使用热键（默认F5/F7）控制。
4. 可在"说明"页查看作者主页、交流群等信息。
5. 右键曲谱可收藏/取消收藏，分页切换显示全部或收藏曲谱。
6. 程序会自动检测Sky/光遇窗口并置顶，未检测到会提示。
7. 窗口大小和位置、收藏数据等会自动保存，无需手动配置。

## 构建与发布（EXE）

本工具提供两种方式获取 Windows 可执行文件（exe）。**乐谱与配置文件均不打包进 exe，需自行放置**（见下方说明）。

### 方式一：GitHub Actions 自动构建（推荐）
1. 进入仓库的 **Actions** 页面，选择 `Build EXE & Release` 工作流。
2. 点击 **Run workflow**，可填写可选的 Release 名称，确认后即开始构建。
3. 构建完成后，自动在 **Releases** 中生成 `build-<序号>` 版本，下载其中的 `SkyAutoMusic.exe` 即可。

> 构建在 GitHub 云端 Windows 环境中完成（依赖 Windows API），无需本地环境。

### 方式二：本地用 PyInstaller 构建
```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --windowed --name SkyAutoMusic ^
  --hidden-import keyboard --hidden-import win32timezone play_music_gui.py
```
生成的 `dist/SkyAutoMusic.exe` 即为可执行文件。

### 运行 exe 前的准备
- 将 `SkyAutoMusic.exe` 放到一个**有写入权限**的目录（如桌面或专门文件夹）。
- 在该 exe **同级目录**放入 `Sheet Music/` 文件夹，并存放你的乐谱 JSON 文件。
- 首次运行会自动在同目录生成 `config.json`、`favorites.json` 等配置文件，设置与收藏可持久化保存。

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
- **收藏与分页**：右键曲谱可收藏，分页按钮切换显示全部/收藏曲谱。
- **乐谱信息展示**：右侧主控区高亮显示歌名、作者、制谱人、文件名。
- **BPM节奏适配**：自动读取乐谱bpm字段，按bpm自动调整演奏节奏。
- **窗口与配置**：窗口大小、位置、收藏等均自动保存，无需手动配置。
- **资源路径适配**：所有资源文件（config.json、favorites.json、Sheet Music）均自动适配开发和打包环境，无需修改路径。

## 常见问题
- **找不到乐谱/收藏/配置文件？**
  - 请确保`Sheet Music`、`config.json`、`favorites.json`在项目目录下。
  - 程序已自动适配路径，无需手动调整。
- **窗口大小和位置未保存？**
  - 程序关闭时会自动保存窗口配置到config.json，重新打开会自动恢复。
- **热键无效？**
  - 请以管理员身份运行程序，或更换为未被系统占用的热键。
- **按键映射不符？**
  - 请在代码中修改`note_to_key`字典。
- **其它问题**
  - 如遇异常可反馈至作者主页或交流群。

## 免责声明
本工具仅供学习与娱乐，请勿用于破坏游戏公平性。

---

如有新功能或需求，README会实时更新。 