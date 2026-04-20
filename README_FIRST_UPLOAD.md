# pre_train_software - 首次上传预备

此目录用于 Git 首次上传预备（当前仅本地准备，不推送远端）。

## 已准备内容

- Rust 源码与配置：`src/`, `Cargo.toml`, `Cargo.lock`, `build.rs`
- 内置工具源码：`embedded_tools_src/`
- Windows 程序图标资源：`resources/app_icon.ico`
- `.gitignore`：已排除训练产物、大模型文件与本地构建目录

## 后续（你确认后再做）

1. `git add .`
2. `git commit -m "chore: first upload baseline"`
3. 绑定远端并 `git push -u origin main`
