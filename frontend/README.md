# War News Bias Frontend

`frontend/` 是 `war_news_bias_detection` 的本地前端与桌面壳项目，技术栈为：

- `React + TypeScript + Vite`
- `Tailwind CSS`
- `Tauri 2`

当前目录的文档入口：

- 开发说明：[前端开发说明文档.md](./前端开发说明文档.md)
- 新设备部署：[新设备部署文档.md](./新设备部署文档.md)

常用命令：

```bash
npm install
npm run sync:data
npm run dev
```

打包前端：

```bash
npm run build
```

运行桌面版：

```bash
npm run tauri:dev
```

说明：

- 样例图片当前使用“相对仓库根目录”的路径方案。
- 仅迁移整个 `war_news_bias_detection` 仓库、且保持内部目录结构不变时，图片路径才可继续解析。
- 纯静态网页部署不支持直接读取仓库里的本地图片；详见 [新设备部署文档.md](./新设备部署文档.md)。
