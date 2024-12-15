### frontend

#### Packages Versions
```
node    v14.18.2
npm     v8.19.4
```

#### Some Issues

1. 启动项目时，平时很好，突然报错：

```bash
Error: web@0.1.0 serve: `vue-cli-service serve`
Exit status 1
```

解决方案：

尝试删除 node_modules 和 package-lock.json，然后重新安装依赖。`npm install` 命令需要在 powershell 中执行。

```bash
rm -rf node_modules package-lock.json
npm install
```

