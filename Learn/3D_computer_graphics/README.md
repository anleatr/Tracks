## 使用说明

### 系统要求

操作系统：linux / windows

python3

### 运行

- linux:在项目目录下运行

```
chmod +x run_linux.sh
./run_linux.sh
```

- windows:双击run_windows.bat

## 功能说明

本编辑器实现了以下全部要求：
(1) 线框 · 非消隐 — 顶点与边的非消隐显示
关闭深度测试，所有边和顶点均可见（包括被遮挡部分）
勾选"显示顶点"可叠加渲染所有顶点
(2) 线框 · 消隐 — 隐藏元素动态去除
先渲染背景色的实体网格填充深度缓冲，再渲染线框
被遮挡的边/顶点自动不显示
(3)(4) 面绘制 + 光照模型
可独立开关三种光照分量：
环境光 (Ambient)
漫反射 — Lambert 模型 (Diffuse)
镜面反射 — Phong 反射模型 (Specular)
可调节光泽度 (Shininess 1~128)
(5) 明暗处理
Gouraud 明暗处理 — 逐顶点计算光照，颜色在片元间插值
Phong 明暗处理 — 法线在片元间插值，逐片元计算光照（高光更精确）
交互操作
操作	功能
鼠标左键拖拽	旋转模型
鼠标右键拖拽	平移视角
滚轮	缩放
按键 1 / 2 / 3	切换显示模式
按键 R	重置视角
技术实现
使用 Three.js (r160) + WebGL 渲染
自定义 GLSL 着色器实现 Gouraud/Phong 两种明暗处理
光照在视图空间 (view space) 中计算，光源位于世界坐标 (5, 10, 7)
通过 polygonOffset 解决线框与面的 z-fighting 问题
Flower.ply（194,818 顶点 / 389,632 面片）通过 PLYLoader 加载