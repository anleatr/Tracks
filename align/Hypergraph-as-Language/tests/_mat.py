import matplotlib.font_manager as fm

# 获取所有已加载字体名称
all_fonts = sorted([font.name for font in fm.fontManager.ttflist])

# 打印全部字体（字体很多，会刷屏）
for font_name in all_fonts:
    print(font_name)
