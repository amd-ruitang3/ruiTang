skill 的格式：
  AAAAA
    -> SKILL.md    # 主入口最重要的东西
    -> reference.md   # 可选，详细说明和知识库
    -> example.md    # 可选，示例
    -> scripts    #可选，辅助脚本
        -> helper.py  


放哪里：
1. ~/.claude/skills 下面     # 用户级
2. 项目/.claude/skills 下面   # 项目级别
3. plugin 里面   


创建
1. 抄别人的
  1.1 claude 官方  /plugin
  1.2 第三方网站 skillsmp
  1.3 github 里有一手的
2. 自己创建 
  2.1 claude 帮你创建
  2.2 claude-creator plugin 帮你创建
      它还可以帮你 eval


怎么触发
1. 自动触发
2. /AAAAA 手动触发
