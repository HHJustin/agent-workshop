"""
Skill 加载器 — 解析 SKILL.md → 注入 Agent

格式参考 OpenSquilla，简化版：
    YAML frontmatter (name/description/trigger_keywords/tools)
    + Markdown body (执行流程 + 约束)

使用:
    loader = SkillLoader()
    skill = loader.discover().get("network_patrol")
    prompt, tools = skill.inject()

Author: 程响
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from app.logger import logger


@dataclass
class SkillSpec:
    """一个 Skill 的定义"""
    name: str = ""
    description: str = ""
    trigger_keywords: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    body: str = ""  # Markdown 执行流程
    source_path: str = ""

    def matches(self, query: str) -> bool:
        """检查 query 是否触发此 Skill"""
        q = query.lower()
        return any(kw.lower() in q for kw in self.trigger_keywords)

    def inject(self) -> tuple[str, list]:
        """返回 (system_prompt_override, tool_names)"""
        prompt = f"""你是{self.name}执行专家。

{self.body}

## 核心约束
1. 严格按上述流程执行，不跳步、不加步
2. 每步完成后检查结果，异常时标注但不自动修复
3. 巡检完成后必须给出明确的结论（正常/异常+详情）
4. 不执行任何修改操作（不重启、不变更配置）"""
        return prompt, list(self.tools)


class SkillLoader:
    """Skill 加载器 — 扫描 bundled/ 目录"""

    def __init__(self, bundled_dir: str = ""):
        if not bundled_dir:
            bundled_dir = str(Path(__file__).resolve().parent / "bundled")
        self.bundled_dir = bundled_dir
        self._skills: dict[str, SkillSpec] = {}

    def discover(self) -> dict[str, SkillSpec]:
        """扫描 bundled/ 下所有 SKILL.md"""
        if self._skills:
            return self._skills

        base = Path(self.bundled_dir)
        if not base.exists():
            logger.warning(f"[SkillLoader] 目录不存在: {base}")
            return {}

        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue

            try:
                spec = _parse_skill_md(md_path)
                spec.source_path = str(md_path)
                skill_key = skill_dir.name
                self._skills[skill_key] = spec
                logger.info(f"[SkillLoader] 加载: {skill_key} → {spec.trigger_keywords}")
            except Exception as e:
                logger.warning(f"[SkillLoader] 解析失败 {md_path}: {e}")

        logger.info(f"[SkillLoader] 发现 {len(self._skills)} 个 Skill")
        return self._skills

    def match(self, query: str) -> SkillSpec | None:
        """匹配第一个触发的 Skill"""
        for spec in self.discover().values():
            if spec.matches(query):
                return spec
        return None


def _parse_skill_md(path: Path) -> SkillSpec:
    """解析 SKILL.md 的 YAML frontmatter + Markdown body"""
    text = path.read_text(encoding="utf-8")

    spec = SkillSpec()
    # 提取 YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if not match:
        spec.body = text
        return spec

    frontmatter = match.group(1)
    spec.body = match.group(2).strip()

    # 简单 YAML 解析（不依赖 pyyaml）
    for line in frontmatter.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()

        if key == "name":
            spec.name = val
        elif key == "description":
            spec.description = val
        elif key == "trigger_keywords":
            spec.trigger_keywords = _parse_yaml_list(line)
        elif key == "tools":
            spec.tools = _parse_yaml_list(line)

    return spec


def _parse_yaml_list(line: str) -> list[str]:
    """解析 YAML 行内列表: [a, b, c] 或 - a \\n - b"""
    # 格式1: [a, b, c]
    bracket = re.search(r'\[(.*?)\]', line)
    if bracket:
        return [x.strip().strip('"').strip("'") for x in bracket.group(1).split(",") if x.strip()]

    return []
