#!/usr/bin/env python3
"""
医学文献迭代检索系统 v7.0

混合方法：
1. 第 1 次检索：精确检索词
2. 第 2 次检索：扩展检索词 (通过不同检索词扩大范围)
3. 第 3 次检索：经典作者检索
4. 第 4 次检索：高分期刊 + 经典文献

关键：使用不同的检索词扩大范围，不是胡乱扩大！

作者：宵宵
日期：2026-07-12
"""

from typing import List, Dict
from datetime import datetime


class IterativeLiteratureSearch:
    """迭代检索系统 v7.0 - 混合方法"""

    def __init__(self):
        from literature_search import PubMedSearcher
        self.searcher = PubMedSearcher()

        # 期刊影响因子
        self.journal_if = {
            'nejm': 158.5, 'lancet': 168.9, 'nature': 64.8,
            'science': 56.9, 'cell': 64.5, 'bmj': 93.1,
            'jama': 123.1, 'molecular psychiatry': 11.4,
            'american journal of psychiatry': 14.4,
            'brain stimulation': 7.7,
        }

        # 检索词库 (按轮次组织)
        self.search_queries = {
            'rTMS': {
                'iter1': [
                    "repetitive transcranial magnetic stimulation depression",
                    "rTMS major depressive disorder",
                ],
                'iter2': [
                    "theta burst stimulation depression",
                    "iTBS depression",
                    "cTBS depression",
                ],
                'iter3': [
                    "Stanford SNT depression",
                    "accelerated TMS depression",
                    " SAINT depression",
                ],
                'iter4': [
                    "transcranial magnetic stimulation treatment-resistant depression",
                    "rTMS depression randomized trial",
                ],
            },
            'esketamine': {
                'iter1': ["esketamine depression", "esketamine treatment-resistant depression"],
                'iter2': ["intranasal esketamine", "spravato depression"],
                'iter3': ["intravenous esketamine", "ketamine depression"],
                'iter4': ["esketamine suicidal ideation", "rapid acting antidepressant"],
            },
        }

        # 排除词
        self.exclude_terms = {
            'rTMS': ['electroconvulsive', 'ECT'],
            'esketamine': ['surgery', 'anesthesia', 'pain'],
        }

    def search_iterative(self, topic: str,
                        max_iterations: int = 4,
                        max_results: int = 50,
                        use_references: bool = False) -> List[Dict]:
        """
        混合迭代检索

        Args:
            topic: 研究领域
            max_iterations: 最大迭代次数
            max_results: 最大结果数
            use_references: 是否尝试通过参考文献扩大 (默认 False，因为 PubMed API 限制)

        Returns:
            文献列表
        """
        print(f"\n{'='*60}")
        print(f"混合迭代检索：{topic}")
        print(f"{'='*60}")

        all_papers = []
        seen_pmids = set()
        current_papers = []

        # 获取排除词
        topic_key = self._get_topic_key(topic)
        exclude = self.exclude_terms.get(topic_key, [])

        # ========== 第 1 次检索：精确检索 ==========
        print(f"\n【第 1 次检索】精确检索...")
        queries = self._get_queries(topic, 'iter1')
        current_papers = self._search_queries(queries, exclude)
        new_count = self._add_papers(current_papers, all_papers, seen_pmids, iteration=1, source='精确检索')
        print(f"  新增：{new_count} 篇 | 累计：{len(all_papers)} 篇")

        # ========== 第 2 次检索：扩展检索 ==========
        if max_iterations >= 2:
            print(f"\n【第 2 次检索】扩展检索...")
            queries = self._get_queries(topic, 'iter2')
            current_papers = self._search_queries(queries, exclude)
            new_count = self._add_papers(current_papers, all_papers, seen_pmids, iteration=2, source='扩展检索')
            print(f"  新增：{new_count} 篇 | 累计：{len(all_papers)} 篇")

        # ========== 第 3 次检索：经典作者/新主题 ==========
        if max_iterations >= 3:
            print(f"\n【第 3 次检索】经典作者/新主题...")
            queries = self._get_queries(topic, 'iter3')
            current_papers = self._search_queries(queries, exclude)
            new_count = self._add_papers(current_papers, all_papers, seen_pmids, iteration=3, source='经典作者')
            print(f"  新增：{new_count} 篇 | 累计：{len(all_papers)} 篇")

        # ========== 第 4 次检索：高分期刊 ==========
        if max_iterations >= 4:
            print(f"\n【第 4 次检索】高分期刊...")
            queries = self._get_queries(topic, 'iter4')
            current_papers = self._search_queries(queries, exclude)
            new_count = self._add_papers(current_papers, all_papers, seen_pmids, iteration=4, source='高分期刊')
            print(f"  新增：{new_count} 篇 | 累计：{len(all_papers)} 篇")

        # ========== 质量评估 ==========
        print(f"\n【质量评估】按 IF+ 研究类型排序...")
        scored = self._score_papers(all_papers)
        sorted_papers = sorted(scored, key=lambda x: x.get('total_score', 0), reverse=True)

        # ========== 限制数量 ==========
        if max_results:
            sorted_papers = sorted_papers[:max_results]

        # ========== 生成报告 ==========
        print(f"\n【检索报告】")
        report = self._generate_report(sorted_papers)
        self._print_report(report)

        return sorted_papers

    def _get_queries(self, topic: str, iteration: str) -> List[str]:
        """获取检索词"""
        topic_key = self._get_topic_key(topic)

        # 正确的字典访问方式
        if topic_key in self.search_queries:
            if iteration in self.search_queries[topic_key]:
                return self.search_queries[topic_key][iteration]

        return [topic]

    def _search_queries(self, queries: List[str], exclude_terms: List[str]) -> List[Dict]:
        """执行多个检索词"""
        papers = []
        seen_pmids = set()

        for query in queries:
            search_papers = self.searcher.search(query, max_results=20)
            for paper in search_papers:
                pmid = paper.get('pmid')
                title = paper.get('title', '').lower()

                # 过滤排除词
                if exclude_terms and any(term in title for term in exclude_terms):
                    continue

                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    papers.append(paper)

        return papers

    def _get_topic_key(self, topic: str) -> str:
        """获取主题关键词 (返回字典中的原始键)"""
        topic_lower = topic.lower()

        # 检查是否包含检索词库中的键 (忽略大小写)
        for key in self.search_queries.keys():
            if key.lower() in topic_lower:
                return key  # 返回原始键

        # 如果没有匹配，使用第一个词
        first_word = topic_lower.split()[0] if topic_lower.split() else topic_lower

        # 模糊匹配 (忽略大小写)
        for key in self.search_queries.keys():
            if first_word.startswith(key.lower()) or key.lower().startswith(first_word):
                return key  # 返回原始键

        return first_word

    def _add_papers(self, papers: List[Dict], all_papers: List[Dict],
                   seen_pmids: set, iteration: int, source: str) -> int:
        """添加论文"""
        new_count = 0
        for paper in papers:
            pmid = paper.get('pmid')
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                paper['iteration'] = iteration
                paper['source'] = source
                all_papers.append(paper)
                new_count += 1
        return new_count

    def _score_papers(self, papers: List[Dict]) -> List[Dict]:
        """评分"""
        scored = []
        current_year = datetime.now().year

        for paper in papers:
            score = 0
            journal = paper.get('journal', '').lower()
            year_str = paper.get('pubdate', '9999')[:4]
            title = paper.get('title', '').lower()

            # 1. IF 评分
            if_score = 0
            for j_name, if_val in self.journal_if.items():
                if j_name in journal:
                    if_score = min(if_val / 2, 50)
                    paper['if_value'] = if_score
                    break

            if not paper.get('if_value'):
                if 'psychiatry' in journal or 'brain' in journal:
                    if_score = 20
                else:
                    if_score = 10
                paper['if_value'] = if_score

            score += if_score

            # 2. 被引频次
            citation_count = paper.get('cited_by_count', 0)
            score += min(citation_count * 3, 20)

            # 3. 研究类型
            if 'meta-analysis' in title:
                score += 15
            elif 'randomized' in title and 'trial' in title:
                score += 12
            elif 'review' in title:
                score += 8
            else:
                score += 8

            # 4. 时效性
            try:
                year = int(year_str) if year_str else 9999
                years_old = current_year - year
                if years_old <= 1:
                    score += 10
                elif years_old <= 3:
                    score += 8
                elif years_old <= 5:
                    score += 6
                elif years_old <= 10:
                    score += 4
                else:
                    score += 2
            except:
                pass

            # 5. 迭代轮次加分
            iteration = paper.get('iteration', 1)
            if iteration >= 2:
                score += 5

            paper['total_score'] = score
            scored.append(paper)

        return scored

    def _generate_report(self, papers: List[Dict]) -> Dict:
        """生成报告"""
        report = {
            'total_count': len(papers),
            'by_iteration': {1: 0, 2: 0, 3: 0, 4: 0},
            'high_if_count': 0,
            'rct_count': 0,
            'meta_count': 0,
        }

        for paper in papers:
            iteration = paper.get('iteration', 1)
            report['by_iteration'][iteration] = report['by_iteration'].get(iteration, 0) + 1

            if paper.get('if_value', 0) >= 50:
                report['high_if_count'] += 1

            title = paper.get('title', '').lower()
            if 'randomized' in title and 'trial' in title:
                report['rct_count'] += 1
            elif 'meta-analysis' in title:
                report['meta_count'] += 1

        return report

    def _print_report(self, report: Dict):
        """打印报告"""
        print(f"\n  文献总数：{report['total_count']} 篇")
        print(f"\n  按迭代轮次 (通过不同检索词扩大范围):")
        for i in range(1, 5):
            count = report['by_iteration'].get(i, 0)
            if count > 0:
                print(f"    第{i}次检索：{count} 篇")

        print(f"\n  通过扩展检索找到：{report['total_count'] - report['by_iteration'].get(1, 0)} 篇")
        print(f"\n  高质量文献:")
        print(f"    IF≥50: {report['high_if_count']} 篇")
        print(f"    RCT: {report['rct_count']} 篇")
        print(f"    Meta 分析：{report['meta_count']} 篇")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    searcher = IterativeLiteratureSearch()

    # 执行 4 轮迭代检索
    papers = searcher.search_iterative(
        topic="rTMS depression",
        max_iterations=4,
        max_results=50
    )

    print(f"\n✅ 混合迭代检索完成!")
    print(f"   总文献数：{len(papers)} 篇")
