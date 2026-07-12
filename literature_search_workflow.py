#!/usr/bin/env python3
"""
医学文献检索工作流 v2.0

基于经典论文:
- PRISMA 声明 (PMID: 19621072)
- Cochrane 检索手册
- PRESS 指南 (PMID: 26931281)

整合:
- 迭代检索 (不同检索词扩大范围)
- 参考文献追溯 (Crossref API)
- 引文追踪 (Semantic Scholar API)
"""

from iterative_literature_search import IterativeLiteratureSearch
from reference_tracker import ReferenceTracker
from typing import List, Dict
import json
from datetime import datetime


class LiteratureSearchWorkflow:
    """医学文献检索完整工作流"""

    def __init__(self):
        self.searcher = IterativeLiteratureSearch()
        self.tracker = ReferenceTracker()
        self.report = {}

    def search(self, topic: str, exclude_terms: List[str] = None,
               max_iterations: int = 4, max_results: int = 50,
               use_reference_tracking: bool = True) -> Dict:
        """
        完整文献检索工作流

        Args:
            topic: 检索主题
            exclude_terms: 排除词列表
            max_iterations: 迭代次数
            max_results: 最大结果数
            use_reference_tracking: 是否启用参考文献追溯

        Returns:
            完整检索结果
        """

        print("="*70)
        print(f"医学文献检索工作流 v2.0")
        print(f"检索主题：{topic}")
        print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()

        # 初始化报告
        self.report = {
            'topic': topic,
            'search_date': datetime.now().strftime('%Y-%m-%d'),
            'steps': {},
            'results': {
                'iterative_papers': [],
                'reference_papers': [],
                'citation_papers': [],
                'all_papers': [],
            },
            'statistics': {},
        }

        # ========== 第 1 步：迭代检索 ==========
        print("【第 1 步】迭代检索 (使用不同检索词扩大范围)")
        print("-"*70)

        papers = self.searcher.search_iterative(
            topic=topic,
            max_iterations=max_iterations,
            max_results=max_results,
        )

        self.report['results']['iterative_papers'] = papers
        self.report['statistics']['iterative_count'] = len(papers)

        print(f"\n迭代检索完成：{len(papers)} 篇")
        print()

        # 用户筛查 (模拟，取前 20 篇作为用户确认)
        user_confirmed = papers[:20] if len(papers) >= 20 else papers
        print(f"用户筛查确认：{len(user_confirmed)} 篇")
        print()

        # ========== 第 2 步：参考文献追溯 ==========
        if use_reference_tracking:
            print("【第 2 步】参考文献追溯 (Crossref API)")
            print("-"*70)

            high_cited_refs = []
            for i, paper in enumerate(user_confirmed[:5]):  # 取前 5 篇
                pmid = paper.get('pmid')
                if pmid:
                    print(f"  [{i+1}/5] PMID {pmid}")
                    refs = self.tracker.get_references(pmid)
                    for ref in refs:
                        ref['source'] = 'reference'
                        high_cited_refs.append(ref)

            self.report['results']['reference_papers'] = high_cited_refs
            self.report['statistics']['reference_count'] = len(high_cited_refs)

            print(f"\n参考文献追溯完成：{len(high_cited_refs)} 篇")
            print()
        else:
            high_cited_refs = []
            print("【第 2 步】跳过参考文献追溯")
            print()

        # ========== 第 3 步：引文追踪 ==========
        if use_reference_tracking and high_cited_refs:
            print("【第 3 步】引文追踪 (Semantic Scholar API)")
            print("-"*70)

            # 找到被引最多的参考文献
            if high_cited_refs:
                landmark_ref = high_cited_refs[0]
                print(f"  开创性文献：{landmark_ref.get('title', 'N/A')[:60]}...")

                citing_papers = self.tracker.find_citing_papers(
                    landmark_ref.get('pmid', ''),
                    min_year=2020
                )

                for paper in citing_papers:
                    paper['source'] = 'citation'

                self.report['results']['citation_papers'] = citing_papers
                self.report['statistics']['citation_count'] = len(citing_papers)

                print(f"\n引文追踪完成：{len(citing_papers)} 篇 (2020 年后)")
                print()
            else:
                citing_papers = []
                print("  无开创性文献，跳过引文追踪")
                print()
        else:
            citing_papers = []
            print("【第 3 步】跳过引文追踪")
            print()

        # ========== 第 4 步：整合结果 ==========
        print("【第 4 步】整合结果")
        print("-"*70)

        all_papers = papers + high_cited_refs + citing_papers
        self.report['results']['all_papers'] = all_papers
        self.report['statistics']['total_count'] = len(all_papers)

        print(f"\n总文献数：{len(all_papers)} 篇")
        print(f"  - 迭代检索：{len(papers)} 篇")
        print(f"  - 参考文献：{len(high_cited_refs)} 篇")
        print(f"  - 引用文献：{len(citing_papers)} 篇")
        print()

        # ========== 生成报告 ==========
        print("【第 5 步】生成检索报告")
        print("-"*70)
        self._generate_report(topic)

        print("\n" + "="*70)
        print("检索完成!")
        print("="*70)

        return self.report

    def _generate_report(self, topic: str):
        """生成检索报告"""

        # 保存到文件
        report_file = f"literature_search_report_{topic.replace(' ', '_')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"  报告已保存：{report_file}")

        # 打印统计
        print(f"\n检索统计:")
        for key, value in self.report['statistics'].items():
            print(f"  - {key}: {value}")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    workflow = LiteratureSearchWorkflow()

    # 检索 rTMS 治疗抑郁症
    report = workflow.search(
        topic='rTMS depression',
        exclude_terms=['electroconvulsive', 'ECT'],
        max_iterations=4,
        max_results=50,
        use_reference_tracking=True,
    )

    print(f"\n最终结果:")
    print(f"  总文献数：{report['statistics']['total_count']} 篇")
