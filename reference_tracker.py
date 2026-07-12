#!/usr/bin/env python3
"""
参考文献追溯模块 (基于经典文献检索方法学论文)

基于:
- PRISMA 声明 (PMID: 19621072)
- Cochrane 检索手册
- PRESS 指南 (PMID: 26931281)

API:
- OpenAlex: 完全免费，无需 Key
- Crossref: 免费，无需 Key
- Semantic Scholar: 免费，需要 Key (可选)
"""

import requests
from typing import List, Dict
from datetime import datetime


class ReferenceTracker:
    """参考文献追溯器 (符合 PRISMA/Cochrane 标准)"""

    def __init__(self, cache_file: str = 'api_cache.json'):
        self.openalex_base = "https://api.openalex.org"
        self.crossref_base = "https://api.crossref.org"
        self.semascholar_base = "https://api.semanticscholar.org/graph/v1"
        self.semascholar_key = None  # 可选
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """加载缓存"""
        try:
            import os
            if os.path.exists(self.cache_file):
                import json
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _save_cache(self):
        """保存缓存"""
        try:
            import json
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except:
            pass

    def get_references(self, pmid: str) -> List[Dict]:
        """
        获取参考文献 (通过 Crossref)

        基于 PRISMA 声明第 7 步：参考文献追溯
        """

        # 1. PMID → DOI
        doi = self._get_doi_from_pmid(pmid)
        if not doi:
            print(f"  PMID {pmid}: 无法获取 DOI")
            return []

        # 2. Crossref 获取参考文献
        refs = self._crossref_get_references(doi)

        if refs:
            print(f"  PMID {pmid}: 找到 {len(refs)} 篇参考文献")
        else:
            print(f"  PMID {pmid}: 未找到参考文献")

        return refs

    def find_citing_papers(self, pmid: str, min_year: int = 2020) -> List[Dict]:
        """
        找到引用该文献的近期文献

        基于 Cochrane 手册第 8 步：引文追踪
        使用 Semantic Scholar API (免费)
        """

        # 1. PMID → DOI
        doi = self._get_doi_from_pmid(pmid)
        if not doi:
            return []

        # 2. Semantic Scholar 引文追踪 (免费 API，无需 Key)
        citing = self._semanticscholar_get_citations(doi, min_year)

        if citing:
            print(f"  PMID {pmid}: 找到 {len(citing)} 篇引用文献 ({min_year}年后)")
        else:
            print(f"  PMID {pmid}: 未找到引用文献")

        return citing

    def _semanticscholar_get_citations(self, doi: str, min_year: int = 2020) -> List[Dict]:
        """Semantic Scholar 引文追踪 (免费 API，无需 Key)"""

        try:
            # 通过 DOI 查找文献
            paper_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            params = {
                'fields': 'title,year,authors,venue,citations.title,citations.year,citations.authors,citations.venue',
            }

            response = requests.get(paper_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                citations = data.get('citations', [])

                # 过滤近年文献
                recent_citations = [c for c in citations if c.get('year') and c.get('year') >= min_year]

                # 解析引用文献
                citing_papers = []
                for paper in recent_citations[:20]:  # 限制 20 篇
                    citing_papers.append({
                        'title': paper.get('title', 'N/A'),
                        'author': ', '.join([a.get('name', 'N/A') for a in paper.get('authors', [])[:3]]),
                        'journal': paper.get('venue', 'N/A'),
                        'year': str(paper.get('year', 'N/A')),
                        'doi': None,
                        'pmid': None,
                    })

                return citing_papers

        except Exception as e:
            print(f"  Semantic Scholar 错误：{e}")

        return []

    def _get_doi_from_pmid(self, pmid: str) -> str:
        """PMID 转 DOI (PubMed API)"""

        # 检查缓存
        cache_key = f"pmid2doi_{pmid}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'rettype': 'docsum',
                'retmode': 'json',
            }

            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                paper = data.get('result', {}).get(str(pmid), {})

                # DOI 可能在 articleids 数组里
                doi = paper.get('doi')
                if not doi:
                    articleids = paper.get('articleids', [])
                    for articleid in articleids:
                        if articleid.get('idtype') == 'doi':
                            doi = articleid.get('value')
                            break

                # 保存到缓存
                if doi:
                    self.cache[cache_key] = doi
                    self._save_cache()

                return doi

        except Exception as e:
            print(f"  PMID→DOI 错误：{e}")

        return None

    def _crossref_get_references(self, doi: str) -> List[Dict]:
        """Crossref 获取参考文献 (免费 API)"""

        # 检查缓存
        cache_key = f"crossref_refs_{doi}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = requests.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})
                references = message.get('reference', [])

                # 解析参考文献
                refs = []
                for ref in references:
                    refs.append({
                        'title': ref.get('article-title', 'N/A'),
                        'author': self._format_author(ref.get('author', [])),
                        'journal': ref.get('journal-title', 'N/A'),
                        'year': str(ref.get('year', 'N/A')),
                        'doi': ref.get('DOI', 'N/A'),
                        'pmid': None,  # 后续可通过 DOI 获取
                    })

                # 保存到缓存
                self.cache[cache_key] = refs
                self._save_cache()

                return refs

        except Exception as e:
            print(f"  Crossref 错误：{e}")

        return []

    def _openalex_get_citations(self, doi: str, min_year: int = 2020) -> List[Dict]:
        """OpenAlex 引文追踪 (免费 API，无需 Key)"""

        try:
            # 通过搜索 API 找到引用该 DOI 的文献
            search_url = f"{self.openalex_base}/works"
            params = {
                'filter': f'cites:{doi},publication_year:{min_year}:*',
                'per_page': 20,
            }

            response = requests.get(search_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                # 解析引用文献
                citing_papers = []
                for paper in results:
                    citing_papers.append({
                        'title': paper.get('title', 'N/A'),
                        'author': self._format_author(paper.get('authorships', [])),
                        'journal': paper.get('published_in', 'N/A'),
                        'year': str(paper.get('publication_year', 'N/A')),
                        'doi': paper.get('doi', 'N/A'),
                        'pmid': None,  # 后续可获取
                    })

                return citing_papers

        except Exception as e:
            print(f"  OpenAlex 错误：{e}")

        return []

    def _format_author(self, authors) -> str:
        """格式化作者列表"""

        if not authors:
            return 'N/A'

        if isinstance(authors, list) and len(authors) > 0:
            if isinstance(authors[0], dict):
                # OpenAlex 格式
                names = []
                for author in authors[:3]:  # 只显示前 3 位
                    given = author.get('author', {}).get('display_name', '')
                    names.append(given)
                return ', '.join(names) if names else 'N/A'
            elif isinstance(authors[0], str):
                # Crossref 格式
                return ', '.join(authors[:3])

        return 'N/A'

    def collect_all_references(self, papers: List[Dict], min_citations: int = 2) -> List[Dict]:
        """
        收集所有参考文献，并统计被引频次

        基于 PRESS 指南：检索词扩展应全面
        """

        all_refs = {}
        ref_counts = {}

        print(f"\n收集 {len(papers)} 篇文献的参考文献...")

        for i, paper in enumerate(papers):
            pmid = paper.get('pmid')
            if not pmid:
                continue

            print(f"  [{i+1}/{len(papers)}] PMID {pmid}")
            refs = self.get_references(pmid)

            for ref in refs:
                ref_title = ref.get('title', '')
                if ref_title not in all_refs:
                    all_refs[ref_title] = ref
                    ref_counts[ref_title] = 0
                ref_counts[ref_title] += 1

        # 找到被引≥min_citations 次的参考文献
        high_cited_refs = []
        for title, count in ref_counts.items():
            if count >= min_citations:
                ref = all_refs[title].copy()
                ref['cited_by_count'] = count
                high_cited_refs.append(ref)

        print(f"\n高被引参考文献 (被引≥{min_citations}次): {len(high_cited_refs)} 篇")

        return high_cited_refs


# ==================== 使用示例 ====================

if __name__ == '__main__':
    tracker = ReferenceTracker()

    # 测试 PMID: Cole 2020 SNT 开创性文献
    test_pmid = '32252538'

    print("="*70)
    print(f"测试 PMID: {test_pmid}")
    print("="*70)
    print()

    # 获取参考文献
    print("【参考文献追溯】")
    refs = tracker.get_references(test_pmid)
    print()

    # 获取引用文献
    print("【引文追踪】")
    citing = tracker.find_citing_papers(test_pmid, min_year=2020)
    print()

    print("="*70)
    print("测试完成!")
    print("="*70)
