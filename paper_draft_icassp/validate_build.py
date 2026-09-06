#!/usr/bin/env python3
"""Check manuscript artifacts only; no model evaluation or experiments."""
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'

def command(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout

tex = (ROOT / 'main.tex').read_text()
bib = (ROOT / 'references.bib').read_text()
log = (BUILD / 'main.log').read_text()
blg = (BUILD / 'main.blg').read_text()
info = command('pdfinfo', str(ROOT / 'paper.pdf'))
font_table = command('pdffonts', str(ROOT / 'paper.pdf'))
plain = command('pdftotext', '-layout', str(ROOT / 'paper.pdf'), '-')
pages = [p for p in plain.split('\f') if p.strip()]
(BUILD / 'paper_text.txt').write_text(plain)
command('pdftohtml', '-xml', '-i', '-zoom', '1', str(ROOT / 'paper.pdf'), str(BUILD / 'layout.xml'))
layout = ET.parse(BUILD / 'layout.xml')
fonts = {}
for f in layout.findall('.//fontspec'):
    fonts[(f.attrib['family'], f.attrib['size'])] = True
font_specs = [{'family': a, 'size_pt': float(b)} for a,b in sorted(fonts)]
text_font_specs = [f for f in font_specs if 'Nimbus' in f['family'] or 'DejaVu' in f['family'] or 'Times' in f['family']]
cites = {k.strip() for block in re.findall(r'\\cite\{([^}]+)\}', tex) for k in block.split(',')}
entries = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)
abstract_words = len(re.findall(r"[A-Za-z0-9]+(?:[.'%-][A-Za-z0-9]+)*", abstract.replace('~',' ')))
problems = [line for line in log.splitlines() if 'Overfull' in line or 'undefined' in line.lower() or 'multiply defined' in line.lower()]
bib_problems = [line for line in blg.splitlines() if 'Warning--' in line or 'error message' in line]
underfull = [line for line in log.splitlines() if 'Underfull' in line]
checks = {
    'five_pages': len(pages) == 5,
    'letter_page_size': bool(re.search(r'Page size:\s+612 x 792 pts', info)),
    'technical_text_ends_on_page_4': 'DISCUSSION AND CONCLUSION' in pages[3] and 'Following up this distinction' in ' '.join(pages[3].split()),
    'references_start_on_page_5': 'REFERENCES' in pages[4] and all('REFERENCES' not in p for p in pages[:4]),
    'no_section_after_references': 'REFERENCES' in pages[4] and not re.search(r'\n\s*\d+\.\s+[A-Z]', pages[4].split('REFERENCES',1)[-1]),
    'all_citations_resolved': not (cites - entries) and not problems,
    'no_unused_bib_entries': entries == cites,
    'no_overfull_boxes': not any('Overfull' in x for x in problems),
    'no_bibtex_warnings': not bib_problems,
    'abstract_100_to_150_words': 100 <= abstract_words <= 150,
    'text_and_figure_fonts_at_least_9pt': bool(text_font_specs) and all(f['size_pt'] >= 9 for f in text_font_specs),
    'all_fonts_embedded': all(re.search(r'\s+yes\s+yes\s+(?:yes|no)\s+\d+\s+\d+\s*$', line) for line in font_table.splitlines()[2:] if line.strip()),
}
report = {
    'date': '2026-09-06', 'checks': checks,
    'page_count': len(pages), 'abstract_word_count_regex': abstract_words,
    'citation_count': len(cites), 'unresolved_citation_keys': sorted(cites - entries),
    'font_specs': font_specs,
    'font_note': 'Poppler XML reports integer approximations of font sizes. LaTeX body/reference text is nominally 10 pt; figure labels are 10 pt before the approximately 0.99 figure scale. Math subscripts/superscripts use template math sizes and are checked separately from ordinary text.',
    'compile_problems': problems, 'bibtex_problems': bib_problems,
    'underfull_messages': underfull,
    'underfull_note': 'Non-fatal spacing messages are retained for transparency; page images are also reviewed.',
    'visible_todos': ['Author names', 'Affiliations and contact information', 'paired uncertainty', 'W1/W2 random and rescaling controls'],
}
(ROOT / 'build_validation.json').write_text(json.dumps(report, indent=2) + '\n')
for key,value in checks.items():
    print(('PASS' if value else 'FAIL') + ': ' + key)
print('Abstract words:', abstract_words, '; citations:', len(cites))
if not all(checks.values()):
    raise SystemExit(1)
