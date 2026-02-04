import re
from dataclasses import dataclass, field
from typing import List

@dataclass
class ColumnMapping:
    """Configuration for CSV column detection and mapping."""
    filename_candidates: List[str] = field(default_factory=lambda: ['filename', 'file', 'task'])
    weight_candidates: List[str] = field(default_factory=lambda: ['sent_num', 'num', 'count'])
    
    col_filename: str = 'filename'
    col_sent_rate: str = 'sent'
    col_word_rate: str = 'word'
    col_weight: str = 'sent_num'
    col_task_group: str = 'task_group'
    
    out_sent_online: str = 'sent_online'
    out_word_online: str = 'word_online'
    out_sent_pending: str = 'sent_pending'
    out_word_pending: str = 'word_pending'
    out_sent_diff: str = 'sent_diff'
    out_word_diff: str = 'word_diff'

@dataclass
class StyleConfig:
    """Configuration for Excel styling."""
    color_header_bg: str = '#2F5597'
    color_header_font: str = 'white'
    color_subheader_bg: str = '#D9E1F2'
    color_subheader_font: str = 'black'
    color_green_text: str = '#006100'
    color_green_bg: str = '#C6EFCE'
    color_red_text: str = '#9C0006'
    color_red_bg: str = '#FFC7CE'
    color_total_bg: str = '#FFFFCC'
    color_group_avg_bg: str = '#E2EFDA'
    threshold_positive: float = 0.0001
    threshold_negative: float = -0.0001

@dataclass
class RegexConfig:
    """Regex patterns for parsing and cleaning."""
    clean_filename: str = r'\.(txt|wav|mp3|pcm)$'
    clean_whitespace: str = r'[\s\-]+'
    clean_numeric_suffix: str = r'_\d{6,}.*'
    task_group_extraction: str = r'task_(.+?)_stat_rlt'
    numeric_only: str = r'^[\d\.]+$'
