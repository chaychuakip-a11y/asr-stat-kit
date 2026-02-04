import os
import glob
import zipfile
import re
import pandas as pd
import xlsxwriter
from typing import Tuple, List, Set, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Relative imports for package structure
from .interfaces import BaseLoader, BaseProcessor, BaseExporter, BaseSearcher
from .config import ColumnMapping, StyleConfig, RegexConfig

class ZipDataLoader(BaseLoader):
    def __init__(self, config: ColumnMapping = ColumnMapping(), regex: RegexConfig = RegexConfig()):
        self.config = config
        self.regex = regex

    def discover(self, directory: str = '.') -> Tuple[str, str]:
        zip_files = glob.glob(os.path.join(directory, "*.zip"))
        zip_files.sort(key=os.path.getmtime, reverse=True)
        if len(zip_files) < 2:
            raise FileNotFoundError("Error: Less than 2 zip files found in the directory.")
        return zip_files[0], zip_files[1]

    def load(self, zip_path: str) -> pd.DataFrame:
        target_filename = 'all_result.csv'
        with zipfile.ZipFile(zip_path, 'r') as z:
            candidates = [n for n in z.namelist() if n.endswith(target_filename)]
            if not candidates:
                raise FileNotFoundError(f"File {target_filename} not found in {zip_path}")
            with z.open(candidates[0]) as f:
                try:
                    df = pd.read_csv(f, sep='\t', dtype={self.config.col_filename: str}, index_col=False)
                    if len(df.columns) < 2: raise ValueError
                except:
                    f.seek(0)
                    df = pd.read_csv(f, sep=None, engine='python', dtype={self.config.col_filename: str}, index_col=False)
        
        df.columns = df.columns.str.strip()
        df = self._standardize_columns(df)
        return self._clean_data(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.col_filename not in df.columns:
            for col in df.columns:
                if any(c in str(col).lower() for c in self.config.filename_candidates):
                    first_val = str(df[col].iloc[0]) if not df.empty else ""
                    if not re.match(r'^[\d\.]+$', first_val):
                        df.rename(columns={col: self.config.col_filename}, inplace=True)
                        break
            if self.config.col_filename not in df.columns:
                df.rename(columns={df.columns[0]: self.config.col_filename}, inplace=True)

        if self.config.col_weight not in df.columns:
            for col in df.columns:
                if any(c in str(col).lower() for c in self.config.weight_candidates):
                    df.rename(columns={col: self.config.col_weight}, inplace=True)
                    break
        if self.config.col_weight not in df.columns:
            df[self.config.col_weight] = 1
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.config.col_filename])
        df[self.config.col_filename] = df[self.config.col_filename].astype(str).str.strip()
        df = df[~df[self.config.col_filename].str.match(self.regex.numeric_only, na=False)]
        for col in [self.config.col_sent_rate, self.config.col_word_rate, self.config.col_weight]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df[df[self.config.col_filename].str.lower() != 'avg']

class StatsProcessor(BaseProcessor):
    def __init__(self, config: ColumnMapping = ColumnMapping(), regex: RegexConfig = RegexConfig()):
        self.config = config
        self.regex = regex

    def process(self, pending_df: pd.DataFrame, online_df: pd.DataFrame) -> pd.DataFrame:
        # 1. Prepare Merge Keys
        pending_df['merge_key'] = self._get_merge_key(pending_df[self.config.col_filename])
        online_df['merge_key'] = self._get_merge_key(online_df[self.config.col_filename])
        online_clean = online_df.drop_duplicates(subset=['merge_key'])
        
        # 2. Select Columns
        cols_needed = [self.config.col_filename, 'merge_key', self.config.col_sent_rate, self.config.col_word_rate, self.config.col_weight]
        # Filter strictly for columns that exist
        p_cols = [c for c in cols_needed if c in pending_df.columns]
        o_cols = [c for c in cols_needed if c != self.config.col_filename and c in online_clean.columns]

        # 3. Merge
        # This will create columns like: sent_pending, sent_online, sent_num_pending, sent_num_online
        df_merged = pd.merge(
            pending_df[p_cols], 
            online_clean[o_cols], 
            on='merge_key', 
            suffixes=('_pending', '_online'), 
            how='inner'
        )

        # 4. Fill NaNs
        numeric_targets = [f"{c}_{s}" for c in [self.config.col_sent_rate, self.config.col_word_rate, self.config.col_weight] for s in ['pending', 'online']]
        for col in numeric_targets:
            if col in df_merged.columns: 
                df_merged[col] = df_merged[col].fillna(0)

        # 5. Extract Groups & Calculate Row Diffs
        df_merged[self.config.col_task_group] = df_merged[self.config.col_filename].apply(self._extract_task_group)
        df_merged[self.config.out_sent_diff] = df_merged[self.config.out_sent_pending] - df_merged[self.config.out_sent_online]
        df_merged[self.config.out_word_diff] = df_merged[self.config.out_word_pending] - df_merged[self.config.out_word_online]
        
        return self._reorder_with_group_stats(df_merged)

    def _get_merge_key(self, series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        for p, r in [(self.regex.clean_filename, ''), (self.regex.clean_whitespace, '_'), (self.regex.clean_numeric_suffix, '')]:
            s = s.str.replace(p, r, regex=True)
        return s

    def _extract_task_group(self, filename: str) -> str:
        match = re.search(self.regex.task_group_extraction, filename)
        return match.group(1) if match else "Other"

    def _calculate_weighted_avg(self, rates: pd.Series, weights: pd.Series) -> float:
        total_weight = weights.sum()
        return (rates * weights).sum() / total_weight if total_weight != 0 else 0.0

    def _get_weight_col(self, metric_name: str) -> str:
        """Determines the correct weight column name based on the metric suffix."""
        # Metric is usually like 'sent_pending' or 'word_online'
        # Weight name is usually 'sent_num'
        # Result should be 'sent_num_pending' or 'sent_num_online'
        
        if metric_name.endswith('_pending'):
            return f"{self.config.col_weight}_pending"
        elif metric_name.endswith('_online'):
            return f"{self.config.col_weight}_online"
        else:
            raise ValueError(f"Unknown metric suffix: {metric_name}")

    def _reorder_with_group_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        final_rows = []
        groups = sorted(df[self.config.col_task_group].unique())
        
        # Define the 4 main metrics we calculate averages for
        metrics_to_calc = [
            self.config.out_sent_pending, 
            self.config.out_sent_online, 
            self.config.out_word_pending, 
            self.config.out_word_online
        ]

        for group in groups:
            group_df = df[df[self.config.col_task_group] == group].sort_values(by=self.config.col_filename)
            if group_df.empty: continue
            
            # --- Calculate Group Average ---
            avg_row = {
                self.config.col_filename: f"Average ({group})",
                self.config.col_task_group: group, 
                'is_group_avg': True
            }
            
            for metric in metrics_to_calc:
                weight_col = self._get_weight_col(metric)
                # Safely get series (handle edge case where column might be missing)
                rates = group_df[metric] if metric in group_df else pd.Series(0, index=group_df.index)
                weights = group_df[weight_col] if weight_col in group_df else pd.Series(0, index=group_df.index)
                
                avg_row[metric] = self._calculate_weighted_avg(rates, weights)
            
            # Calculate Diff for Average Row
            avg_row[self.config.out_sent_diff] = avg_row[self.config.out_sent_pending] - avg_row[self.config.out_sent_online]
            avg_row[self.config.out_word_diff] = avg_row[self.config.out_word_pending] - avg_row[self.config.out_word_online]
            
            final_rows.append(pd.DataFrame([avg_row]))
            final_rows.append(group_df)

        # --- Calculate Grand Total ---
        total_row = {
            self.config.col_filename: 'Total', 
            self.config.col_task_group: 'ZZZ_Total', 
            'is_total': True
        }
        
        for metric in metrics_to_calc:
            weight_col = self._get_weight_col(metric)
            rates = df[metric] if metric in df else pd.Series(0, index=df.index)
            weights = df[weight_col] if weight_col in df else pd.Series(0, index=df.index)
            
            total_row[metric] = self._calculate_weighted_avg(rates, weights)

        total_row[self.config.out_sent_diff] = total_row[self.config.out_sent_pending] - total_row[self.config.out_sent_online]
        total_row[self.config.out_word_diff] = total_row[self.config.out_word_pending] - total_row[self.config.out_word_online]
        
        final_rows.append(pd.DataFrame([total_row]))
        
        # --- Final Concatenation ---
        df_final = pd.concat(final_rows, ignore_index=True)
        
        # Columns to keep
        cols = [
            self.config.col_filename, 
            self.config.out_sent_online, self.config.out_word_online, 
            self.config.out_sent_pending, self.config.out_word_pending, 
            self.config.out_sent_diff, self.config.out_word_diff
        ]
        # Preserve metadata flags for styling
        for mc in ['is_group_avg', 'is_total']:
            if mc in df_final.columns: cols.append(mc)
            
        return df_final[cols].fillna(0)

class ExcelExporter(BaseExporter):
    def __init__(self, style_config: StyleConfig = StyleConfig()):
        self.style = style_config

    def export(self, df: pd.DataFrame, output_path: str) -> None:
        print(f"Generating report: {output_path}")
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        export_cols = [c for c in df.columns if c not in ['is_group_avg', 'is_total']]
        df[export_cols].to_excel(writer, sheet_name='Comparison', index=False, header=False, startrow=2)
        workbook, worksheet = writer.book, writer.sheets['Comparison']
        
        fmt_header = workbook.add_format({'bold': True, 'font_color': self.style.color_header_font, 'bg_color': self.style.color_header_bg, 'border': 1, 'align': 'center'})
        fmt_sub = workbook.add_format({'bold': True, 'font_color': self.style.color_subheader_font, 'bg_color': self.style.color_subheader_bg, 'border': 1, 'align': 'center'})
        worksheet.merge_range(0, 0, 1, 0, "Test Set", fmt_header)
        worksheet.merge_range(0, 1, 0, 2, "Online", fmt_header)
        worksheet.write(1, 1, "Sent Acc", fmt_sub); worksheet.write(1, 2, "Word Acc", fmt_sub)
        worksheet.merge_range(0, 3, 0, 4, "Pending", fmt_header)
        worksheet.write(1, 3, "Sent Acc", fmt_sub); worksheet.write(1, 4, "Word Acc", fmt_sub)
        worksheet.merge_range(0, 5, 0, 6, "Diff", fmt_header)
        worksheet.write(1, 5, "Sent Acc", fmt_sub); worksheet.write(1, 6, "Word Acc", fmt_sub)
        
        worksheet.set_column(0, 0, 45); worksheet.set_column(1, 6, 12)
        diff_indices = [5, 6]
        
        for r_idx in range(len(df)):
            row_num = r_idx + 2
            row_data = df.iloc[r_idx]
            is_total, is_avg = row_data.get('is_total', False), row_data.get('is_group_avg', False)
            
            lbl_fmt = workbook.add_format({'border': 1, 'align': 'left'})
            if is_total: lbl_fmt.set_bg_color(self.style.color_total_bg); lbl_fmt.set_bold()
            elif is_avg: lbl_fmt.set_bg_color(self.style.color_group_avg_bg); lbl_fmt.set_bold()
            worksheet.write(row_num, 0, row_data[export_cols[0]], lbl_fmt)
            
            for c_idx in range(1, len(export_cols)):
                val = row_data[export_cols[c_idx]]
                base_fmt = workbook.add_format({'border': 1, 'num_format': '0.00', 'align': 'center'})
                if is_total: base_fmt.set_bg_color(self.style.color_total_bg); base_fmt.set_bold()
                elif is_avg: base_fmt.set_bg_color(self.style.color_group_avg_bg); base_fmt.set_bold()
                
                if c_idx in diff_indices and not is_total:
                    if val > self.style.threshold_positive: base_fmt.set_font_color(self.style.color_green_text); 
                    elif val < self.style.threshold_negative: base_fmt.set_font_color(self.style.color_red_text)
                    if not is_avg: base_fmt.set_bg_color(self.style.color_green_bg if val > self.style.threshold_positive else self.style.color_red_bg if val < self.style.threshold_negative else 'white')
                
                worksheet.write(row_num, c_idx, val, base_fmt)
        worksheet.freeze_panes(2, 1)
        writer.close()

class CorpusSearcher(BaseSearcher):
    """Multi-threaded corpus searcher with detailed provenance."""
    
    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        # Store data as tuples: (lowercase_text, source_file, sheet_name, row, col)
        self.corpus_entries: List[Tuple[str, str, str, int, int]] = []
        self._load_corpus_threaded()

    def _load_corpus_threaded(self):
        print(f"Scanning corpus directory: {self.corpus_dir} ...")
        
        # 1. Discover all Excel files first
        files_to_process = []
        for root, _, fs in os.walk(self.corpus_dir):
            for f in fs:
                if f.endswith(('.xlsx', '.xls')):
                    files_to_process.append(os.path.join(root, f))
        
        if not files_to_process:
            print("No Excel files found in corpus directory.")
            return

        print(f"Found {len(files_to_process)} corpus files. Loading in parallel...")
        
        # 2. Process files in parallel
        # Adjust max_workers based on your CPU (usually os.cpu_count() * 5 for I/O bound tasks)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all file load tasks
            future_to_file = {executor.submit(self._read_excel_file, f): f for f in files_to_process}
            
            for future in as_completed(future_to_file):
                try:
                    entries = future.result()
                    self.corpus_entries.extend(entries)
                except Exception as exc:
                    print(f"Error loading {future_to_file[future]}: {exc}")
                    
        print(f"Corpus loaded: {len(self.corpus_entries)} total cells indexed.")

    def _read_excel_file(self, filepath: str) -> List[Tuple[str, str, str, int, int]]:
        """Worker function to read a single Excel file."""
        local_entries = []
        filename = os.path.basename(filepath)
        
        try:
            # Read all sheets
            xls = pd.ExcelFile(filepath)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                
                # Iterate through dataframe
                for r_idx, row in df.iterrows():
                    for c_idx, val in enumerate(row):
                        if pd.notna(val):
                            text_str = str(val).strip()
                            if text_str:
                                # Store format: (lower_text, filename, sheet, row, col)
                                local_entries.append((
                                    text_str.lower(), 
                                    filename, 
                                    sheet_name, 
                                    r_idx + 1, # 1-based index for humans
                                    c_idx + 1
                                ))
        except Exception:
            # Silently fail for individual bad files to keep pipeline running
            return []
            
        return local_entries

    def search(self, input_path: str) -> pd.DataFrame:
        print(f"Reading input file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            search_terms = [line.strip() for line in f if line.strip()]
        
        print(f"Searching {len(search_terms)} terms against {len(self.corpus_entries)} corpus entries...")
        results = []

        # Optimization: Converting list to DataFrame once might be faster for very large datasets,
        # but for substring search, iterating the list is often simpler unless we use vectorization.
        # Given the requirement for "substring match", simple iteration is robust.
        
        for term in search_terms:
            term_lower = term.lower()
            found = False
            
            # Linear scan (Regex or 'in' check)
            # For massive corpora, this part should ideally be inverted index, 
            # but per instructions, we stick to substring logic.
            for entry_text, fname, sheet, row, col in self.corpus_entries:
                if term_lower in entry_text:
                    results.append({
                        'search_term': term,
                        'status': 'FOUND',
                        'corpus_file': fname,
                        'sheet': sheet,
                        'location': f"R{row}:C{col}",
                        'matched_content': entry_text  # Original text not stored to save RAM, using lower
                    })
                    found = True
            
            if not found:
                 results.append({
                    'search_term': term,
                    'status': 'NOT FOUND',
                    'corpus_file': '', 'sheet': '', 'location': '', 'matched_content': ''
                })

        return pd.DataFrame(results)