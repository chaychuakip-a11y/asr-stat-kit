#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import sys
import argcomplete
from datetime import datetime
from .components import ZipDataLoader, StatsProcessor, ExcelExporter, CorpusSearcher
from .pipeline import EvaluationPipeline

def cmd_compare(args):
    try:
        pipeline = EvaluationPipeline(ZipDataLoader(), StatsProcessor(), ExcelExporter())
        pipeline.run(args.pending, args.online)
    except Exception as e:
        print(f"Pipeline Error: {e}"); sys.exit(1)

def cmd_search(args):
    try:
        searcher = CorpusSearcher(args.corpus)
        df = searcher.search(args.input_file)
        out = f"search_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        df.to_excel(out, index=False)
        print(f"Saved to: {out}")
    except Exception as e:
        print(f"Search Error: {e}"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ASR Stat Kit")
    sub = parser.add_subparsers(dest='command', required=True)
    
    p_cmp = sub.add_parser('compare', help='Compare Versions')
    p_cmp.add_argument('--pending').completer = argcomplete.completers.FilesCompleter
    p_cmp.add_argument('--online').completer = argcomplete.completers.FilesCompleter
    
    p_sch = sub.add_parser('search', help='Search Corpus')
    p_sch.add_argument('input_file').completer = argcomplete.completers.FilesCompleter
    p_sch.add_argument('--corpus', required=True).completer = argcomplete.completers.DirectoriesCompleter
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.command == 'compare': cmd_compare(args)
    elif args.command == 'search': cmd_search(args)

if __name__ == "__main__":
    main()
