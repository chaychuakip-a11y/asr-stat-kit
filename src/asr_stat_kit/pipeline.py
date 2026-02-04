import os
import hashlib
from datetime import datetime
from .interfaces import BaseLoader, BaseProcessor, BaseExporter

class EvaluationPipeline:
    def __init__(self, loader: BaseLoader, processor: BaseProcessor, exporter: BaseExporter):
        self.loader = loader
        self.processor = processor
        self.exporter = exporter

    def run(self, pending_path: str = None, online_path: str = None) -> None:
        if not pending_path or not online_path:
            print("Auto-discovering versions...")
            pending_path, online_path = self.loader.discover()
        
        print(f"Pending: {os.path.basename(pending_path)}")
        print(f"Online : {os.path.basename(online_path)}")

        df_pending = self.loader.load(pending_path)
        df_online = self.loader.load(online_path)
        df_result = self.processor.process(df_pending, df_online)
        
        date_str = datetime.now().strftime("%Y%m%d")
        with open(pending_path, "rb") as f: md5 = hashlib.md5(f.read()).hexdigest()[-4:]
        output_name = f"report_comparison_{date_str}_{md5}.xlsx"
        
        self.exporter.export(df_result, output_name)
