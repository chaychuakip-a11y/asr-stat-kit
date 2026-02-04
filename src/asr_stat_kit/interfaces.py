from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def discover(self, directory: str) -> Tuple[str, str]:
        pass

    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        pass

class BaseProcessor(ABC):
    @abstractmethod
    def process(self, pending_df: pd.DataFrame, online_df: pd.DataFrame) -> pd.DataFrame:
        pass

class BaseExporter(ABC):
    @abstractmethod
    def export(self, df: pd.DataFrame, output_path: str) -> None:
        pass

class BaseSearcher(ABC):
    @abstractmethod
    def search(self, input_path: str) -> pd.DataFrame:
        pass
